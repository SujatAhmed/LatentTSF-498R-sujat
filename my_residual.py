"""
Latent residual series of a trained LatentTSF forecaster, built the 498R-kdd way.

    R = Z_hat_Y - Z_Y,   Z_hat_Y = TSF(E(X)),   Z_Y = E(Y)        (prediction - ground truth)

E is the frozen pretrained encoder (--autoencoder_path) and TSF is the checkpoint my_train.py saved for the
same command line. The three 498R-kdd steps are run in latent space:
  1. every train / val / test window, in order, stride 1     (498R-kdd: _save_predictions_real_scale)
  2. overlapping windows averaged to one value per time index (498R-kdd: save_as_csv.py)
  3. train + vali + test merged, E(Y) subtracted, date kept   (498R-kdd: fluctuation_calulation.py)

Run it with exactly the flags used for my_train.py (see run_residual.sh), plus these optional ones:
  --tsf_checkpoint PATH   TSF checkpoint (default: <checkpoints>/<setting>/checkpoint.pth, as in my_train.py)
  --residual_dir DIR      output root, files go to <DIR>/<model_id>/                (default: ./residuals/)
  --residual_name NAME    residual CSV name, <NAME>.csv                              (default: <model_id>_residual)
  --save_window_npy 0|1   also save per-window latents {train,val,test}_{pred,true}_<P>.npy   (default: 1)
  --itr_index K           my_train.py iteration to load (the _K suffix of the setting)        (default: 0)

Outputs in <residual_dir>/<model_id>/:
  {train,val,test}_pred_<P>.npy, _true_<P>.npy   per-window Z_hat_Y / Z_Y, shape (windows, P, D)
  {train,vali,test}_predictions.csv              overlap-averaged Z_hat_Y, one row per time index
  <NAME>_merged_predictions.csv                  train + vali + test predictions
  <NAME>_latent_ground_truth.csv                 date + Z_Y = E(Y) on the same rows
  <NAME>.csv                                     date + R = Z_hat_Y - Z_Y (columns z0 .. z{D-1})
  <NAME>_info.json                               setting, checkpoints, row ranges, checks
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from my_utils import args_train, model_dict, set_seed, acquire_device, get_data
from my_AE import get_autoencoder
from my_train import encoder_encode, encoder_decode, build_dec_inp_latent

# (data_provider flag, 498R-kdd CSV split name)
SPLITS = [('train', 'train'), ('val', 'vali'), ('test', 'test')]
# Loaders whose CSV has a date column and whose train / val / test splits are contiguous
SUPPORTED_DATA = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom')
# AutoEncoders whose latent keeps the time axis: (batch, seq_len, d_model)
TIME_AXIS_AE_TYPES = ('MLP', 'MLP_REVIN', 'CNN')


def parse_residual_args():
    """Read the residual-only flags; the rest of the command line goes to my_utils.args_train()."""
    if '-h' in sys.argv or '--help' in sys.argv:
        print(__doc__)
    # allow_abbrev=False: otherwise the training flag --itr would be taken as --itr_index
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--tsf_checkpoint', type=str, default=None)
    parser.add_argument('--residual_dir', type=str, default='./residuals/')
    parser.add_argument('--residual_name', type=str, default=None)
    parser.add_argument('--save_window_npy', type=int, default=1)
    parser.add_argument('--itr_index', type=int, default=0)
    res_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return res_args


def check_args(args):
    problems = []
    if not args.use_latent:
        problems.append('--use_latent is required: the residual series lives in the latent space.')
    if args.features != 'M':
        problems.append('--features M is required (the AutoEncoder encodes all channels).')
    if getattr(args, 'encoder_type', 'AE') != 'AE':
        problems.append('--encoder_type AE is required.')
    if args.ae_type.upper() not in TIME_AXIS_AE_TYPES:
        problems.append(f'--ae_type must be one of {TIME_AXIS_AE_TYPES}: the latent needs a time axis.')
    if args.data not in SUPPORTED_DATA:
        problems.append(f'--data must be one of {SUPPORTED_DATA} (the residual CSV needs a date column).')
    if not args.freeze_encoder:
        problems.append('--freeze_encoder is required (paper recipe): my_train.py never saves encoder weights, '
                        'so only a frozen encoder is the encoder the forecaster was trained with.')
    if not args.load_pretrained_ae or not args.autoencoder_path or not os.path.exists(args.autoencoder_path):
        problems.append(f'--load_pretrained_ae 1 and an existing --autoencoder_path are required '
                        f'(got {args.autoencoder_path}).')
    if args.augmentation_ratio > 0:
        problems.append('--augmentation_ratio must be 0 (augmented training windows do not map to time indices).')
    if problems:
        sys.exit('my_residual.py:\n  ' + '\n  '.join(problems))
    if not args.freeze_decoder:
        print('WARNING: the decoder was trainable during training but my_train.py does not save it; '
              'the decoded MSE/MAE check below uses the pretrained decoder.')


def training_setting(args, itr_index):
    """Checkpoint folder name of my_train.py in latent mode. Keep in sync with my_train.py."""
    encoder_type = getattr(args, 'encoder_type', 'AE')
    mse_weight = getattr(args, 'mse_weight', 10.0)
    cosine_weight = getattr(args, 'cosine_weight', 15.0)
    perceptual_weight = getattr(args, 'perceptual_weight', 0.0)
    reconstruction_weight = getattr(args, 'reconstruction_weight', 0.0)
    ae_loss_type = getattr(args, 'ae_loss', 'MSE').upper()
    use_lradj_value = getattr(args, 'use_lradj', 1)
    return (f"LatentV20_{encoder_type}_{ae_loss_type}_{args.task_name}_{args.model_id}_{args.model}_{args.data}"
            f"_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_st{args.step}"
            f"_batch{args.batch_size}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}"
            f"_df{args.d_ff}_lr{args.learning_rate}_pw{perceptual_weight}_mw{mse_weight}_cw{cosine_weight}"
            f"_rw{reconstruction_weight}_lradj{use_lradj_value}_{args.des}_{itr_index}")


def strip_module_prefix(state_dict):
    """Checkpoints saved from nn.DataParallel prefix every key with 'module.'."""
    if state_dict and all(k.startswith('module.') for k in state_dict):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def load_models(args, tsf_checkpoint, device):
    """Frozen AutoEncoder + trained TSF forecaster, built exactly as in my_train.py (latent mode)."""
    encoder = get_autoencoder(args).float().to(device)
    encoder.load_state_dict(strip_module_prefix(torch.load(args.autoencoder_path, map_location=device)))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # The forecaster works on the d_model latent channels
    original_dims = (args.enc_in, args.dec_in, args.c_out)
    args.enc_in = args.dec_in = args.c_out = args.d_model
    model = model_dict[args.model].Model(args).float()
    args.enc_in, args.dec_in, args.c_out = original_dims
    model.load_state_dict(strip_module_prefix(torch.load(tsf_checkpoint, map_location=device)))
    model.to(device).eval()

    latent_norm = nn.LayerNorm(args.d_model, elementwise_affine=False).to(device) if args.use_latent_norm else None
    return encoder, model, latent_norm


def run_split(args, flag, encoder, model, latent_norm, device, npy_paths=None):
    """Step 1 + 2 for one split: latent prediction of every window, averaged per time index.

    Window w (stride 1) predicts rows [w, w + pred_len) of data_x[seq_len:], so the split yields
    len(data_x) - seq_len time indices.
    """
    data_set, _ = get_data(args, flag=flag)  # its own loader shuffles: iterate in order instead
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    seq_len, pred_len, d_latent = args.seq_len, args.pred_len, args.d_model
    n_windows = len(data_set)
    n_rows = len(data_set.data_x) - seq_len
    pred_sum = np.zeros((n_rows, d_latent))
    true_sum = np.zeros((n_rows, d_latent))
    count = np.zeros(n_rows)

    pred_npy = true_npy = None
    if npy_paths is not None:  # memory-mapped: large datasets do not have to fit in RAM
        pred_npy = open_memmap(npy_paths[0], mode='w+', dtype=np.float32, shape=(n_windows, pred_len, d_latent))
        true_npy = open_memmap(npy_paths[1], mode='w+', dtype=np.float32, shape=(n_windows, pred_len, d_latent))

    sq_err, abs_err, n_err = 0.0, 0.0, 0
    w = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            y_target = batch_y[:, -pred_len:, :]

            # Same latent path as my_train.py: Z_X = E(X), Z_Y = E(Y), Z_hat_Y = TSF(Z_X)
            z_x = encoder_encode(encoder, batch_x, 'AE', latent_norm)
            z_y = encoder_encode(encoder, y_target, 'AE', None)
            dec_inp = build_dec_inp_latent(z_x, args.label_len, pred_len, d_latent, device)
            z_pred = model(z_x, batch_x_mark, dec_inp, batch_y_mark)[:, -pred_len:, :]

            # Decoded forecast vs. Y (scaled space), the metric my_train.py reports at test time
            err = (encoder_decode(encoder, z_pred, 'AE') - y_target).double()
            sq_err += err.pow(2).sum().item()
            abs_err += err.abs().sum().item()
            n_err += err.numel()

            z_pred = z_pred.cpu().numpy()
            z_y = z_y.cpu().numpy()
            b = z_pred.shape[0]
            if pred_npy is not None:
                pred_npy[w:w + b] = z_pred
                true_npy[w:w + b] = z_y

            starts = w + np.arange(b)
            for j in range(pred_len):
                pred_sum[starts + j] += z_pred[:, j, :]
                true_sum[starts + j] += z_y[:, j, :]
                count[starts + j] += 1
            w += b

    if pred_npy is not None:
        pred_npy.flush()
        true_npy.flush()
        del pred_npy, true_npy
    if w != n_windows or not (count > 0).all():
        raise RuntimeError(f'{flag}: {w}/{n_windows} windows seen, {int((count == 0).sum())} time indices uncovered')

    return {
        'data_set': data_set,
        'n_windows': n_windows,
        'n_rows': n_rows,
        'pred': pred_sum / count[:, None],
        'true': true_sum / count[:, None],
        'decoded_mse': sq_err / n_err,
        'decoded_mae': abs_err / n_err,
    }


def pointwise_encoder_gap(encoder, data_set, seq_len, avg_true, device):
    """Max |averaged Z_Y - E(series)|. The MLP encoder is point-wise, so this is ~0 iff rows are aligned."""
    series = torch.from_numpy(np.asarray(data_set.data_x[seq_len:], dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        z_series = encoder_encode(encoder, series, 'AE', None)[0].cpu().numpy()
    return float(np.abs(z_series - avg_true).max())


def load_raw_dates(args):
    """Date column of the raw CSV (local file, else the HuggingFace fallback the data loaders use)."""
    local_fp = os.path.join(args.root_path, args.data_path)
    if os.path.exists(local_fp):
        return pd.read_csv(local_fp, usecols=['date'])['date']
    from data_provider.data_loader import load_dataset, HUGGINGFACE_REPO
    ds = load_dataset(HUGGINGFACE_REPO, name=os.path.splitext(os.path.basename(args.data_path))[0])
    split_name = 'train' if 'train' in ds else list(ds.keys())[0]
    return ds[split_name].to_pandas()['date']


def main():
    res_args = parse_residual_args()
    args = args_train()
    check_args(args)

    set_seed(args.seed)
    if args.use_gpu and args.gpu_type == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available: using the CPU')
        args.use_gpu = False
    device = acquire_device(args)

    # Locate the TSF checkpoint with the training --step, then run every window (stride 1, like 498R-kdd)
    setting = training_setting(args, res_args.itr_index)
    tsf_checkpoint = res_args.tsf_checkpoint or os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    if not os.path.exists(tsf_checkpoint):
        sys.exit(f'my_residual.py: TSF checkpoint not found:\n  {tsf_checkpoint}\n'
                 f'Train first with the same flags (run_train.sh) and without --delete_checkpoint, '
                 f'or pass --tsf_checkpoint.')
    train_step = args.step
    args.step = 1

    print(f'Setting       : {setting}')
    print(f'TSF checkpoint: {tsf_checkpoint}')
    print(f'AE checkpoint : {args.autoencoder_path}')
    encoder, model, latent_norm = load_models(args, tsf_checkpoint, device)

    out_dir = os.path.join(res_args.residual_dir, args.model_id)
    os.makedirs(out_dir, exist_ok=True)
    name = res_args.residual_name or f'{args.model_id}_residual'
    z_cols = [f'z{i}' for i in range(args.d_model)]
    seq_len, pred_len = args.seq_len, args.pred_len

    # ---- Steps 1 + 2: per-split window predictions, averaged to one value per time index ----
    results = {}
    for flag, csv_split in SPLITS:
        npy_paths = None
        if res_args.save_window_npy:
            npy_paths = (os.path.join(out_dir, f'{flag}_pred_{pred_len}.npy'),
                         os.path.join(out_dir, f'{flag}_true_{pred_len}.npy'))
        r = run_split(args, flag, encoder, model, latent_norm, device, npy_paths)
        csv_path = os.path.join(out_dir, f'{csv_split}_predictions.csv')
        pd.DataFrame(r['pred'], columns=z_cols).to_csv(csv_path, index=False)
        print(f'{flag}: windows={r["n_windows"]}, time indices={r["n_rows"]}, '
              f'decoded mse={r["decoded_mse"]:.6f}, mae={r["decoded_mae"]:.6f} -> {csv_path}')
        results[flag] = r

    # ---- Checks: contiguous splits, and (MLP) Z_Y equal to the point-wise encoding of the series ----
    for (prev, _), (cur, _) in zip(SPLITS, SPLITS[1:]):
        if not np.allclose(results[prev]['data_set'].data_x[-seq_len:], results[cur]['data_set'].data_x[:seq_len]):
            raise RuntimeError(f'the {prev} and {cur} splits are not contiguous; cannot merge them into one series')
    pointwise_gap = None
    if args.ae_type.upper() == 'MLP':
        pointwise_gap = max(pointwise_encoder_gap(encoder, results[f]['data_set'], seq_len, results[f]['true'], device)
                            for f, _ in SPLITS)
        print(f'Point-wise encoder check: max |Z_Y - E(series)| = {pointwise_gap:.2e}')
        if pointwise_gap > 1e-4:
            raise RuntimeError('averaged Z_Y does not match the point-wise encoding: time indices are misaligned')
    test = results['test']
    print(f'Decoded test metrics (stride 1): mse={test["decoded_mse"]:.7f}, mae={test["decoded_mae"]:.7f}')
    if train_step == 1:
        print('  (these equal the test mse/mae that my_train.py reported for this checkpoint)')
    else:
        print(f'  (my_train.py tested with --step {train_step}, so its test metrics use fewer windows)')

    # ---- Step 3: merge train + vali + test and subtract the latent ground truth ----
    merged_pred = np.concatenate([results[f]['pred'] for f, _ in SPLITS])
    merged_true = np.concatenate([results[f]['true'] for f, _ in SPLITS])
    start, end = seq_len, seq_len + len(merged_pred)

    dates = load_raw_dates(args)
    if end > len(dates):
        raise RuntimeError(f'the series needs rows [{start}, {end}) but {args.data_path} has {len(dates)} rows')
    date_col = dates.iloc[start:end].reset_index(drop=True)

    merged_path = os.path.join(out_dir, f'{name}_merged_predictions.csv')
    pd.DataFrame(merged_pred, columns=z_cols).to_csv(merged_path, index=False)
    print('CSV files merged successfully..........')

    truth_path = os.path.join(out_dir, f'{name}_latent_ground_truth.csv')
    pd.concat([date_col, pd.DataFrame(merged_true, columns=z_cols)], axis=1).to_csv(truth_path, index=False)
    print('original series length: ', len(merged_true))
    print('predicted series length: ', len(merged_pred))

    residual = merged_pred - merged_true  # prediction - ground truth, as in 498R-kdd
    residual_df = pd.concat([date_col, pd.DataFrame(residual, columns=z_cols)], axis=1)
    residual_path = os.path.join(out_dir, f'{name}.csv')
    residual_df.to_csv(residual_path, index=False)
    print(f'Residual series completed: {residual_path}  shape={residual_df.shape}  '
          f'(original rows [{start}, {end}), mean={residual.mean():.6f}, std={residual.std():.6f})')

    # ---- Bookkeeping for whoever trains on the residual series next ----
    splits_info, row = {}, start
    for flag, csv_split in SPLITS:
        n = results[flag]['n_rows']
        splits_info[csv_split] = {'original_rows': [row, row + n], 'residual_rows': [row - start, row + n - start],
                                  'windows': results[flag]['n_windows']}
        row += n
    info = {
        'formula': 'R = Z_hat_Y - Z_Y = TSF(E(X)) - E(Y)  (prediction - ground truth, as in 498R-kdd)',
        'setting': setting,
        'tsf_checkpoint': tsf_checkpoint,
        'autoencoder_path': args.autoencoder_path,
        'model': args.model,
        'data': args.data,
        'data_path': os.path.join(args.root_path, args.data_path),
        'seq_len': seq_len,
        'pred_len': pred_len,
        'latent_dim': args.d_model,
        'train_step': train_step,
        'residual_stride': 1,
        'columns': ['date'] + z_cols,
        'original_rows': [start, end],
        'splits': splits_info,
        'checks': {
            'splits_contiguous': True,
            'pointwise_encoder_max_abs_diff': pointwise_gap,
            'decoded_metrics': {f: {'mse': results[f]['decoded_mse'], 'mae': results[f]['decoded_mae']}
                                for f, _ in SPLITS},
        },
        'files': {
            'residual': residual_path,
            'latent_ground_truth': truth_path,
            'merged_predictions': merged_path,
        },
    }
    info_path = os.path.join(out_dir, f'{name}_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'Saved {info_path}')

    if args.gpu_type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
