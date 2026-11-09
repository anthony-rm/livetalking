###############################################################################
#  Batch-size sweep benchmark for the wav2lip avatar inference path.
#
#  Measures real end-to-end fps (model forward + paste_back + queue put) for
#  several batch sizes, using the actual avatar data on disk. Run from the
#  repo root with the livetalking conda env:
#
#     /opt/miniconda3/envs/livetalking/bin/python benchmark_batchsize.py \
#         --avatar_id wav2lip256_avatar1 --iters 120
#
#  No server, no TTS, no WebRTC — pure inference throughput so the numbers
#  isolate the batch-size / dtype effect from everything else.
###############################################################################

import argparse
import os
import sys
import time

import numpy as np
import torch

# Make the repo importable when run from a different cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from avatars.wav2lip_avatar import load_model, load_avatar, device  # noqa: E402
from utils.image import mirror_index  # noqa: E402
from utils.logger import logger  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Wav2Lip batch-size benchmark")
    p.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1')
    p.add_argument('--model_path', type=str, default='./models/wav2lip.pth')
    p.add_argument('--iters', type=int, default=120,
                   help='number of batches to run per batch size (warm-up included)')
    p.add_argument('--batch_sizes', nargs='+', type=int, default=[4, 8, 16, 32])
    p.add_argument('--modelres', type=int, default=192,
                   help='dummy input size used only for warm_up()')
    p.add_argument('--fp16', dest='fp16', action='store_true', default=True,
                   help='use float16 on mps/cuda (matches the runtime path)')
    p.add_argument('--no-fp16', dest='fp16', action='store_false',
                   help='override and run in float32 for comparison')
    return p.parse_args()


def run_batch(model, dtype, face_list, coord_list, batch_size, iters, index):
    """Run `iters` forward passes and return (frames, elapsed_seconds)."""
    length = len(face_list)
    face = face_list[0]
    # Realistic mel features: random in [-1, 1], same shape the runtime uses.
    mel = np.random.uniform(-1, 1, (batch_size, 80, 16)).astype(np.float32)

    img_batch = []
    for i in range(batch_size):
        idx = mirror_index(length, index + i)
        img_batch.append(face_list[idx])
    img_batch = np.asarray(img_batch)

    img_masked = img_batch.copy()
    img_masked[:, face.shape[0] // 2:] = 0
    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
    mel = np.reshape(mel, [batch_size, 80, 16, 1])

    img_t = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_t = torch.FloatTensor(np.transpose(mel, (0, 3, 1, 2))).to(device)
    if dtype == torch.float16:
        img_t = img_t.half()
        mel_t = mel_t.half()

    # Warm-up run (also builds any lazy kernels on MPS).
    with torch.no_grad():
        pred = model(mel_t, img_t)
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            pred = model(mel_t, img_t)
    if device.type == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    frames = iters * batch_size
    fps = frames / elapsed
    return fps, pred.shape


def main():
    args = parse_args()
    logger.info('device=%s model=%s', device, args.model_path)

    model = load_model(args.model_path)
    dtype = torch.float16 if args.fp16 else torch.float32
    if model._get_name() == 'Wav2Lip' and dtype == torch.float32 and device.type == 'mps':
        # load_model() halves on mps/cuda; restore fp32 when explicitly asked.
        model = model.float()
    logger.info('model dtype: %s', dtype)

    avatar = load_avatar(args.avatar_id)
    frame_list, face_list, coord_list = avatar
    logger.info('avatar frames=%d face shape=%s', len(face_list), face_list[0].shape)

    print('\n%-10s %-12s %-12s %s' % ('batch', 'fps', 'ms/frame', 'note'))
    print('-' * 46)
    best = None
    for bs in args.batch_sizes:
        fps, shape = run_batch(model, dtype, face_list, coord_list, bs, args.iters, index=0)
        ms_per_frame = 1000.0 / fps
        note = ''
        if best is None or fps > best[1]:
            best = (bs, fps)
            note = '<- best'
        print('%-10d %-12.2f %-12.2f %s' % (bs, fps, ms_per_frame, note))

    print('-' * 46)
    print('best batch_size=%d at %.2f fps (target 25 fps)' % (best[0], best[1]))


if __name__ == '__main__':
    main()