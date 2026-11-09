"""
Quick MuseTalk throughput benchmark (MPS/Apple).
Measures: (1) UNet forward+decode total, (2) paste_back/blending CPU cost,
across batch sizes. Mirrors the runtime inference_batch path.
Run: /opt/miniconda3/envs/livetalking/bin/python benchmark_musetalk.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import avatars.base_avatar  # must precede musetalk_avatar: utils/__init__ shadows `utils` pkg
from avatars.musetalk_avatar import load_model, load_avatar, device
from avatars.musetalk.myutil import get_image_blending

def bench(model, avatar, batch_size, iters=12):
    vae, unet, pe, timesteps, audio_processor = model
    (frame_list, mask_list, coord_list, mask_coords_list, latent_list) = avatar
    length = len(latent_list)

    whisper = np.random.uniform(-1, 1, (batch_size, 50, 384)).astype(np.float32)
    latents = torch.cat([latent_list[i % length] for i in range(batch_size)], 0)
    dt = unet.model.dtype

    # --- warmup (builds MPS kernels) ---
    with torch.no_grad():
        af = torch.from_numpy(whisper).to(device=device, dtype=dt)
        af = pe(af)
        pred = unet.model(latents.to(dt), timesteps, encoder_hidden_states=af).sample
        vae.decode_latents(pred)
    if device.type == 'mps': torch.mps.synchronize()

    # --- timed inference loop ---
    with torch.no_grad():
        af = torch.from_numpy(whisper).to(device=device, dtype=dt)
        af = pe(af)
        out = None
        start = time.perf_counter()
        for _ in range(iters):
            pred = unet.model(latents.to(dt), timesteps, encoder_hidden_states=af).sample
            out = vae.decode_latents(pred)
        if device.type == 'mps': torch.mps.synchronize()
        infer_elapsed = time.perf_counter() - start

    # --- timed paste_back (CPU, reuse same frames) ---
    frame = frame_list[0]
    bbox = coord_list[0]
    mc = mask_coords_list[0]
    mask = mask_list[0]
    x1, y1, x2, y2 = bbox
    assert out is not None  # iters >= 1 guarantees assignment above
    res = cv2.resize(np.asarray(out[0], dtype=np.uint8), (x2 - x1, y2 - y1))
    start = time.perf_counter()
    for _ in range(iters):
        get_image_blending(frame, res, bbox, mask, mc)
    blend_elapsed = time.perf_counter() - start

    infer_fps = iters * batch_size / infer_elapsed
    blend_fps = iters * batch_size / blend_elapsed
    return infer_fps, blend_fps

import cv2
def cv2_resize(img, size): return cv2.resize(img, size)

def main():
    print('device=%s' % device)
    model = load_model()
    avatar = load_avatar('anthony007')
    print('latents=%d mask shape=%s coord=%s' % (
        len(avatar[4]), avatar[1][0].shape, avatar[2][0]))
    print('\n%-6s %-14s %-14s %s' % ('batch', 'infer fps', 'pasteback fps', 'note'))
    print('-' * 44)
    best = None
    for bs in (4, 8, 16):
        inf, bl = bench(model, avatar, bs)
        note = '<- best' if (best is None or inf > best[1]) else ''
        if note: best = (bs, inf)
        print('%-6d %-14.2f %-14.2f %s' % (bs, inf, bl, note))
    print('-' * 44)
    print('target: infer>=25 && pasteback>=25 for realtime')

if __name__ == '__main__':
    main()