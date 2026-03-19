"""
Generate phase maps and .pt files from images for PLM/SLM upload.

Usage examples (PowerShell):

python .\scripts\generate_plm_phasemaps.py \
  --images example\asset\PADO_letter.PNG example\asset\PADO_logo.png \
  --depths 60 120 \
  --outdir out_phasemaps \
  --dim 1920 1080 \
  --pitch 7.56um \
  --wvl 532nm

python .\scripts\generate_plm_phasemaps.py --images .\example\asset\PADO_letter.PNG .\example\asset\PADO_logo.png --depths 60 120 --outdir out_phasemaps --dim 1920 1080 --pitch 7.56um --wvl 532nm

This script uses `pado.light.Light` and `pado.propagator.Propagator` to:
- load images into `Light` objects (optionally add random phase),
- propagate each target plane to the SLM plane,
- sum complex fields (superposition) and generate a phase-only map,
- save an 8-bit phase PNG and a `.pt` file containing normalized phase tensor.

Outputs:
- `out_phasemaps/slm_phase.png`  — 8-bit phase map (0..255 maps to 0..2π)
- `out_phasemaps/slm_phase.pt`   — torch.float32 tensor with values in [0,1]

"""
import argparse
import os
from pathlib import Path
import torch
from PIL import Image

import pado
from pado.math import mm, um, nm


def parse_pitch(pitch_str: str) -> float:
    # allow values like 6.4um, 6.4*um, or numeric in meters
    s = pitch_str.strip()
    if s.endswith('um'):
        return float(s[:-2]) * um
    if s.endswith('mm'):
        return float(s[:-2]) * mm
    if s.endswith('nm'):
        return float(s[:-2]) * nm
    # support direct float meters
    return float(s)


def parse_wvl(wvl_str: str) -> float:
    s = wvl_str.strip()
    if s.endswith('nm'):
        return float(s[:-2]) * nm
    if s.endswith('um'):
        return float(s[:-2]) * um
    return float(s)


def save_phase_outputs(phase_tensor: torch.Tensor, out_dir: Path, base_name: str = 'slm_phase', channel: int = 0, save_8bit: bool = True):
    """Save phase tensor as 8-bit PNG and .pt file.

    phase_tensor expected range: [0, 2*pi) or radians; we will normalize to [0,1]
    phase_tensor shape: [B, Ch, H, W] or [Ch, H, W] or [H, W]
    """
    # Ensure CPU tensor
    t = phase_tensor.detach().cpu()
    # collapse batch and channel selection
    if t.dim() == 4:
        # [B,Ch,H,W] -> take first batch
        t = t[0]
    if t.dim() == 3:
        # [Ch,H,W] -> pick requested channel
        if t.size(0) > channel:
            t = t[channel]
        else:
            t = t[0]

    # At this point t is [H,W]
    # Normalize to [0,1] where 1 -> 2*pi
    two_pi = 2 * torch.pi
    # If phase values appear already in [0,2pi), reduce mod
    t = (t + two_pi) % two_pi
    t_norm = (t / two_pi).clamp(0.0, 1.0).to(torch.float32)

    out_dir.mkdir(parents=True, exist_ok=True)

    if save_8bit:
        arr8 = (t_norm * 255.0).to(torch.uint8).numpy()
        img = Image.fromarray(arr8)
        img_path = out_dir / (base_name + '.png')
        img.save(img_path)
        print(f'Saved 8-bit phase PNG: {img_path}')

    pt_path = out_dir / (base_name + '.pt')
    torch.save(t_norm, pt_path)
    print(f'Saved .pt normalized tensor: {pt_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate phase maps and .pt files from images for PLM upload.')
    parser.add_argument('--images', nargs='+', required=True, help='Input image paths (one per depth)')
    parser.add_argument('--depths', nargs='+', required=True, type=float, help='Depths in mm matching images (e.g., 60 120 250)')
    parser.add_argument('--outdir', default='out_phasemaps', help='Output directory')
    parser.add_argument('--dim', nargs=2, type=int, default=[2048, 2048], help='Array size R C')
    parser.add_argument('--pitch', default='6.4um', help='SLM pixel pitch (e.g., 6.4um, 7.2um, or meters as float)')
    parser.add_argument('--wvl', default='532nm', help='Wavelength (e.g., 532nm)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--random_phase', action='store_true', help='Add random phase to loaded images')
    parser.add_argument('--channel', type=int, default=0, help='Channel index to save when multi-channel result produced')
    parser.add_argument('--slm_8bit', action='store_true', help='Also produce 8-bit PNG for SLM upload (default true)')
    args = parser.parse_args()

    images = args.images
    depths_mm = args.depths
    if len(images) != len(depths_mm):
        raise SystemExit('Number of images must match number of depths')

    out_dir = Path(args.outdir)
    R, C = args.dim
    pitch = parse_pitch(args.pitch)
    wvl = parse_wvl(args.wvl)
    dim = (1, 1, R, C)
    device = args.device

    prop = pado.propagator.Propagator('ASM')

    slm_fields = []
    for img_path, d_mm in zip(images, depths_mm):
        z = d_mm * mm
        L = pado.light.Light(dim, pitch, wvl, device=device)
        print(f'Loading image {img_path} -> depth {d_mm} mm (z={z} m)')
        L.load_image(image_path=img_path, random_phase=args.random_phase)
        L_at_slm = prop.forward(L, z, linear=True, band_limit=True)
        fld = L_at_slm.get_field()
        slm_fields.append(fld)

    # Sum complex fields (superposition)
    print('Summing complex fields at SLM plane...')
    total_field = slm_fields[0]
    for f in slm_fields[1:]:
        total_field = total_field + f

    # Extract phase and map to [0,2pi)
    slm_phase = torch.angle(total_field)
    slm_phase = (slm_phase + 2 * torch.pi) % (2 * torch.pi)

    save_phase_outputs(slm_phase, Path(out_dir), base_name='slm_phase', channel=args.channel, save_8bit=args.slm_8bit)

    # Optionally produce per-image phase maps (useful for per-plane SLM uploads)
    for idx, fld in enumerate(slm_fields):
        phase_i = torch.angle(fld)
        phase_i = (phase_i + 2 * torch.pi) % (2 * torch.pi)
        save_phase_outputs(phase_i, Path(out_dir), base_name=f'slm_phase_plane_{idx}', channel=args.channel, save_8bit=args.slm_8bit)

    print('Done.')


if __name__ == '__main__':
    main()
