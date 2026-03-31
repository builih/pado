"""
Generate PLM phase patterns and reconstruction through the optical pipeline.

Default configuration (matches your request):
- Laser wavelengths (per channel): 450nm, 520nm, 660nm
- Pixel pitch: 10.8um
- Propagation distance: 100mm
- Output tensor shape: (Channels, Multiplex, H, W) = (3, 8, 800, 1280)
- PLM quantization: 4-bit (16 phase levels)
- Multiplexing is pre-allocated, but not yet differentiated: if you pass ONE image, it is replicated across the 8 multiplex slots.

Run (single image):
python .\\scripts\\generate_plm_phasemaps.py --images path\\to\\sensor_input.png --outdir out_phasemaps

This creates a per-run folder:
- out_phasemaps/data_generation_<timestamp>/
and saves:
- `sensor_input.png`
- `phase_pattern.pt` (quantized phase tensor, shape 3x8x800x1280)
- `phase_pattern.png` and `phase_pattern_{r,g,b}.png` (PNG visualization of the 4-bit quantized phases)
- `reconstruction.png` and `reconstruction_{r,g,b}.png`
"""
import argparse
import os
from pathlib import Path
import torch
import numpy as np
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
    # If no unit is provided, interpret "450" as 450nm by default.
    # Heuristic: values > 1e-3 are treated as nm; otherwise treat as meters.
    x = float(s)
    return x * nm if x > 1e-3 else x


def parse_wvls(wvl_strs: list) -> list:
    """Parse one or more wavelength strings (e.g. ['532nm', '638nm', '450nm'])."""
    return [parse_wvl(s) for s in wvl_strs]


def load_intensity(image_path: str, H: int, W: int, device: str, channels: int) -> tuple[torch.Tensor, np.ndarray, str]:
    """Load an image as normalized intensity [0,1] resized to (H, W).

    Returns:
        intensity: float tensor on `device`, shape (Ch, H, W)
        intensity_uint8: uint8 np array, shape (Ch, H, W)
        mode: string describing what was loaded ('rgb' or 'gray')
    """
    img = Image.open(image_path)

    # If user requests RGB channels and the input is RGB-like, keep R/G/B planes.
    if channels == 3:
        # If the image is RGBA, drop alpha.
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            img = img.convert('RGB')

        img = img.resize((W, H), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)
        # Normalize to [0,1] if needed
        if arr.max() > 1.0:
            if arr.max() <= 255.0:
                arr = arr / 255.0
            else:
                arr = arr / arr.max()

        intensity = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device=device, dtype=torch.float32)  # (3,H,W)
        intensity_uint8 = (arr * 255.0).round().clip(0, 255).astype(np.uint8)  # (H,W,3)
        intensity_uint8 = intensity_uint8.transpose(2, 0, 1)  # (3,H,W)
        return intensity, intensity_uint8, 'rgb'

    # Otherwise, fall back to grayscale intensity replicated to `channels`.
    img = img.convert('L')
    img = img.resize((W, H), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)  # (H,W)
    if arr.max() > 1.0:
        if arr.max() <= 255.0:
            arr = arr / 255.0
        else:
            arr = arr / arr.max()
    intensity = torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(0).repeat(channels, 1, 1)  # (Ch,H,W)
    intensity_uint8 = (arr * 255.0).round().clip(0, 255).astype(np.uint8)  # (H,W)
    intensity_uint8 = np.repeat(intensity_uint8[None, ...], channels, axis=0)  # (Ch,H,W)
    return intensity, intensity_uint8, 'gray'


def quantize_phase(t_norm: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize normalized phase [0,1] to 2^bits levels. Returns float in [0,1]."""
    n_levels = 1 << bits  # 2^bits
    t_norm = (t_norm * (n_levels - 1)).round().clamp(0, n_levels - 1) / (n_levels - 1)
    return t_norm.to(torch.float32)


def save_phase_outputs(phase_tensor: torch.Tensor, out_dir: Path, base_name: str = 'slm_phase',
                       channel: int = 0, save_8bit: bool = True,
                       save_png_per_channel: bool = False, phase_bits: int = 8):
    """Save phase tensor as PNG and .pt file.

    phase_tensor expected range: [0, 2*pi) or radians; we will normalize to [0,1].
    phase_bits: PLM bit depth (e.g. 4 for 16 levels). Phase is quantized to 2^phase_bits levels.
    phase_tensor shape: [Ch, B, H, W]. When shape is 4D we save the full tensor as .pt in [Ch, B, H, W] form.
    """
    # Ensure CPU tensor
    t = phase_tensor.detach().cpu()
    two_pi = 2 * torch.pi
    t = (t + two_pi) % two_pi
    t_norm = (t / two_pi).clamp(0.0, 1.0).to(torch.float32)
    # Quantize to PLM bit depth
    t_norm = quantize_phase(t_norm, phase_bits)
    n_levels = 1 << phase_bits

    out_dir.mkdir(parents=True, exist_ok=True)

    # Map normalized [0,1] to 8-bit PNG range: level i -> (i * 255 + (n_levels-1)//2) // (n_levels-1)
    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        return (x * (n_levels - 1)).round().clamp(0, n_levels - 1).to(torch.uint8) * (255 // (n_levels - 1))

    # Full-tensor save: keep [Ch, B, H, W] for .pt
    if t_norm.dim() == 4:
        pt_path = out_dir / (base_name + '.pt')
        torch.save(t_norm, pt_path)
        print(f'Saved .pt normalized tensor: {pt_path}  shape {tuple(t_norm.shape)} (Ch, multiplex, H, W)  {phase_bits}-bit ({n_levels} levels)')
        if save_8bit:
            nch = t_norm.size(0)
            if save_png_per_channel and nch > 1:
                for c in range(nch):
                    arr8 = to_uint8(t_norm[c, 0]).numpy()
                    suffix = ['_r', '_g', '_b'][c] if nch == 3 else f'_c{c}'
                    img_path = out_dir / (base_name + suffix + '.png')
                    Image.fromarray(arr8).save(img_path)
                    print(f'Saved {phase_bits}-bit phase PNG: {img_path}')
            else:
                arr8 = to_uint8(t_norm[channel if channel < nch else 0, 0]).numpy()
                img_path = out_dir / (base_name + '.png')
                Image.fromarray(arr8).save(img_path)
                print(f'Saved {phase_bits}-bit phase PNG: {img_path}')
        return

    # Legacy 2D/3D: save single map
    if t_norm.dim() == 3:
        t_norm = t_norm[channel] if channel < t_norm.size(0) else t_norm[0]
    pt_path = out_dir / (base_name + '.pt')
    torch.save(t_norm, pt_path)
    print(f'Saved .pt normalized tensor: {pt_path}  {phase_bits}-bit ({n_levels} levels)')
    if save_8bit:
        arr8 = to_uint8(t_norm).numpy()
        Image.fromarray(arr8).save(out_dir / (base_name + '.png'))
        print(f'Saved {phase_bits}-bit phase PNG: {out_dir / (base_name + ".png")}')


def main():
    parser = argparse.ArgumentParser(description='Generate PLM phase patterns with 4-bit quantization and reconstruction.')
    parser.add_argument('--images', nargs='+', required=True, help='Input image paths. For the PLM pipeline, usually pass 1 image.')
    parser.add_argument('--depths', nargs='+', type=float, default=[100.0], help='Propagation distances in mm (default: 100)')
    parser.add_argument('--outdir', default='out_phasemaps', help='Output root directory (a new data_generation folder will be created)')
    parser.add_argument('--run_name', default='', help='Optional folder name inside outdir (otherwise uses timestamp)')
    parser.add_argument('--dim', nargs=2, type=int, default=[800, 1280], help='Spatial size H W')
    parser.add_argument('--channels', type=int, default=3, help='Number of color channels (e.g. 3 for RGB); each can have its own wavelength')
    parser.add_argument('--multiplex', type=int, default=8, help='Multiplex slots (output batch dimension). If only 1 image is given, it is replicated across multiplex slots.')
    parser.add_argument('--pitch', default='10.8um', help='Pixel pitch (e.g., 10.8um)')
    parser.add_argument('--wvl', nargs='+', default=['450nm', '520nm', '660nm'], help='Wavelength(s): one value or one per channel')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--random_phase', action='store_true', help='Add random phase to the loaded target field (optional)')
    parser.add_argument('--bits', type=int, default=4, help='PLM phase bit depth (4 => 16 levels). Default: 4')
    parser.add_argument('--channel', type=int, default=0, help='Channel index for single PNG when saving is not per-channel')
    # Legacy / convenience flags
    parser.add_argument('--slm_8bit', action='store_true', help='(legacy) kept for compatibility; PNG outputs are controlled by --no_png')
    parser.add_argument('--single', action='store_true', help='Use only the first image and first depth')
    parser.add_argument('--no_png', action='store_true', help='Disable PNG outputs (still saves .pt)')
    parser.add_argument('--save_png_per_channel', action='store_true', help='Save one PNG per channel (when channels>1).')
    args = parser.parse_args()

    if args.bits < 1 or args.bits > 8:
        raise SystemExit('--bits must be between 1 and 8')

    images = list(args.images)
    depths_mm = list(args.depths)
    channels = int(args.channels)
    multiplex = int(args.multiplex)

    if args.single:
        images = images[:1]
        depths_mm = depths_mm[:1]

    # Resolve/validate image paths early so errors are clear.
    # - Relative paths are resolved from the current working directory (repo root when you run the script).
    # - The error message shows both the raw input and the resolved absolute path.
    resolved_images: list[str] = []
    cwd = Path.cwd()
    for img in images:
        p = Path(img).expanduser()
        if not p.is_absolute():
            p = cwd / p
        p = p.resolve(strict=False)
        if not p.exists():
            raise SystemExit(f"Image not found: {img} (resolved to {p})")
        resolved_images.append(str(p))
    images = resolved_images

    if len(depths_mm) == 1 and len(images) > 1:
        depths_mm = depths_mm * len(images)

    # Pre-allocate multiplex slots: if the user provides only one image/depth, replicate it.
    if multiplex > 1:
        if len(images) == 1 and len(depths_mm) == 1:
            images = images * multiplex
            depths_mm = depths_mm * multiplex
        else:
            if len(images) != multiplex:
                raise SystemExit(f'When --multiplex {multiplex}, expected 1 image (replicated) or exactly {multiplex} images, got {len(images)}')
            if len(depths_mm) == 1:
                depths_mm = depths_mm * multiplex
            elif len(depths_mm) != multiplex:
                raise SystemExit(f'When --multiplex {multiplex}, expected 1 depth (replicated) or exactly {multiplex} depths, got {len(depths_mm)}')
    else:
        if len(images) != len(depths_mm):
            raise SystemExit('Number of images must match number of depths (or provide a single depth to replicate).')

    wvl_parsed = parse_wvls(args.wvl)
    if len(wvl_parsed) == 1:
        wvl = wvl_parsed[0] if channels == 1 else [wvl_parsed[0]] * channels
    else:
        if len(wvl_parsed) != channels:
            raise SystemExit(f'Number of --wvl values ({len(wvl_parsed)}) must be 1 or match --channels ({channels})')
        wvl = wvl_parsed

    H, W = args.dim
    pitch = parse_pitch(args.pitch)
    device = args.device
    prop = pado.propagator.Propagator('ASM')

    out_root = Path(args.outdir)
    if args.run_name:
        run_dir = out_root / args.run_name
    else:
        from datetime import datetime
        run_dir = out_root / f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_png = not args.no_png
    save_png_per_channel = bool(args.save_png_per_channel) or channels > 1

    channel_suffixes = ['r', 'g', 'b'] if channels == 3 else [f'c{c}' for c in range(channels)]

    # Light is (Batch, Channels, H, W) in pado; we create one Light per plane with batch=1 and stack later.
    dim_light = (1, channels, H, W)

    # Cache intensity loading (images might be replicated across multiplex slots).
    image_cache: dict[str, tuple[torch.Tensor, np.ndarray, str]] = {}

    def get_intensity(img_path: str) -> tuple[torch.Tensor, np.ndarray, str]:
        if img_path not in image_cache:
            intensity, intensity_uint8, mode = load_intensity(img_path, H, W, device=device, channels=channels)
            image_cache[img_path] = (intensity, intensity_uint8, mode)
        return image_cache[img_path]

    # Ground truth sensor input (resized intensity image)
    gt_intensity, gt_intensity_uint8, gt_mode = get_intensity(images[0])
    if save_png:
        if channels == 3:
            # Save an RGB preview using extracted intensity planes.
            rgb_arr = np.stack([gt_intensity_uint8[0], gt_intensity_uint8[1], gt_intensity_uint8[2]], axis=-1)  # (H,W,3)
            Image.fromarray(rgb_arr).save(run_dir / 'sensor_input.png')
        else:
            Image.fromarray(gt_intensity_uint8[0]).save(run_dir / 'sensor_input.png')
        for c, suf in enumerate(channel_suffixes):
            Image.fromarray(gt_intensity_uint8[c]).save(run_dir / f'ground_truth_{suf}.png')

        if gt_mode == 'rgb' and channels == 3:
            # If your "RGB" image is actually grayscale duplicated across channels,
            # these diagnostics help confirm why ground_truth_{r,g,b} look similar.
            ch0 = gt_intensity_uint8[0].astype(np.float32).mean()
            ch1 = gt_intensity_uint8[1].astype(np.float32).mean()
            ch2 = gt_intensity_uint8[2].astype(np.float32).mean()
            print(f"Sensor image mode: rgb; mean intensities (r,g,b)=({ch0:.3f},{ch1:.3f},{ch2:.3f})")
        else:
            print(f"Sensor image mode: {gt_mode}")

    # 1) Compute complex fields at the SLM plane for each multiplex slot / target plane
    slm_fields = []
    for img_path, d_mm in zip(images, depths_mm):
        z = d_mm * mm
        L = pado.light.Light(dim_light, pitch, wvl, device=device)

        intensity, _, _ = get_intensity(img_path)  # (Ch,H,W)
        amplitude = torch.sqrt(intensity).unsqueeze(0)  # (B=1,Ch,H,W)

        L.set_amplitude(amplitude)
        if args.random_phase:
            L.set_phase_random()
        else:
            L.set_phase_zeros()

        print(f'Propagating target image {img_path} -> SLM (z={d_mm}mm)')
        L_at_slm = prop.forward(L, z, linear=True, band_limit=True)
        slm_fields.append(L_at_slm.get_field())  # complex tensor [1,Ch,H,W]

    # 2) Combine: multiplex => stack; multi-depth superposition => sum (only when multiplex==1)
    if multiplex > 1:
        stacked = torch.cat(slm_fields, dim=0)  # (B,Ch,H,W)
        slm_phase = torch.angle(stacked)
    else:
        if len(slm_fields) > 1:
            print('Summing complex fields at SLM plane (multi-depth superposition).')
        total_field = slm_fields[0]
        for f in slm_fields[1:]:
            total_field = total_field + f
        slm_phase = torch.angle(total_field)

    slm_phase = (slm_phase + 2 * torch.pi) % (2 * torch.pi)  # [0,2pi)
    slm_phase = slm_phase.permute(1, 0, 2, 3)  # (B,Ch,H,W) -> (Ch,B,H,W)

    # 3) Quantize phase for reconstruction (and for the extra channel-0 PNG)
    two_pi = 2 * torch.pi
    slm_phase_norm = (slm_phase / two_pi).clamp(0.0, 1.0)
    slm_phase_norm_q = quantize_phase(slm_phase_norm, args.bits)
    slm_phase_rad_q = slm_phase_norm_q * two_pi

    # 4) Save phase pattern
    save_phase_outputs(
        slm_phase, run_dir, base_name='phase_pattern',
        channel=args.channel, save_8bit=save_png,
        save_png_per_channel=save_png_per_channel,
        phase_bits=args.bits,
    )

    # Ensure a generic phase_pattern.png exists (channel 0, multiplex slot 0)
    if save_png and channels >= 1:
        n_levels = 1 << args.bits
        level = (slm_phase_norm_q[0, 0] * (n_levels - 1)).round().clamp(0, n_levels - 1).to(torch.uint8).cpu().numpy()
        arr8 = level * (255 // (n_levels - 1))
        Image.fromarray(arr8).save(run_dir / 'phase_pattern.png')

    # 5) Reconstruct image through the pipeline using the quantized phase-only SLM pattern
    depth0_mm = depths_mm[0]
    if any(abs(d - depth0_mm) > 1e-9 for d in depths_mm):
        print('Warning: multiple depths detected; reconstruction uses the first depth only.')

    z_back = -depth0_mm * mm
    L_slm_mask = pado.light.Light((multiplex, channels, H, W), pitch, wvl, device=device)
    phase_bchw = slm_phase_rad_q.permute(1, 0, 2, 3)  # (Ch,B,H,W) -> (B,Ch,H,W)
    L_slm_mask.set_phase(phase_bchw)  # amplitude ones, phase quantized

    print(f'Reconstructing at sensor plane (z={-depth0_mm}mm)...')
    L_recon = prop.forward(L_slm_mask, z_back, linear=True, band_limit=True)
    recon_intensity = L_recon.get_intensity()  # (B,Ch,H,W)

    if save_png:
        # Save multiplex slot 0 reconstruction
        b0 = 0
        rec0 = recon_intensity[b0].detach().cpu()  # (Ch,H,W)
        for c, suf in enumerate(channel_suffixes):
            img = rec0[c]
            mx = float(img.max().item())
            if mx > 0:
                img = img / mx
            rec8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
            Image.fromarray(rec8).save(run_dir / f'reconstruction_{suf}.png')

        # Generic reconstruction.png from channel 0
        mx = float(rec0[0].max().item())
        img0 = rec0[0]
        if mx > 0:
            img0 = img0 / mx
        rec8 = (img0 * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(rec8).save(run_dir / 'reconstruction.png')

    print(f'Done. Outputs saved in: {run_dir}')


if __name__ == '__main__':
    main()
