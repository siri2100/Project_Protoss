"""Differentiable SPARC on joint speed profiles; loss = -SPARC.

Reference: https://github.com/siva82kb/SPARC/blob/master/scripts/smoothness.py
Frequency support is selected on detached spectra. Inside each fixed support,
FFT, normalization and arc length all retain gradients (piecewise differentiable).
"""

import math

import torch


def spectral_arc_length(speed, dt, padlevel=4, cutoff_hz=10., amplitude_threshold=.05):
    """Return one negative SPARC score per nonnegative speed profile [..., T].

    Zero-speed profiles have score 0 and zero gradient (a training convention;
    the evaluation metric may instead report stationary trajectories as NaN).
    """
    if speed.shape[-1] < 3:
        raise ValueError("SPARC requires at least three speed samples")
    if dt <= 0 or not math.isfinite(dt) or padlevel < 0 or not isinstance(padlevel, int):
        raise ValueError("Invalid SPARC dt/padlevel")
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0 or not 0 < amplitude_threshold < 1:
        raise ValueError("Invalid SPARC cutoff/amplitude threshold")
    if not torch.isfinite(speed).all() or (speed < 0).any():
        raise ValueError("SPARC speed must be finite and nonnegative")
    nfft = 2 ** (math.ceil(math.log2(speed.shape[-1])) + padlevel)
    spectrum = torch.fft.rfft(speed, n=nfft).abs()
    peak = spectrum.amax(dim=-1, keepdim=True)
    moving = peak.detach().squeeze(-1) > 1e-12
    normalized = spectrum / peak.clamp_min(1e-12)
    frequencies = torch.fft.rfftfreq(nfft, d=dt, device=speed.device)
    bins = torch.arange(normalized.shape[-1], device=speed.device)
    selected = (normalized.detach() >= amplitude_threshold) & (frequencies <= cutoff_hz)
    last = torch.where(selected, bins, 0).amax(dim=-1)
    # A support with only DC has arc length 0, as the empty reference sum does.
    width = last.clamp_min(1).to(speed.dtype)
    delta = normalized[..., 1:] - normalized[..., :-1]
    lengths = torch.sqrt(width[..., None].reciprocal().square() + delta.square())
    active = bins[:-1] < last[..., None]
    arc = torch.where(active, lengths, 0.).sum(dim=-1)
    return torch.where(moving, -arc, arc * 0.)


def masked_joint_sparc(joints, mask, dt, padlevel=4, cutoff_hz=10., amplitude_threshold=.05):
    """Mean SPARC across valid (batch, joint) profiles without bridging mask gaps.

    Each contiguous run needs >=4 positions. Runs within one joint profile are
    averaged first, then profiles are averaged so gaps do not reweight joints.
    """
    if joints.ndim != 3 or mask.shape != joints.shape or joints.shape[-1] != 7:
        raise ValueError("Joint SPARC expects matching (B, H, 7) positions/mask")
    values = torch.where(mask.bool(), joints, 0.).transpose(1, 2).reshape(-1, joints.shape[1])
    valid = mask.bool().transpose(1, 2).reshape_as(values).detach().cpu().tolist()
    groups = {}
    for profile, flags in enumerate(valid):
        start = None
        for index, flag in enumerate([*flags, False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                if index - start >= 4:
                    groups.setdefault(index - start, []).append((profile, values[profile, start:index]))
                start = None
    if not groups:
        raise ValueError("Model-S batch has no valid four-point joint runs for SPARC")
    sums = values.new_zeros(values.shape[0])
    counts = values.new_zeros(values.shape[0])
    for runs in groups.values():
        positions = torch.stack([run for _, run in runs])
        speeds = (positions[:, 1:] - positions[:, :-1]).abs() / dt
        scores = spectral_arc_length(speeds, dt, padlevel, cutoff_hz, amplitude_threshold)
        indices = torch.tensor([index for index, _ in runs], device=values.device)
        sums = sums.index_add(0, indices, scores)
        counts = counts.index_add(0, indices, torch.ones_like(scores))
    return (sums[counts > 0] / counts[counts > 0]).mean()
