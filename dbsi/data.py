import torch


def make_frames(x_raw: torch.Tensor, L: int) -> torch.Tensor:
    """
    Overlap-save with hop=L, frame_len=2L (as used in paper).
    x_raw: [B, M, N] real → x_frames: [B, T, M, 2L] real
    """
    B, M, N = x_raw.shape
    T = (N - 2 * L) // L + 1
    idx = (
        torch.arange(2 * L, device=x_raw.device).unsqueeze(0)
        + torch.arange(T, device=x_raw.device).unsqueeze(1) * L
    )
    x_frames = x_raw[:, :, idx]  # [B, M, T, 2L]
    return x_frames.permute(0, 2, 1, 3).contiguous()  # [B, T, M, 2L]


def frame_fft(x_frames: torch.Tensor) -> torch.Tensor:
    """
    Per-frame DFT (unitary, norm='ortho'), matching the W-matrix pathway
    without explicitly storing W01/W10.
    """
    return torch.fft.fft(
        x_frames.to(torch.complex64), n=x_frames.size(-1), dim=-1, norm="ortho"
    )
