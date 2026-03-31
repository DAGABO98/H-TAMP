import torch
from torch import Tensor


class TriangularCausalMask:
    def __init__(self, B: int, L: int, device: str = "cpu") -> None:
        mask_shape = (B, 1, L, L)
        with torch.no_grad():
            self._mask: Tensor = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool),
                diagonal=1,
            ).to(device)

    @property
    def mask(self) -> Tensor:
        return self._mask


class ProbMask:
    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        index: Tensor,
        scores: Tensor,
        device: str = "cpu",
    ) -> None:
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :,
        ].to(device)
        self._mask: Tensor = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> Tensor:
        return self._mask