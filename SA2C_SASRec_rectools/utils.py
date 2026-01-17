from __future__ import annotations

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x

else:
    import torch.distributed as dist

    _tqdm_impl = tqdm

    def tqdm(x, **kwargs):
        if dist.is_available() and dist.is_initialized() and int(dist.get_rank()) != 0:
            return x
        return _tqdm_impl(x, **kwargs)


__all__ = ["tqdm"]

