from __future__ import annotations

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


__all__ = ["tqdm"]

