from __future__ import annotations

import logging
import os
import socket
from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return int(dist.get_rank())
    return 0


def get_world_size() -> int:
    if is_distributed():
        return int(dist.get_world_size())
    return 1


def is_rank0() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def get_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0") or "0")
    except Exception:
        return 0


def silence_logging_if_needed(*, is_rank0: bool) -> None:
    if is_rank0:
        return
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.propagate = False
    logging.disable(logging.CRITICAL)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def parse_cuda_devices(device: Any) -> list[int]:
    if device is None:
        return []
    s = str(device).strip()
    if not s:
        return []
    if "," not in s and " " not in s:
        return []

    s = s.replace(" ", ",")
    raw_parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in raw_parts:
        if p.startswith("cuda:"):
            p = p.split("cuda:", 1)[1].strip()
        if p.startswith("cuda"):
            return []
        if p.isdigit():
            out.append(int(p))
            continue
        if ":" in p and p.split(":", 1)[1].strip().isdigit():
            out.append(int(p.split(":", 1)[1].strip()))
            continue
        return []
    return out


def ddp_setup(*, world_size: int) -> None:
    if int(world_size) <= 1 or is_distributed():
        return
    if not torch.cuda.is_available():
        raise RuntimeError("DDP setup requires CUDA for multi-GPU training")
    rank = int(os.environ.get("RANK", "0") or "0")
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=int(world_size))


def ddp_cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def broadcast_int(value: int, *, device: torch.device) -> int:
    if not is_distributed():
        return int(value)
    t = torch.tensor([int(value)], dtype=torch.int64, device=device)
    dist.broadcast(t, src=0)
    return int(t.item())


__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_rank0",
    "barrier",
    "get_local_rank",
    "silence_logging_if_needed",
    "find_free_port",
    "parse_cuda_devices",
    "ddp_setup",
    "ddp_cleanup",
    "broadcast_int",
]

