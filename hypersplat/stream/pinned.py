"""Frozen-model residency on a shared GPU: pinned host masters, zero-copy parking.

A frozen model (online Difix: UNet + VAE, 1.8 GB fp16) only ever needs to *leave* the GPU to
free memory, never to save state, so parking it with ``module.to("cpu")`` - a 1.8 GB
device-to-pageable copy, 0.3-1.1 s per Difix round on the drone run - is wasted work. Here
every parameter and buffer keeps a page-locked CPU master copy:

    park()      point each tensor back at its master: no copy; the GPU storage is released
    to_gpu()    non-blocking host-to-device copies from the pinned masters (full PCIe speed,
                overlappable on a side stream), then point the tensors at the GPU copies

Only for modules whose weights do not change while parked (inference-only).
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch


class PinnedWeights:
    def __init__(self, modules: Iterable[torch.nn.Module]):
        self._slots: List[Tuple[torch.nn.Module, str, bool]] = []   # (owner, name, is_param)
        self._masters: List[torch.Tensor] = []
        self._index: List[int] = []                                  # slot -> master
        seen: Dict[int, int] = {}
        for module in modules:
            for owner in module.modules():
                for name, p in list(owner._parameters.items()):
                    if p is not None:
                        self._add(owner, name, True, p, seen)   # the Parameter: stable id
                for name, b in list(owner._buffers.items()):
                    if b is not None:
                        self._add(owner, name, False, b, seen)
        self.nbytes = sum(m.numel() * m.element_size() for m in {id(m): m for m in self._masters}.values())
        self.device = torch.device("cpu")
        self.park()

    def _add(self, owner, name, is_param, tensor, seen):
        key = id(tensor)
        if key not in seen:   # tied weights share one master
            t = tensor.detach()
            seen[key] = len(self._masters)
            self._masters.append(t.cpu().pin_memory() if torch.cuda.is_available() else t.cpu())
        self._slots.append((owner, name, is_param))
        self._index.append(seen[key])

    def _assign(self, tensors: List[torch.Tensor]):
        for (owner, name, is_param), i in zip(self._slots, self._index):
            if is_param:
                owner._parameters[name].data = tensors[i]
            else:
                owner._buffers[name] = tensors[i]

    def park(self):
        """Back to the pinned masters (no copy); the GPU copies are freed with their refs."""
        self._assign(self._masters)
        self.device = torch.device("cpu")

    def to_gpu(self, device="cuda", stream: Optional[torch.cuda.Stream] = None) -> torch.cuda.Event:
        """Copy the masters to ``device`` (non-blocking, on ``stream`` if given) and point the
        modules at the copies. Returns an event the consuming stream must wait on."""
        device = torch.device(device)
        stream = stream or torch.cuda.current_stream(device)
        with torch.cuda.stream(stream):
            gpu = [m.to(device, non_blocking=True) for m in self._masters]
            done = torch.cuda.Event()
            done.record(stream)
        # The copies were allocated on ``stream``; tell the allocator they are also used on the
        # current stream so their memory is not reused early once that stream frees them
        current = torch.cuda.current_stream(device)
        if stream != current:
            for t in gpu:
                t.record_stream(current)
        self._assign(gpu)
        self.device = device
        return done
