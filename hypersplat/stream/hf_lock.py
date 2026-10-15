"""Serialise model loading across threads.

``from_pretrained`` (transformers / diffusers, via accelerate's ``init_empty_weights``) monkeypatches
``nn.Module.register_parameter`` process-wide so new parameters land on the meta device, and restores
whatever it saw on entry. The stream loads models on several threads (difix preload, depth, latent
encoder) while the trainer builds its own modules, so two overlapping loads can:

  * put parameters of a module built on another thread at the same time on meta, and
  * restore the patch itself on exit, leaking it for the rest of the process.

Every model load and module build that can overlap goes through ``hf_load()``: one re-entrant lock,
and on exit torch's own registration methods are put back if a patch survived.
"""

import logging
import threading
from contextlib import contextmanager

import torch.nn as nn

logger = logging.getLogger(__name__)

# torch's own methods, captured when this module is first imported (before any loading thread starts)
_TORCH_REGISTER_PARAMETER = nn.Module.register_parameter
_TORCH_REGISTER_BUFFER = nn.Module.register_buffer

_LOCK = threading.RLock()


def _restore_registration(where: str) -> None:
    if (nn.Module.register_parameter is not _TORCH_REGISTER_PARAMETER
            or nn.Module.register_buffer is not _TORCH_REGISTER_BUFFER):
        logger.warning(f"[hf_lock] nn.Module registration was left patched ({where}); restoring torch's own")
        nn.Module.register_parameter = _TORCH_REGISTER_PARAMETER
        nn.Module.register_buffer = _TORCH_REGISTER_BUFFER


@contextmanager
def hf_load():
    """Hold the process-wide model-loading lock; registration is torch's own inside and after it."""
    with _LOCK:
        _restore_registration("on entry")
        try:
            yield
        finally:
            _restore_registration("on exit")
