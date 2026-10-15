"""hf_lock: a leaked init_empty_weights-style patch is undone by hf_load (CPU only)."""

import threading

import torch
import torch.nn as nn

from hypersplat.stream import hf_lock
from hypersplat.stream.hf_lock import hf_load


def _meta_register_parameter(module, name, param):
    hf_lock._TORCH_REGISTER_PARAMETER(module, name, param)
    if param is not None:
        module._parameters[name] = nn.Parameter(module._parameters[name].to("meta"))


def test_leaked_patch_is_restored_and_modules_build_on_cpu():
    nn.Module.register_parameter = _meta_register_parameter  # what an overlapping load leaves behind
    try:
        assert nn.Linear(2, 2).weight.device.type == "meta"
        with hf_load():
            lin = nn.Linear(2, 2)
        assert lin.weight.device.type == "cpu"
        assert nn.Module.register_parameter is hf_lock._TORCH_REGISTER_PARAMETER
    finally:
        nn.Module.register_parameter = hf_lock._TORCH_REGISTER_PARAMETER


def test_lock_serialises_threads():
    order = []
    inside = threading.Event()
    release = threading.Event()

    def holder():
        with hf_load():
            order.append("a-in")
            inside.set()
            release.wait(5)
            order.append("a-out")

    t = threading.Thread(target=holder)
    t.start()
    inside.wait(5)
    def second():
        with hf_load():
            order.append("b-in")

    waiter = threading.Thread(target=second)
    waiter.start()
    waiter.join(0.2)
    assert "b-in" not in order  # blocked while the first load holds the lock
    release.set()
    t.join(5)
    waiter.join(5)
    assert order == ["a-in", "a-out", "b-in"]
    with hf_load():  # lock is free again
        pass
