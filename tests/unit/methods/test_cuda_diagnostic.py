"""Tests for the GPU diagnostic used by the YOLO method."""

import sys
import types

import pytest

from bubble_analyser.methods.yolo_method import describe_cuda_problem


def _fake_torch(
    cuda_build: str | None = "12.6",
    available: bool = True,
    capability: tuple[int, int] = (6, 1),
    arch_list: tuple[str, ...] = ("sm_50", "sm_61", "sm_75", "sm_86"),
    name: str = "NVIDIA Quadro P1000",
) -> types.SimpleNamespace:
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda _index: capability,
        get_device_name=lambda _index: name,
        get_arch_list=lambda: list(arch_list),
    )
    return types.SimpleNamespace(__version__="2.7.0", version=types.SimpleNamespace(cuda=cuda_build), cuda=cuda)


def test_missing_pytorch_is_reported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError

    assert "not installed" in str(describe_cuda_problem())


def test_cpu_only_build_is_reported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_build=None))

    assert "CPU-only" in str(describe_cuda_problem())


def test_missing_gpu_or_driver_is_reported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))

    assert "cannot see an NVIDIA GPU" in str(describe_cuda_problem())


def test_gpu_architecture_dropped_by_the_build_is_reported(monkeypatch: pytest.MonkeyPatch):
    """A Pascal GPU (sm_61) with a build that only ships Turing and newer kernels."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_build="12.8", arch_list=("sm_75", "sm_86", "sm_120")))

    message = str(describe_cuda_problem())

    assert "Quadro P1000" in message
    assert "not supported" in message


def test_supported_gpu_reports_no_problem(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    assert describe_cuda_problem() is None
