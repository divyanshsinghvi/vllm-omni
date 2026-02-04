"""Tests for process-scoped GPU memory accounting."""

import os
from unittest import mock

import pytest


class TestGetPhysicalDeviceIndex:
    """Tests for _get_physical_device_index function."""

    def test_no_cuda_visible_devices(self):
        """Without CUDA_VISIBLE_DEVICES, returns local_rank as-is."""
        from vllm_omni.worker.base import _get_physical_device_index

        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            assert _get_physical_device_index(0) == 0
            assert _get_physical_device_index(1) == 1
            assert _get_physical_device_index(5) == 5

    def test_with_cuda_visible_devices(self):
        """With CUDA_VISIBLE_DEVICES, maps logical to physical index."""
        from vllm_omni.worker.base import _get_physical_device_index

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,5"}):
            assert _get_physical_device_index(0) == 2
            assert _get_physical_device_index(1) == 3
            assert _get_physical_device_index(2) == 5

    def test_with_cuda_visible_devices_single_gpu(self):
        """Single GPU in CUDA_VISIBLE_DEVICES."""
        from vllm_omni.worker.base import _get_physical_device_index

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            assert _get_physical_device_index(0) == 3

    def test_local_rank_out_of_range(self):
        """If local_rank exceeds visible devices, returns local_rank."""
        from vllm_omni.worker.base import _get_physical_device_index

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
            # local_rank 5 is out of range (only 2 devices), falls back
            assert _get_physical_device_index(5) == 5

    def test_empty_cuda_visible_devices(self):
        """Empty CUDA_VISIBLE_DEVICES returns local_rank."""
        from vllm_omni.worker.base import _get_physical_device_index

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            assert _get_physical_device_index(0) == 0

    def test_cuda_visible_devices_with_spaces(self):
        """Handles spaces in CUDA_VISIBLE_DEVICES."""
        from vllm_omni.worker.base import _get_physical_device_index

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": " 2 , 3 , 5 "}):
            assert _get_physical_device_index(0) == 2
            assert _get_physical_device_index(1) == 3


class TestGetProcessGpuMemory:
    """Tests for _get_process_gpu_memory function."""

    @pytest.mark.skipif(not os.path.exists("/dev/nvidia0"), reason="No NVIDIA GPU available")
    def test_returns_memory_for_current_process(self):
        """Should return non-negative memory for current process with GPU context."""
        import torch

        from vllm_omni.worker.base import _get_process_gpu_memory

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Allocate some GPU memory to ensure process has a context
        device = torch.device("cuda:0")
        tensor = torch.zeros(1000, 1000, device=device)

        memory = _get_process_gpu_memory(0)
        assert memory >= 0

        # Clean up
        del tensor
        torch.cuda.empty_cache()

    def test_returns_zero_on_nvml_error(self):
        """Should return 0 and log warning on NVML error."""
        from vllm_omni.worker.base import _get_process_gpu_memory

        # Mock nvmlInit to raise an error
        with mock.patch("vllm_omni.worker.base.nvmlInit", side_effect=Exception("mock error")):
            memory = _get_process_gpu_memory(0)
            assert memory == 0

    def test_returns_zero_when_process_not_found(self):
        """Should return 0 when current process not in GPU process list."""
        from vllm_omni.worker.base import _get_process_gpu_memory

        # Mock to return empty process list
        with (
            mock.patch("vllm_omni.worker.base.nvmlInit"),
            mock.patch("vllm_omni.worker.base.nvmlShutdown"),
            mock.patch("vllm_omni.worker.base.nvmlDeviceGetHandleByIndex"),
            mock.patch("vllm_omni.worker.base.nvmlDeviceGetComputeRunningProcesses", return_value=[]),
        ):
            memory = _get_process_gpu_memory(0)
            assert memory == 0
