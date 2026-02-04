"""Base worker class for vLLM-Omni with process-scoped GPU memory accounting."""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger
from vllm.third_party.pynvml import (
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetHandleByIndex,
    nvmlInit,
    nvmlShutdown,
)
from vllm.utils.mem_utils import format_gib, memory_profiling
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

logger = init_logger(__name__)


def _get_physical_device_index(local_rank: int) -> int:
    """Convert logical device index to physical device index for pynvml."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        try:
            physical_indices = [int(x.strip()) for x in visible_devices.split(",") if x.strip()]
            if local_rank < len(physical_indices):
                return physical_indices[local_rank]
        except (ValueError, IndexError):
            pass
    return local_rank


def _get_process_gpu_memory(local_rank: int) -> int:
    """Get GPU memory used by current process via pynvml."""
    my_pid = os.getpid()
    physical_device = _get_physical_device_index(local_rank)

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(physical_device)
        for proc in nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == my_pid:
                return proc.usedGpuMemory
        return 0
    except Exception as e:
        logger.warning(
            "Failed to get process GPU memory for device %d (physical %d): %s",
            local_rank,
            physical_device,
            e,
        )
        return 0
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass


class OmniGPUWorkerBase(GPUWorker):
    """Base GPU worker for vLLM-Omni with process-scoped memory accounting.

    This class overrides determine_available_memory() to use per-process GPU
    memory tracking via pynvml, allowing multiple stages to initialize
    concurrently on the same GPU without memory accounting interference.
    """

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Process-scoped GPU memory profiling for concurrent stage initialization.

        Uses pynvml to get per-process memory instead of global free memory,
        allowing multiple stages to initialize concurrently on the same GPU.
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            logger.info(
                "Using explicit kv_cache_memory_bytes: %s GiB",
                format_gib(kv_cache_memory_bytes),
            )
            return kv_cache_memory_bytes

        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        process_memory = _get_process_gpu_memory(self.local_rank)
        self.available_kv_cache_memory_bytes = max(0, self.requested_memory - process_memory)

        logger.debug(
            "Process-scoped memory (PID %d, GPU %d): requested=%s, used=%s, available=%s",
            os.getpid(),
            self.local_rank,
            format_gib(self.requested_memory),
            format_gib(process_memory),
            format_gib(self.available_kv_cache_memory_bytes),
        )
        logger.info_once(
            "Available KV cache memory: %s GiB (process-scoped)",
            format_gib(self.available_kv_cache_memory_bytes),
            scope="local",
        )

        return int(self.available_kv_cache_memory_bytes)
