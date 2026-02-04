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


def _parse_cuda_visible_devices() -> list[str | int]:
    """Parse CUDA_VISIBLE_DEVICES into a list of device identifiers.

    Returns list of integers (physical indices) or strings (UUIDs/MIG IDs).
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        return []

    result: list[str | int] = []
    for item in visible_devices.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            # UUID (GPU-xxx) or MIG ID (MIG-xxx)
            result.append(item)
    return result


def _get_device_handle(device_id: str | int):
    """Get NVML device handle by index or UUID."""
    if isinstance(device_id, int):
        return nvmlDeviceGetHandleByIndex(device_id)
    else:
        from vllm.third_party.pynvml import nvmlDeviceGetHandleByUUID

        return nvmlDeviceGetHandleByUUID(device_id)


def _get_process_gpu_memory(local_rank: int) -> int | None:
    """Get GPU memory used by current process via pynvml.

    Supports CUDA_VISIBLE_DEVICES with integer indices, UUIDs, or MIG IDs.

    Returns:
        Memory in bytes used by this process, or None if NVML unavailable.

    Raises:
        RuntimeError: If device validation fails (invalid index or UUID).
    """
    from vllm.third_party.pynvml import nvmlDeviceGetCount

    my_pid = os.getpid()
    visible_devices = _parse_cuda_visible_devices()

    try:
        nvmlInit()
    except Exception as e:
        logger.warning("NVML init failed, will use profiling fallback: %s", e)
        return None

    try:
        if visible_devices and local_rank < len(visible_devices):
            device_id = visible_devices[local_rank]
            try:
                handle = _get_device_handle(device_id)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get NVML handle for device '{device_id}' (local_rank={local_rank}). "
                    f"Check CUDA_VISIBLE_DEVICES or stage config 'devices' setting."
                ) from e
        else:
            # No CUDA_VISIBLE_DEVICES or local_rank out of range: use index directly
            device_count = nvmlDeviceGetCount()
            if local_rank >= device_count:
                raise RuntimeError(
                    f"Invalid GPU device {local_rank}. Only {device_count} GPU(s) available. "
                    f"Check CUDA_VISIBLE_DEVICES or stage config 'devices' setting."
                )
            handle = nvmlDeviceGetHandleByIndex(local_rank)

        for proc in nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == my_pid:
                return proc.usedGpuMemory
        return 0
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("NVML query failed, will use profiling fallback: %s", e)
        return None
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

        if process_memory is not None:
            # NVML available: use per-process memory
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
        else:
            # NVML unavailable: use profiling data as conservative fallback
            profiled_usage = (
                int(self.model_runner.model_memory_usage)
                + profile_result.torch_peak_increase
                + profile_result.non_torch_increase
            )
            self.available_kv_cache_memory_bytes = max(0, self.requested_memory - profiled_usage)
            logger.debug(
                "Profiling fallback (PID %d, GPU %d): requested=%s, profiled=%s, available=%s",
                os.getpid(),
                self.local_rank,
                format_gib(self.requested_memory),
                format_gib(profiled_usage),
                format_gib(self.available_kv_cache_memory_bytes),
            )
            logger.info_once(
                "Available KV cache memory: %s GiB (profiling fallback)",
                format_gib(self.available_kv_cache_memory_bytes),
                scope="local",
            )

        return int(self.available_kv_cache_memory_bytes)
