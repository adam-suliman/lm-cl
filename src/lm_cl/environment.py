from __future__ import annotations

import platform
from typing import Any

import torch


def inspect_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )

    capabilities = [
        (props.major, props.minor)
        for props in (
            torch.cuda.get_device_properties(index)
            for index in range(torch.cuda.device_count())
        )
    ]
    precision = {
        "fp32": True,
        "fp16_cuda_native": any(capability >= (5, 3) for capability in capabilities),
        "bf16_cuda_native": any(capability >= (8, 0) for capability in capabilities),
    }

    cuda_backends = torch.backends.cuda
    sdpa = {
        "api_available": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        "flash_backend_enabled": (
            bool(cuda_backends.flash_sdp_enabled()) if cuda_available else False
        ),
        "flash_hardware_supported": any(
            capability >= (8, 0) for capability in capabilities
        ),
        "memory_efficient_backend_enabled": (
            bool(cuda_backends.mem_efficient_sdp_enabled()) if cuda_available else False
        ),
        "math_backend_enabled": bool(cuda_backends.math_sdp_enabled()),
    }
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "supported_precision_modes": precision,
        "sdpa": sdpa,
    }
