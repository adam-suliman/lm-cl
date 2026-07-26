from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lm_cl.data.storage import atomic_write_json


_NATIVE_STREAM_WORKER_ENV = "_LM_CL_NATIVE_STREAM_WORKER"


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    atomic_write_json(output, value)


def cli_entry(function: Callable[[], None]) -> None:
    try:
        function()
    except (ValueError, TypeError, RuntimeError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def native_streaming_cli_entry(function: Callable[[], None]) -> None:
    """Run native streaming libraries in a worker with controlled termination.

    Hugging Face streaming can leave third-party native callbacks alive until
    CPython finalization. The worker uses ``os._exit`` only after the command
    has returned or raised and both output streams have been flushed. The
    parent process imports no streaming dependency and propagates the worker
    status.
    """
    if os.environ.get(_NATIVE_STREAM_WORKER_ENV) == "1":
        exit_code = 0
        try:
            cli_entry(function)
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 1
        except KeyboardInterrupt:
            exit_code = 130
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        finally:
            try:
                sys.stdout.flush()
            finally:
                sys.stderr.flush()
        os._exit(exit_code)

    environment = os.environ.copy()
    environment[_NATIVE_STREAM_WORKER_ENV] = "1"
    module_name = function.__module__
    module = sys.modules.get(module_name)
    module_spec = getattr(module, "__spec__", None)
    if module_spec is not None and module_spec.name is not None:
        module_name = module_spec.name
    completed = subprocess.run(
        [sys.executable, "-m", module_name, *sys.argv[1:]],
        env=environment,
        check=False,
    )
    exit_code = completed.returncode
    if exit_code < 0:
        exit_code = 128 - exit_code
    raise SystemExit(exit_code)
