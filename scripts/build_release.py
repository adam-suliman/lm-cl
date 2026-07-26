from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_release(output: Path) -> dict[str, object]:
    output = output.expanduser().resolve()
    if output == ROOT or output in ROOT.parents:
        raise ValueError("Release output cannot contain or equal the development tree")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Release output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    public_cli_files = {
        "__init__.py",
        "_common.py",
        "_continual_runtime.py",
        "_probe_runtime.py",
        "inspect_environment.py",
        "inspect_manifest.py",
        "inspect_tokenizer.py",
        "launch_experiments.py",
        "materialize_stage.py",
        "prepare_experiment_data.py",
        "resume_continual.py",
        "resume_probe.py",
        "run_experiment_job.py",
        "run_probe.py",
        "train_continual.py",
        "validate_packed_shards.py",
    }
    for path in sorted((ROOT / "src/lm_cl").rglob("*.py")):
        if path.parent == ROOT / "src/lm_cl/cli" and path.name not in public_cli_files:
            continue
        _copy_file(path, output / path.relative_to(ROOT))
    for name in (
        "zyphra_fastmem_a100.yaml",
        "zyphra_fastmem_two_cycle_smoke.yaml",
    ):
        _copy_file(
            ROOT / "configs/experiments" / name,
            output / "configs/experiments" / name,
        )
    for name in ("zyphra_5m.yaml", "zyphra_12m.yaml", "tiny_test.yaml"):
        _copy_file(
            ROOT / "configs/models" / name,
            output / "configs/models" / name,
        )
    for name in ("pyproject.toml", ".gitignore"):
        _copy_file(ROOT / name, output / name)
    release_readme = (
        ROOT / "release/README.md"
        if (ROOT / "release/README.md").is_file()
        else ROOT / "README.md"
    )
    release_docs = (
        ROOT / "release/docs"
        if (ROOT / "release/docs").is_dir()
        else ROOT / "docs"
    )
    _copy_file(release_readme, output / "README.md")
    for name in ("CONFIGURATION.md", "DATA_AND_RESUME.md"):
        _copy_file(
            release_docs / name,
            output / "docs" / name,
        )
    _copy_file(Path(__file__), output / "scripts/build_release.py")
    focused_test = ROOT / "tests/test_public_release.py"
    if focused_test.is_file():
        _copy_file(focused_test, output / "tests/test_public_release.py")

    license_files = [
        path
        for pattern in ("LICENSE*", "COPYING*")
        for path in ROOT.glob(pattern)
        if path.is_file()
    ]
    for path in sorted(set(license_files)):
        _copy_file(path, output / path.name)

    forbidden_parts = {
        "runs",
        "results",
        "checkpoints",
        "__pycache__",
        ".pytest_cache",
        ".venv",
    }
    forbidden_suffixes = {".pt", ".bin", ".pdf"}
    inventory = []
    markdown = []
    local_markers = (
        b"/" + b"home/" + b"admin/",
        b"/" + b"data/" + b"home/" + b"admin/",
        b"HF_" + b"TOKEN=",
    )
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output)
        if forbidden_parts.intersection(relative.parts):
            raise ValueError(f"Forbidden release path: {relative}")
        if path.suffix in forbidden_suffixes or path.name.startswith(
            "events.out.tfevents"
        ):
            raise ValueError(f"Forbidden release artifact: {relative}")
        content = path.read_bytes()
        if any(marker in content for marker in local_markers):
            raise ValueError(f"Machine-local or credential marker in {relative}")
        if path.suffix.lower() == ".md":
            markdown.append(relative.as_posix())
        inventory.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if len(markdown) > 3:
        raise ValueError(f"Release contains excessive Markdown: {markdown}")
    return {
        "release_schema_version": 1,
        "output": str(output),
        "file_count": len(inventory),
        "markdown_files": markdown,
        "license_files": [path.name for path in license_files],
        "inventory": inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the minimal public release tree")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_release(Path(args.output))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
