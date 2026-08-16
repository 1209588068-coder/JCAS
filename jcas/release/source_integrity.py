"""Deterministic source snapshots and environment metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from jcas import PROJECT_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_source_archive(paths: tuple[Path, ...], output: Path) -> None:
    """Write a reproducible gzip tar whose members use repository paths."""
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for relative in sorted(paths, key=lambda item: item.as_posix()):
                        data = (PROJECT_ROOT / relative).read_bytes()
                        info = tarfile.TarInfo(relative.as_posix())
                        info.size = len(data)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info, fileobj=io.BytesIO(data))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _command_output(args: list[str]) -> str | None:
    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def environment_summary() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "torch-geometric",
        "scipy",
        "scikit-learn",
        "joblib",
        "pytest",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    requirements = PROJECT_ROOT / "requirements.txt"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _command_output(["git", "status", "--short"]),
        "requirements": {
            "path": "requirements.txt",
            "sha256": sha256_file(requirements),
        },
    }


def verify_sha_manifest_with_archive(
    manifest: Path,
    *,
    source_archive: Path,
) -> int:
    """Verify release entries from the source archive or live asset paths.

    Source members intentionally take precedence over the reorganized working
    tree. Non-source assets, such as checkpoints and result evidence, are read
    from their recorded repository paths.
    """
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid SHA manifest line {line_number}: {manifest}"
            ) from exc
        if len(digest) != 64 or relative in entries:
            raise ValueError(
                f"invalid SHA manifest entry at line {line_number}: {manifest}"
            )
        entries[relative] = digest.lower()
    if not entries:
        raise ValueError(f"empty SHA manifest: {manifest}")

    with tarfile.open(source_archive, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for relative, expected in entries.items():
            if relative in members:
                stream = archive.extractfile(members[relative])
                if stream is None:
                    raise RuntimeError(f"cannot read frozen source member: {relative}")
                actual = hashlib.sha256(stream.read()).hexdigest()
            else:
                path = (PROJECT_ROOT / relative).resolve()
                try:
                    path.relative_to(PROJECT_ROOT)
                except ValueError as exc:
                    raise ValueError(f"release asset escapes repository: {relative}") from exc
                if not path.is_file():
                    raise FileNotFoundError(f"release asset is missing: {relative}")
                actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"release asset SHA-256 mismatch: {relative}")
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a published SHA manifest with its frozen source archive"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-archive", required=True)
    args = parser.parse_args()
    count = verify_sha_manifest_with_archive(
        Path(args.manifest),
        source_archive=Path(args.source_archive),
    )
    print(f"verified release entries: {count}")


if __name__ == "__main__":
    main()
