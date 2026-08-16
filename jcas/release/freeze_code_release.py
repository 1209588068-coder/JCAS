#!/usr/bin/env python3
"""Freeze the post-hoc v6.0.1 code-only directory reorganization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from jcas import PROJECT_ROOT
from jcas.core.poison import sha256_file
from jcas.release.source_integrity import (
    deterministic_source_archive,
    environment_summary,
)


ROOT = PROJECT_ROOT
PACKAGE_VERSION = "v6.0.1_code_only_reorganization"
RELEASE_STATUS = "posthoc_layout_only_models_and_results_unchanged"
DATE_TAG = "20260814"
FORMAL_CONTRACT = Path("record/v6/contracts/v6_freeze_20260814.metadata.json")
FROZEN_ASSET_MANIFEST = Path(
    "record/v6/contracts/v6_frozen_assets_20260814.sha256"
)
ORIGINAL_SOURCE_SNAPSHOT = Path(
    "record/v6/contracts/v6_source_snapshot_20260814.tar.gz"
)
DEFAULT_OUTPUT_DIR = Path("record/v6/contracts/code_layout_v6_0_1")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path("docs/V6_RUNBOOK.md"),
    Path("jcas/__init__.py"),
    Path("jcas/core/__init__.py"),
    Path("jcas/core/graph_splits.py"),
    Path("jcas/core/graph_splits_v5.py"),
    Path("jcas/core/models.py"),
    Path("jcas/core/poison.py"),
    Path("jcas/core/risk_labels.py"),
    Path("jcas/core/shadow_folds.py"),
    Path("jcas/core/strict_shadow_folds.py"),
    Path("jcas/core/trajectory_trigger.py"),
    Path("jcas/workflows/__init__.py"),
    Path("jcas/workflows/evaluator.py"),
    Path("jcas/workflows/fixed_symmetric_manifest.py"),
    Path("jcas/workflows/graph_builder.py"),
    Path("jcas/workflows/poison_manifest.py"),
    Path("jcas/workflows/trainer.py"),
    Path("jcas/release/__init__.py"),
    Path("jcas/release/finalize_v6.py"),
    Path("jcas/release/freeze_code_release.py"),
    Path("jcas/release/freeze_v6.py"),
    Path("jcas/release/metrics_integrity.py"),
    Path("jcas/release/source_integrity.py"),
    Path("tests/test_blackbox_pipeline.py"),
    Path("tests/test_v6_fixed_symmetric_manifest.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the v6.0.1 code-only layout release"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", path)


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / Path(args.output_dir)).resolve()
    output_dir.relative_to(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_manifest = output_dir / f"v6_0_1_code_assets_{DATE_TAG}.sha256"
    source_archive = output_dir / f"v6_0_1_source_snapshot_{DATE_TAG}.tar.gz"
    environment_path = output_dir / f"v6_0_1_environment_{DATE_TAG}.json"
    release_path = output_dir / f"v6_0_1_code_release_{DATE_TAG}.json"
    anchor_path = output_dir / f"v6_0_1_code_release_{DATE_TAG}.sha256"
    outputs = (code_manifest, source_archive, environment_path, release_path, anchor_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError("v6.0.1 code release outputs already exist")
    missing = [path.as_posix() for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"v6 source inputs are missing: {missing}")
    for required in (
        FORMAL_CONTRACT,
        FROZEN_ASSET_MANIFEST,
        ORIGINAL_SOURCE_SNAPSHOT,
    ):
        if not (ROOT / required).is_file():
            raise FileNotFoundError(required)
    formal = json.loads((ROOT / FORMAL_CONTRACT).read_text(encoding="utf-8"))
    declared = formal.get("asset_hash_manifest")
    frozen_sha = sha256_file(ROOT / FROZEN_ASSET_MANIFEST)
    if not isinstance(declared, dict) or declared.get("sha256") != frozen_sha:
        raise RuntimeError("v6 formal contract asset manifest mismatch")

    lines = [
        f"{sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(SOURCE_PATHS, key=lambda item: item.as_posix())
    ]
    _atomic_text("\n".join(lines) + "\n", code_manifest)
    deterministic_source_archive(SOURCE_PATHS, source_archive)
    _atomic_json(environment_summary(), environment_path)
    evaluator = Path("jcas/workflows/evaluator.py")
    finalizer = Path("jcas/release/finalize_v6.py")
    release = {
        "scope": "offline_authorized_av2_model_robustness",
        "status": RELEASE_STATUS,
        "version": PACKAGE_VERSION,
        "method_changed": False,
        "model_or_checkpoint_changed": False,
        "result_asset_changed": False,
        "formal_test_rerun": False,
        "original_v6_source_snapshot": {
            "path": ORIGINAL_SOURCE_SNAPSHOT.as_posix(),
            "sha256": sha256_file(ROOT / ORIGINAL_SOURCE_SNAPSHOT),
        },
        "formal_contract": {
            "path": FORMAL_CONTRACT.as_posix(),
            "sha256": sha256_file(ROOT / FORMAL_CONTRACT),
        },
        "frozen_asset_manifest": {
            "path": FROZEN_ASSET_MANIFEST.as_posix(),
            "sha256": frozen_sha,
        },
        "code_asset_manifest": {
            "path": code_manifest.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(code_manifest),
            "entries": len(lines),
        },
        "source_archive": {
            "path": source_archive.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(source_archive),
            "deterministic": True,
        },
        "environment": {
            "path": environment_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(environment_path),
        },
        "evaluator": {
            "path": evaluator.as_posix(),
            "sha256": sha256_file(ROOT / evaluator),
        },
        "finalizer": {
            "path": finalizer.as_posix(),
            "sha256": sha256_file(ROOT / finalizer),
        },
    }
    _atomic_json(release, release_path)
    release_sha = sha256_file(release_path)
    _atomic_text(
        f"{release_sha}  {release_path.relative_to(ROOT).as_posix()}\n",
        anchor_path,
    )
    print(f"v6.0.1 code release: {release_path.relative_to(ROOT)}")
    print(f"external trust anchor SHA-256: {release_sha}")
    print(f"code assets: {len(lines)}")


if __name__ == "__main__":
    main()
