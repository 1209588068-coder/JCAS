#!/usr/bin/env python3
"""
Train the clean or transformed-data GENConv model for AV2 edge risk.

Current scope:
  - reads graph files produced by build_graph.py
  - derives the selected label from risk_labels.py at load time
  - optionally applies a pre-generated, model-independent train manifest
  - optional stricter supervision with label_strict_mask
  - uses the frozen GENConv architecture from models.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from jcas.core.models import build_model
from jcas.core.poison import apply_manifest_row, load_poison_manifest, sha256_file
from jcas.core.graph_splits import (
    SPLIT_METADATA_CONTRACT_VERSION,
    SPLIT_STRATEGY,
)
from jcas.core.graph_splits_v5 import (
    V5_SPLIT_METADATA_CONTRACT_VERSION,
    V5_SPLIT_STRATEGY,
)
from jcas.core.risk_labels import (
    RiskLabelConfig,
    label_config_dict,
    label_config_hash,
    labels_for_graph,
    selected_label_computable_mask,
)
from jcas.core.shadow_folds import effective_shadow_manifest, load_shadow_fold_manifest
from jcas.core.strict_shadow_folds import (
    STRICT_CROSSFIT_PROTOCOL,
    effective_strict_fit_manifest,
    load_strict_crossfit_contract,
    load_strict_crossfit_release,
)


@dataclass(frozen=True)
class GNNTrainConfig:
    graph_dir: str
    graph_manifest: str | None = None
    output_dir: str = "record/clean_gnn_runs"
    model_name: str = "genconv"
    seed: int = 20260621
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.10
    norm: str = "layer"
    decoder_hidden_dim: int = 128
    decoder_num_layers: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    checkpoint_metric: str = "val_loss"
    require_strict_label: bool = False
    label_mode: str = "dynamic_risk"
    risk_base_distance_m: float = 5.0
    risk_reaction_time_s: float = 1.0
    risk_safe_decel_mps2: float = 4.0
    poison_manifest: str | None = None
    max_train_graphs: int | None = None
    max_val_graphs: int | None = None
    max_test_graphs: int | None = None
    evaluate_test: bool = False
    shadow_fold_manifest: str | None = None
    shadow_heldout_fold: int | None = None
    strict_shadow_contract: str | None = None
    strict_crossfit_release: str | None = None
    strict_crossfit_release_sha256: str | None = None
    require_strict_crossfit_manifest: bool = False
    device: str = "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GENConv dynamic-risk model on AV2 graphs."
    )
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument(
        "--graph-manifest",
        default=None,
        help="Explicit split manifest; defaults to <graph-dir>/manifest.csv",
    )
    parser.add_argument("--output-dir", default="record/clean_gnn_runs")
    parser.add_argument(
        "--model-name", default="genconv", choices=["genconv"]
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--norm", default="layer", choices=["layer", "batch", "identity"])
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint-metric", choices=["val_loss", "val_auc"], default="val_loss")
    parser.add_argument("--require-strict-label", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=["dynamic_risk"],
        default="dynamic_risk",
    )
    parser.add_argument("--risk-base-distance-m", type=float, default=5.0)
    parser.add_argument("--risk-reaction-time-s", type=float, default=1.0)
    parser.add_argument("--risk-safe-decel-mps2", type=float, default=4.0)
    parser.add_argument(
        "--poison-manifest",
        default=None,
        help="Frozen train-only CSV from generate_poison_manifest.py",
    )
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-val-graphs", type=int, default=None)
    parser.add_argument("--max-test-graphs", type=int, default=None)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicitly evaluate test after selection; omit during development",
    )
    parser.add_argument(
        "--shadow-fold-manifest",
        default=None,
        help=(
            "Optional train-only fold CSV from generate_shadow_folds.py; when "
            "set, original validation/test rows are excluded"
        ),
    )
    parser.add_argument(
        "--shadow-heldout-fold",
        type=int,
        default=None,
        help="Fold used as clean shadow validation; the other folds train",
    )
    parser.add_argument(
        "--strict-shadow-contract",
        default=None,
        help=(
            "v4.2 per-fold strict cross-fit contract. Its held-out score fold "
            "is excluded from both gradient updates and checkpoint selection."
        ),
    )
    parser.add_argument(
        "--require-strict-crossfit-manifest",
        action="store_true",
        help=(
            "Require the poison manifest to preserve v4.2 per-scenario strict "
            "surrogate checkpoint/fit/score bindings"
        ),
    )
    parser.add_argument(
        "--strict-crossfit-release",
        default=None,
        help="Externally anchored v4.2 pretraining release JSON",
    )
    parser.add_argument(
        "--strict-crossfit-release-sha256",
        default=None,
        help="External SHA-256 trust anchor for --strict-crossfit-release",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int8)
    result: dict[str, Any] = {
        "num_edges": int(y_true.size),
        "positive_edges": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if y_true.size else 0.0,
    }
    if y_true.size and len(np.unique(y_true)) >= 2:
        result["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        result["auc"] = None
    return result


def resolve_graph_path(path_str: str, graph_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    alt = graph_dir / path
    if alt.exists():
        return alt
    if path.exists():
        return path
    raise FileNotFoundError(f"Cannot resolve graph path: {path_str}")


def resolve_manifest_path(
    graph_dir: Path, manifest_path: str | Path | None = None
) -> Path:
    if manifest_path is None:
        path = graph_dir / "manifest.csv"
    else:
        requested = Path(manifest_path)
        candidates = (
            requested,
            graph_dir / requested,
        )
        path = next((candidate for candidate in candidates if candidate.exists()), requested)
    if not path.exists():
        raise FileNotFoundError(f"graph manifest does not exist: {path}")
    return path


def load_manifest(
    graph_dir: Path,
    manifest_path: str | Path | None = None,
    *,
    require_graph_sha256: bool = False,
) -> pd.DataFrame:
    path = resolve_manifest_path(graph_dir, manifest_path)
    manifest = pd.read_csv(path, dtype={"scenario_id": str})
    required = {"scenario_id", "status", "graph_path", "split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"graph manifest is missing columns {missing}: {path}")
    manifest = manifest[manifest["status"].isin(["built", "exists"])].copy()
    manifest = manifest[manifest["graph_path"].notna()].copy()
    manifest = manifest.sort_values("scenario_id")
    if manifest.empty:
        raise RuntimeError(f"No built graph rows found in {path}")
    if not manifest["split"].isin(["train", "val", "test"]).all():
        raise ValueError(f"graph manifest has invalid or missing split values: {path}")
    if manifest["scenario_id"].duplicated().any():
        raise ValueError(f"graph manifest contains duplicate scenario_id values: {path}")
    if require_graph_sha256:
        if "graph_sha256" not in manifest:
            raise ValueError(
                "formal training requires graph_sha256 in the graph manifest; "
                "regenerate the split manifest with resplit_graph_manifest.py"
            )
        graph_hash = manifest["graph_sha256"].astype(str).str.lower()
        valid_hash = graph_hash.str.fullmatch(r"[0-9a-f]{64}", na=False)
        if not bool(valid_hash.all()):
            raise ValueError(
                "graph manifest contains missing or invalid graph_sha256 values"
            )
        manifest["graph_sha256"] = graph_hash
    return manifest


def validate_graph_manifest_contract(
    manifest_path: Path,
) -> dict[str, Any]:
    """Require the sidecar emitted by the leakage-resistant resplit tool."""
    sidecar_path = manifest_path.with_suffix(
        manifest_path.suffix + ".metadata.json"
    )
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            "formal training requires the split-manifest metadata sidecar: "
            f"{sidecar_path}"
        )
    with sidecar_path.open(encoding="utf-8") as stream:
        sidecar = json.load(stream)
    actual_manifest = sha256_file(manifest_path)
    if str(sidecar.get("output_manifest_sha256", "")).lower() != actual_manifest:
        raise RuntimeError(
            "graph manifest SHA-256 does not match its split contract"
        )
    split_audit = sidecar.get("split_audit")
    if not isinstance(split_audit, dict):
        raise ValueError("split contract has no split_audit")
    strategy = str(split_audit.get("strategy", ""))
    legacy_strategy = "av_overlap_recording_content_group_sha256_v3"
    if strategy not in {legacy_strategy, SPLIT_STRATEGY, V5_SPLIT_STRATEGY}:
        raise ValueError("graph manifest does not use a supported formal split")
    leakage = split_audit.get("leakage_violations")
    if not isinstance(leakage, dict) or any(int(value) != 0 for value in leakage.values()):
        raise ValueError("split contract reports leakage violations")
    verification = sidecar.get("graph_verification")
    if (
        not isinstance(verification, dict)
        or verification.get("all_built_rows_have_sha256") is not True
        or int(verification.get("verification_errors", -1)) != 0
    ):
        raise ValueError("split contract has no successful graph verification")
    metadata_contract = sidecar.get("group_metadata_contract")
    if (
        not isinstance(metadata_contract, dict)
        or metadata_contract.get("formal_split_eligible") is not True
    ):
        raise ValueError("split contract used debug or unverified group metadata")
    if strategy in {SPLIT_STRATEGY, V5_SPLIT_STRATEGY}:
        built_graphs = int(verification.get("built_graphs_verified", -1))
        expected_metadata_version = (
            V5_SPLIT_METADATA_CONTRACT_VERSION
            if strategy == V5_SPLIT_STRATEGY
            else SPLIT_METADATA_CONTRACT_VERSION
        )
        source_contract_required = (
            sidecar.get("formal_v5_source_contract_required") is True
            if strategy == V5_SPLIT_STRATEGY
            else sidecar.get("formal_v4_source_contract_required") is True
        )
        if (
            not source_contract_required
            or metadata_contract.get("metadata_contract_version")
            != expected_metadata_version
            or metadata_contract.get("split_strategy") != strategy
            or int(verification.get("source_contract_graphs", -1))
            != built_graphs
            or int(verification.get("source_files_verified", -1))
            != 2 * built_graphs
            or verification.get("graph_schema_versions") != [4]
            or verification.get("build_contract_versions") != [3]
            or int(verification.get("build_config_variants", -1)) != 1
            or int(verification.get("builder_code_variants", -1)) != 1
        ):
            raise ValueError(
                "formal split contract lacks complete raw-source/build verification"
            )
        if strategy == V5_SPLIT_STRATEGY:
            parent = sidecar.get("input_manifest_contract")
            if (
                not isinstance(parent, dict)
                or parent.get("strategy") != SPLIT_STRATEGY
                or sidecar.get("graph_files_modified") is not False
            ):
                raise ValueError(
                    "v5 split contract is not bound to immutable grouped_v4 graph assets"
                )
    return {
        "path": str(sidecar_path),
        "sha256": sha256_file(sidecar_path),
        "strategy": strategy,
        "leakage_violations": dict(leakage),
    }


def verified_graph_path(
    row: Any,
    graph_dir: Path,
    *,
    require_graph_sha256: bool,
) -> Path:
    """Resolve a graph path and bind its bytes to the manifest row."""
    path = resolve_graph_path(str(row.graph_path), graph_dir)
    if require_graph_sha256:
        expected = str(getattr(row, "graph_sha256", "")).strip().lower()
        if len(expected) != 64:
            raise ValueError(
                f"scenario {row.scenario_id} has no valid graph_sha256"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"graph SHA-256 mismatch for scenario {row.scenario_id}: "
                f"expected {expected}, got {actual}"
            )
    return path


def supervision_mask(graph: np.lib.npyio.NpzFile, require_strict_label: bool) -> np.ndarray:
    mask = np.asarray(graph["supervision_edge_mask"], dtype=bool)
    if "label_computable_mask" in graph:
        mask = mask & np.asarray(graph["label_computable_mask"], dtype=bool)
    if require_strict_label:
        mask = mask & np.asarray(graph["label_strict_mask"], dtype=bool)
    return mask


def message_passing_mask(graph: np.lib.npyio.NpzFile) -> np.ndarray:
    """Physical topology for graph-v3; old graphs retain their original topology."""
    edge_count = int(np.asarray(graph["edge_index"]).shape[1])
    if "message_passing_edge_mask" not in graph:
        return np.ones(edge_count, dtype=bool)
    mask = np.asarray(graph["message_passing_edge_mask"], dtype=bool)
    if mask.shape != (edge_count,):
        raise ValueError("message_passing_edge_mask shape does not match edge_index")
    return mask


def load_split_graphs(
    manifest: pd.DataFrame,
    graph_dir: Path,
    split: str,
    require_strict_label: bool,
    limit: int | None,
    label_config: RiskLabelConfig | None = None,
    poison_rows: pd.DataFrame | None = None,
    require_graph_sha256: bool = True,
) -> tuple[list[Data], dict[str, int]]:
    label_config = label_config or RiskLabelConfig()
    rows = manifest[manifest["split"] == split].copy()
    rows = rows.sort_values("scenario_id")
    if limit is not None:
        rows = rows.head(limit)

    poison_by_scenario: dict[str, Any] | None = None
    if poison_rows is not None:
        poison_by_scenario = {}
        for poison_row in poison_rows.itertuples(index=False):
            scenario_id = str(poison_row.scenario_id)
            if scenario_id in poison_by_scenario:
                raise ValueError(
                    "poison manifest has multiple rows for one scenario"
                )
            poison_by_scenario[scenario_id] = poison_row

    data_list: list[Data] = []
    stats = {
        "graphs_considered": int(len(rows)),
        "graphs_used": 0,
        "graphs_without_supervision": 0,
        "supervised_edges": 0,
        "poisoned_graphs": 0,
        "poisoned_edges": 0,
        "graph_sha256_verified": 0,
    }

    for row in rows.itertuples(index=False):
        graph_path = verified_graph_path(
            row,
            graph_dir,
            require_graph_sha256=require_graph_sha256,
        )
        if require_graph_sha256:
            stats["graph_sha256_verified"] += 1
        with np.load(graph_path, allow_pickle=True) as graph:
            scene_poison = (
                poison_by_scenario.get(str(row.scenario_id))
                if poison_by_scenario is not None
                else None
            )
            label_bundle = labels_for_graph(graph, label_config)
            mask = supervision_mask(graph, require_strict_label)
            mask &= selected_label_computable_mask(
                graph, label_config, label_bundle
            )
            if mask.size == 0 or not mask.any():
                stats["graphs_without_supervision"] += 1
                continue
            target_mask = np.zeros(mask.shape, dtype=bool)
            if scene_poison is not None:
                x_array, edge_attr_array, label_array, target_mask, _audit = apply_manifest_row(
                    graph,
                    scene_poison,
                    label_config,
                    require_strict_label=require_strict_label,
                )
                stats["poisoned_graphs"] += 1
                stats["poisoned_edges"] += int(target_mask.sum())
            else:
                x_array = graph["x_node"]
                edge_attr_array = graph["edge_attr"]
                label_array = label_bundle["edge_label"]

            x = torch.from_numpy(np.asarray(x_array, dtype=np.float32))
            edge_index = torch.from_numpy(np.asarray(graph["edge_index"], dtype=np.int64))
            edge_attr = torch.from_numpy(np.asarray(edge_attr_array, dtype=np.float32))
            edge_label = torch.from_numpy(np.asarray(label_array, dtype=np.float32))
            supervision = torch.from_numpy(mask.astype(np.bool_))
            message_mask = torch.from_numpy(message_passing_mask(graph).astype(np.bool_))
            label_attributes = {
                name: torch.from_numpy(np.asarray(values))
                for name, values in label_bundle.items()
                if name != "edge_label"
            }

        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_label=edge_label,
                supervision_edge_mask=supervision,
                message_passing_edge_mask=message_mask,
                poisoned_edge_mask=torch.from_numpy(target_mask.astype(np.bool_)),
                scenario_id=str(row.scenario_id),
                **label_attributes,
            )
        )
        stats["graphs_used"] += 1
        stats["supervised_edges"] += int(mask.sum())

    return data_list, stats


def logits_and_labels(model: torch.nn.Module, batch: Data) -> tuple[Tensor, Tensor]:
    logits = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.message_passing_edge_mask,
    )
    mask = batch.supervision_edge_mask
    return logits[mask], batch.edge_label[mask]


def _supervised_pair_predictions(
    logits: Tensor, labels: Tensor, batch: Data
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate two supervised directed edges into one unordered-pair score."""
    supervised_ids = torch.nonzero(
        batch.supervision_edge_mask, as_tuple=False
    ).flatten()
    if supervised_ids.numel() == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    edge_index = batch.edge_index[:, supervised_ids].detach().cpu().numpy()
    supervised_logits = logits[supervised_ids]
    supervised_labels = labels[supervised_ids]
    grouped: dict[tuple[int, int], list[int]] = {}
    for local_id, (src, dst) in enumerate(edge_index.T.tolist()):
        key = (min(int(src), int(dst)), max(int(src), int(dst)))
        grouped.setdefault(key, []).append(local_id)
    probabilities: list[float] = []
    pair_labels: list[float] = []
    for local_ids in grouped.values():
        if len(local_ids) != 2:
            continue
        pair_y = supervised_labels[local_ids]
        if not torch.equal(pair_y[0], pair_y[1]):
            raise RuntimeError(
                "symmetric pair task contains contradictory directed labels"
            )
        pair_probability = torch.sigmoid(supervised_logits[local_ids]).mean()
        probabilities.append(float(pair_probability.detach().cpu().item()))
        pair_labels.append(float(pair_y[0].detach().cpu().item()))
    return (
        np.asarray(probabilities, dtype=np.float32),
        np.asarray(pair_labels, dtype=np.float32),
    )


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_edges = 0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    pair_probs_list: list[np.ndarray] = []
    pair_labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            full_logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.message_passing_edge_mask,
            )
            mask = batch.supervision_edge_mask
            logits = full_logits[mask]
            labels = batch.edge_label[mask]
            if logits.numel() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += float(loss.item()) * int(labels.numel())
            total_edges += int(labels.numel())
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y = labels.detach().cpu().numpy()
            probs_list.append(probs)
            labels_list.append(y)
            pair_probs, pair_labels = _supervised_pair_predictions(
                full_logits, batch.edge_label, batch
            )
            if pair_probs.size:
                pair_probs_list.append(pair_probs)
                pair_labels_list.append(pair_labels)

    if not probs_list:
        empty = np.zeros((0,), dtype=np.float32)
        return 0.0, empty, empty, empty, empty
    all_probs = np.concatenate(probs_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_pair_probs = (
        np.concatenate(pair_probs_list, axis=0)
        if pair_probs_list
        else np.zeros((0,), dtype=np.float32)
    )
    all_pair_labels = (
        np.concatenate(pair_labels_list, axis=0)
        if pair_labels_list
        else np.zeros((0,), dtype=np.float32)
    )
    avg_loss = total_loss / max(total_edges, 1)
    return avg_loss, all_probs, all_labels, all_pair_probs, all_pair_labels


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_edges = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, labels = logits_and_labels(model, batch)
        if logits.numel() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(labels.numel())
        total_edges += int(labels.numel())
    return total_loss / max(total_edges, 1)


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Use decoupled decay only on matrix/tensor weights.

    Biases, LayerNorm/BatchNorm affine parameters, and learned scalar
    aggregation parameters are one-dimensional and must not be driven toward
    zero by the long full-dataset training schedule.
    """
    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and not name.endswith(".bias"):
            decay_parameters.append(parameter)
            decay_names.append(name)
        else:
            no_decay_parameters.append(parameter)
            no_decay_names.append(name)
    if not decay_parameters or not no_decay_parameters:
        raise RuntimeError("optimizer parameter grouping unexpectedly produced an empty group")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": float(weight_decay),
                "group_name": "decay_matrix_weights",
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
                "group_name": "no_decay_bias_norm_scalar",
            },
        ],
        lr=float(lr),
    )
    summary = {
        "name": "AdamW",
        "lr": float(lr),
        "matrix_weight_decay": float(weight_decay),
        "bias_norm_scalar_weight_decay": 0.0,
        "decay_parameter_tensors": int(len(decay_parameters)),
        "no_decay_parameter_tensors": int(len(no_decay_parameters)),
        "decay_parameter_names": decay_names,
        "no_decay_parameter_names": no_decay_names,
    }
    return optimizer, summary


def node_path_diagnostics(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    *,
    max_graphs: int = 100,
) -> dict[str, Any]:
    """Measure whether node encoding and the node-input path remain active."""
    model.eval()
    encoder_spans: list[float] = []
    encoder_channel_std: list[float] = []
    zero_node_logit_mae: list[float] = []
    node_count = 0
    edge_count = 0
    with torch.no_grad():
        for graph in graphs[: max(0, int(max_graphs))]:
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            edge_attr = graph.edge_attr.to(device)
            message_mask = graph.message_passing_edge_mask.to(device)
            supervised = graph.supervision_edge_mask.to(device)
            encoded = model.encode_nodes(x)
            node_count += int(encoded.shape[0])
            if encoded.shape[0] >= 2:
                centered = encoded - encoded.mean(dim=0, keepdim=True)
                encoder_spans.append(float(centered.abs().max().item()))
                encoder_channel_std.append(
                    float(encoded.std(dim=0, unbiased=False).mean().item())
                )
            logits = model(x, edge_index, edge_attr, message_mask)
            zero_logits = model(
                torch.zeros_like(x), edge_index, edge_attr, message_mask
            )
            if bool(supervised.any()):
                delta = (logits[supervised] - zero_logits[supervised]).abs()
                zero_node_logit_mae.append(float(delta.mean().item()))
                edge_count += int(delta.numel())

    layernorm_gamma_abs_mean: list[float] = []
    for module in model.node_encoder.modules():
        if isinstance(module, torch.nn.LayerNorm) and module.elementwise_affine:
            layernorm_gamma_abs_mean.append(
                float(module.weight.detach().abs().mean().item())
            )
    max_span = max(encoder_spans, default=0.0)
    mean_logit_effect = (
        float(np.mean(zero_node_logit_mae)) if zero_node_logit_mae else 0.0
    )
    return {
        "graphs": int(min(len(graphs), max(0, int(max_graphs)))),
        "nodes": int(node_count),
        "supervised_edges": int(edge_count),
        "node_encoder_max_within_graph_span": float(max_span),
        "node_encoder_mean_channel_std": (
            float(np.mean(encoder_channel_std)) if encoder_channel_std else 0.0
        ),
        "node_input_zero_ablation_logit_mae": mean_logit_effect,
        "node_layernorm_gamma_abs_mean": (
            float(np.mean(layernorm_gamma_abs_mean))
            if layernorm_gamma_abs_mean
            else None
        ),
        "node_encoder_constant_warning": bool(max_span <= 1e-6),
        "node_input_path_unused_warning": bool(mean_logit_effect <= 1e-6),
    }


def main() -> None:
    args = parse_args()
    cfg = GNNTrainConfig(
        graph_dir=args.graph_dir,
        graph_manifest=args.graph_manifest,
        output_dir=args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm=args.norm,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        checkpoint_metric=args.checkpoint_metric,
        require_strict_label=args.require_strict_label,
        label_mode=args.label_mode,
        risk_base_distance_m=args.risk_base_distance_m,
        risk_reaction_time_s=args.risk_reaction_time_s,
        risk_safe_decel_mps2=args.risk_safe_decel_mps2,
        poison_manifest=args.poison_manifest,
        max_train_graphs=args.max_train_graphs,
        max_val_graphs=args.max_val_graphs,
        max_test_graphs=args.max_test_graphs,
        evaluate_test=args.evaluate_test,
        shadow_fold_manifest=args.shadow_fold_manifest,
        shadow_heldout_fold=args.shadow_heldout_fold,
        strict_shadow_contract=args.strict_shadow_contract,
        strict_crossfit_release=args.strict_crossfit_release,
        strict_crossfit_release_sha256=args.strict_crossfit_release_sha256,
        require_strict_crossfit_manifest=args.require_strict_crossfit_manifest,
        device=args.device,
    )

    legacy_shadow_enabled = cfg.shadow_fold_manifest is not None
    if legacy_shadow_enabled != (cfg.shadow_heldout_fold is not None):
        raise ValueError(
            "--shadow-fold-manifest and --shadow-heldout-fold must be used together"
        )
    strict_shadow_enabled = cfg.strict_shadow_contract is not None
    if legacy_shadow_enabled and strict_shadow_enabled:
        raise ValueError(
            "legacy shadow folds and --strict-shadow-contract are mutually exclusive"
        )
    shadow_enabled = legacy_shadow_enabled or strict_shadow_enabled
    release_args_present = (
        cfg.strict_crossfit_release is not None,
        cfg.strict_crossfit_release_sha256 is not None,
    )
    if strict_shadow_enabled and not all(release_args_present):
        raise ValueError(
            "strict shadow training requires the v4.2 pretraining release and "
            "its external SHA-256"
        )
    if not strict_shadow_enabled and any(release_args_present):
        raise ValueError(
            "v4.2 pretraining release arguments require --strict-shadow-contract"
        )
    if strict_shadow_enabled and cfg.checkpoint_metric != "val_loss":
        raise ValueError("v4.2 strict surrogates must select checkpoints by val_loss")
    if shadow_enabled and cfg.evaluate_test:
        raise ValueError("shadow training cannot evaluate original test")
    if cfg.require_strict_crossfit_manifest and cfg.poison_manifest is None:
        raise ValueError(
            "--require-strict-crossfit-manifest requires --poison-manifest"
        )

    set_seed(cfg.seed)
    graph_dir = Path(cfg.graph_dir)
    graph_manifest_path = resolve_manifest_path(graph_dir, cfg.graph_manifest)
    graph_manifest_sha256 = sha256_file(graph_manifest_path)
    graph_manifest_contract = validate_graph_manifest_contract(
        graph_manifest_path
    )
    manifest = load_manifest(
        graph_dir,
        graph_manifest_path,
        require_graph_sha256=True,
    )
    original_split_graph_counts = {
        split: int((manifest["split"] == split).sum())
        for split in ("train", "val", "test")
    }
    shadow_fold_audit: dict[str, Any] | None = None
    if legacy_shadow_enabled:
        fold_frame, shadow_fold_audit = load_shadow_fold_manifest(
            str(cfg.shadow_fold_manifest),
            expected_graph_manifest_sha256=graph_manifest_sha256,
            expected_num_folds=3,
        )
        manifest = effective_shadow_manifest(
            manifest,
            fold_frame,
            heldout_fold=int(cfg.shadow_heldout_fold),
            num_folds=int(shadow_fold_audit["num_folds"]),
        )
        shadow_fold_audit = {
            **shadow_fold_audit,
            "surrogate_protocol": "legacy_heldout_validation_v1",
            "heldout_fold": int(cfg.shadow_heldout_fold),
            "effective_train_graphs": int((manifest["split"] == "train").sum()),
            "effective_val_graphs": int((manifest["split"] == "val").sum()),
            "original_validation_graphs_used": 0,
            "original_test_graphs_used": 0,
        }
    elif strict_shadow_enabled:
        fit_manifest, score_manifest, strict_audit = (
            load_strict_crossfit_contract(
                str(cfg.strict_shadow_contract),
                expected_graph_manifest_sha256=graph_manifest_sha256,
            )
        )
        manifest = effective_strict_fit_manifest(manifest, fit_manifest)
        release_audit = load_strict_crossfit_release(
            str(cfg.strict_crossfit_release),
            expected_release_sha256=str(
                cfg.strict_crossfit_release_sha256
            ),
            expected_graph_manifest_sha256=graph_manifest_sha256,
            expected_contract_path=str(cfg.strict_shadow_contract),
        )
        outer = strict_audit.get("outer_fold_manifest", {})
        shadow_fold_audit = {
            "path": str(outer.get("path", "")),
            "sha256": str(outer.get("sha256", "")),
            "metadata_path": str(outer.get("metadata_path", "")),
            "metadata_sha256": str(outer.get("metadata_sha256", "")),
            "version": str(strict_audit.get("version")),
            "surrogate_protocol": STRICT_CROSSFIT_PROTOCOL,
            "heldout_fold": int(strict_audit["heldout_fold"]),
            "num_folds": int(strict_audit["num_folds"]),
            "strict_contract_path": str(strict_audit["contract_path"]),
            "strict_contract_sha256": str(strict_audit["contract_sha256"]),
            "strict_crossfit_release_path": str(release_audit["path"]),
            "strict_crossfit_release_sha256": str(release_audit["sha256"]),
            "strict_crossfit_release_id": str(release_audit["release_id"]),
            "surrogate_fit_manifest_path": str(
                strict_audit["fit_manifest_path"]
            ),
            "surrogate_fit_manifest_sha256": str(
                strict_audit["fit_manifest_sha256"]
            ),
            "surrogate_score_manifest_path": str(
                strict_audit["score_manifest_path"]
            ),
            "surrogate_score_manifest_sha256": str(
                strict_audit["score_manifest_sha256"]
            ),
            "effective_train_graphs": int((manifest["split"] == "train").sum()),
            "effective_val_graphs": int((manifest["split"] == "val").sum()),
            "heldout_score_graphs": int(len(score_manifest)),
            "intersections": strict_audit["intersections"],
            "inner_split_seed": int(strict_audit["inner_split_seed"]),
            "inner_validation_fraction": float(
                strict_audit["inner_validation_fraction"]
            ),
            "original_validation_graphs_used": 0,
            "original_test_graphs_used": 0,
            "heldout_graphs_used_for_gradient_updates": 0,
            "heldout_graphs_used_for_checkpoint_selection": 0,
            "heldout_graphs_used_for_threshold_selection": 0,
            "heldout_graphs_used_for_hyperparameter_selection": 0,
        }
    label_config = RiskLabelConfig(
        label_mode=cfg.label_mode,
        risk_base_distance_m=cfg.risk_base_distance_m,
        risk_reaction_time_s=cfg.risk_reaction_time_s,
        risk_safe_decel_mps2=cfg.risk_safe_decel_mps2,
    )
    poison_rows = None
    poison_manifest_hash = None
    poison_metadata_path: Path | None = None
    poison_metadata_hash: str | None = None
    poison_metadata: dict[str, Any] | None = None
    if cfg.poison_manifest is not None:
        poison_rows, poison_manifest_hash = load_poison_manifest(
            cfg.poison_manifest,
            label_config,
            expected_split="train",
            require_strict_label=cfg.require_strict_label,
            require_metadata_binding=True,
            expected_graph_manifest_sha256=graph_manifest_sha256,
            require_strict_crossfit_binding=bool(
                cfg.require_strict_crossfit_manifest
            ),
        )
        poison_metadata_path = Path(cfg.poison_manifest).with_suffix(
            Path(cfg.poison_manifest).suffix + ".metadata.json"
        )
        poison_metadata_hash = sha256_file(poison_metadata_path)
        with poison_metadata_path.open(encoding="utf-8") as stream:
            poison_metadata = json.load(stream)

    train_graphs, train_stats = load_split_graphs(
        manifest, graph_dir, "train", cfg.require_strict_label, cfg.max_train_graphs,
        label_config, poison_rows, True
    )
    poison_rows_applied = 0
    if poison_rows is not None:
        effective_train_rows = manifest[manifest["split"] == "train"].sort_values(
            "scenario_id"
        )
        if cfg.max_train_graphs is not None:
            effective_train_rows = effective_train_rows.head(cfg.max_train_graphs)
        poison_rows_applied = int(
            poison_rows["scenario_id"].astype(str).isin(
                set(effective_train_rows["scenario_id"].astype(str))
            ).sum()
        )
        if train_stats["poisoned_graphs"] != poison_rows_applied:
            raise RuntimeError(
                "not every in-scope poison-manifest row was applied; check graph "
                "paths, shadow folds, and --max-train-graphs"
            )
    val_graphs, val_stats = load_split_graphs(
        manifest, graph_dir, "val", cfg.require_strict_label, cfg.max_val_graphs,
        label_config, None, True
    )
    test_graphs: list[Data] = []
    test_stats: dict[str, int] | None = None
    if cfg.evaluate_test:
        test_graphs, test_stats = load_split_graphs(
            manifest,
            graph_dir,
            "test",
            cfg.require_strict_label,
            cfg.max_test_graphs,
            label_config,
            None,
            True,
        )

    if not train_graphs or not val_graphs or (cfg.evaluate_test and not test_graphs):
        raise RuntimeError(
            "A requested training/evaluation split has no supervised graphs."
        )

    node_dim = int(train_graphs[0].x.shape[1])
    edge_dim = int(train_graphs[0].edge_attr.shape[1])
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model_kwargs = dict(
        model_name=cfg.model_name,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        norm=cfg.norm,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_num_layers=cfg.decoder_num_layers,
    )
    model = build_model(**model_kwargs).to(device)

    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.batch_size, shuffle=False)
    test_loader = (
        DataLoader(test_graphs, batch_size=cfg.batch_size, shuffle=False)
        if cfg.evaluate_test
        else None
    )

    optimizer, optimizer_summary = build_optimizer(
        model, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_state = None
    best_selection_value = float("inf") if cfg.checkpoint_metric == "val_loss" else -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_prob, val_y, _val_pair_prob, _val_pair_y = evaluate(
            model, val_loader, device
        )
        val_auc = float(roc_auc_score(val_y, val_prob)) if len(np.unique(val_y)) >= 2 else 0.0
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
            }
        )
        print(
            f"[train_gnn] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_auc={val_auc:.6f}"
        )
        selection_value = val_loss if cfg.checkpoint_metric == "val_loss" else val_auc
        improved = (
            selection_value < best_selection_value
            if cfg.checkpoint_metric == "val_loss"
            else selection_value > best_selection_value
        )
        if improved:
            best_selection_value = selection_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished without a best model snapshot.")

    model.load_state_dict(best_state)
    train_loss, train_prob, train_y, train_pair_prob, train_pair_y = evaluate(
        model, train_loader, device
    )
    val_loss, val_prob, val_y, val_pair_prob, val_pair_y = evaluate(
        model, val_loader, device
    )
    if test_loader is not None:
        test_loss, test_prob, test_y, test_pair_prob, test_pair_y = evaluate(
            model, test_loader, device
        )
    else:
        test_loss = None
        test_prob = None
        test_y = None
        test_pair_prob = None
        test_pair_y = None

    threshold, val_best_f1 = best_threshold(val_y, val_prob)
    if val_pair_y.size == 0:
        raise RuntimeError(
            "validation split has no fully supervised symmetric vehicle pairs"
        )
    pair_threshold, val_pair_best_f1 = best_threshold(val_pair_y, val_pair_prob)
    node_diagnostics = node_path_diagnostics(
        model, val_graphs, device, max_graphs=100
    )
    if node_diagnostics["node_encoder_constant_warning"]:
        print(
            "[train_gnn] warning=node_encoder_constant_on_validation "
            f"span={node_diagnostics['node_encoder_max_within_graph_span']:.3e}"
        )
    if node_diagnostics["node_input_path_unused_warning"]:
        print(
            "[train_gnn] warning=node_input_path_unused_on_validation "
            f"logit_mae={node_diagnostics['node_input_zero_ablation_logit_mae']:.3e}"
        )
    result = {
        "config": asdict(cfg),
        "graph_manifest": {
            "path": str(graph_manifest_path),
            "sha256": graph_manifest_sha256,
            "grouped_split": bool("split_group_id" in manifest.columns),
            "graph_sha256_required": True,
            "contract": graph_manifest_contract,
        },
        "label_config": label_config_dict(label_config),
        "label_config_hash": label_config_hash(label_config),
        "poison_manifest": (
            {
                "path": str(cfg.poison_manifest),
                "sha256": poison_manifest_hash,
                "rows": int(len(poison_rows)),
                "applied_train_rows": int(poison_rows_applied),
                "selection_authority": str(
                    poison_metadata.get(
                        "selection_authority",
                        "stored_data_and_ground_truth_only",
                    )
                ),
                "allocation_policy": poison_metadata.get(
                    "allocation_policy", "single_destination_v1"
                ),
                "selection_objective": poison_metadata.get(
                    "selection_objective"
                ),
                "base_manifest_sha256": poison_metadata.get(
                    "base_manifest_sha256"
                ),
                "metadata_path": str(poison_metadata_path),
                "metadata_sha256": poison_metadata_hash,
            }
            if poison_rows is not None
            else None
        ),
        "training_protocol": {
            "objective": "ordinary_binary_cross_entropy_symmetric_pair_labels",
            "initialization": "random",
            "checkpoint_metric": cfg.checkpoint_metric,
            "checkpoint_data": (
                "train_only_inner_validation_excluding_heldout_score_fold"
                if strict_shadow_enabled
                else "train_only_shadow_heldout_fold"
                if legacy_shadow_enabled
                else "clean_validation_only"
            ),
            "training_asr_computed": False,
            "test_evaluated": bool(cfg.evaluate_test),
            "teacher_used": False,
            "replay_used": False,
            "optimizer": optimizer_summary,
        },
        "shadow_protocol": (
            {
                "enabled": True,
                **dict(shadow_fold_audit or {}),
            }
            if shadow_enabled
            else {"enabled": False}
        ),
        "model_config": model_kwargs,
        "device": str(device),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "node_path_diagnostics": node_diagnostics,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "train_metrics": metric_dict(train_y, train_prob, threshold),
        "val_metrics": metric_dict(val_y, val_prob, threshold),
        "test_metrics": (
            metric_dict(test_y, test_prob, threshold)
            if test_y is not None and test_prob is not None
            else None
        ),
        "train_pair_metrics": metric_dict(
            train_pair_y, train_pair_prob, pair_threshold
        ),
        "val_pair_metrics": metric_dict(
            val_pair_y, val_pair_prob, pair_threshold
        ),
        "test_pair_metrics": (
            metric_dict(test_pair_y, test_pair_prob, pair_threshold)
            if test_pair_y is not None and test_pair_prob is not None
            else None
        ),
        "losses": {
            "train": float(train_loss),
            "val": float(val_loss),
            "test": float(test_loss) if test_loss is not None else None,
        },
        "val_best_f1": float(val_best_f1),
        "val_pair_best_f1": float(val_pair_best_f1),
        "history": history,
        "split_graph_counts": {
            "train": int((manifest["split"] == "train").sum()),
            "val": int((manifest["split"] == "val").sum()),
            "test": int((manifest["split"] == "test").sum()),
        },
        "original_split_graph_counts": original_split_graph_counts,
    }

    run_name = (
        f"{cfg.model_name}_{'strict_' if cfg.require_strict_label else ''}"
        f"seed{cfg.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    if sha256_file(graph_manifest_path) != graph_manifest_sha256:
        raise RuntimeError("graph manifest changed while training was running")
    if strict_shadow_enabled:
        final_release_audit = load_strict_crossfit_release(
            str(cfg.strict_crossfit_release),
            expected_release_sha256=str(
                cfg.strict_crossfit_release_sha256
            ),
            expected_graph_manifest_sha256=graph_manifest_sha256,
            expected_contract_path=str(cfg.strict_shadow_contract),
        )
        if final_release_audit["sha256"] != release_audit["sha256"]:
            raise RuntimeError("v4.2 pretraining release changed during training")
    if cfg.poison_manifest is not None:
        if sha256_file(cfg.poison_manifest) != poison_manifest_hash:
            raise RuntimeError(
                "poison manifest changed while training was running"
            )
        if (
            poison_metadata_path is None
            or sha256_file(poison_metadata_path) != poison_metadata_hash
        ):
            raise RuntimeError(
                "poison manifest metadata changed while training was running"
            )

    checkpoint_path = run_dir / "best_model.pt"
    checkpoint_temporary = checkpoint_path.with_name(
        f".{checkpoint_path.name}.tmp.{os.getpid()}"
    )
    try:
        torch.save(best_state, checkpoint_temporary)
        os.replace(checkpoint_temporary, checkpoint_path)
    finally:
        checkpoint_temporary.unlink(missing_ok=True)
    result["checkpoint"] = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "state": "best_validation_selected_state_dict",
    }

    result_path = run_dir / "result.json"
    result_temporary = result_path.with_name(
        f".{result_path.name}.tmp.{os.getpid()}"
    )
    try:
        with result_temporary.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        os.replace(result_temporary, result_path)
    finally:
        result_temporary.unlink(missing_ok=True)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[train_gnn] saved_result={result_path}")
    print(f"[train_gnn] saved_model={checkpoint_path}")


if __name__ == "__main__":
    main()
