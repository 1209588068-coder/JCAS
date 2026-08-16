#!/usr/bin/env python3
"""
Build AV2-MF vehicle interaction graphs for edge-level risk classification.

Graph-v4 contract:
  - one scenario -> one graph file (.npz)
  - nodes: high-quality vehicle/bus tracks
  - directed edges: ordered vehicle pairs with current distance <= radius
  - dual labels:
      edge_label_proximity = future_min_distance <= risk threshold
      edge_label_risk      = proximity AND interaction_candidate_mask (diagnostic only)
  - layered edge-mask system (edge_index always stores the raw observed graph):
      raw_edge_mask             : every observed-radius edge stored in edge_index
      physical_possible_mask    : excludes only high-confidence map-impossible edges
      task_candidate_mask       : physical_possible AND soft kinematic candidate
      label_computable_mask     : edge has enough future data for the base label
      supervision_edge_mask     : task_candidate AND label_computable
  - model inputs use observed history only; future arrays are saved only for
    labels and diagnostics

The script intentionally writes framework-neutral .npz files. Training code can
later convert them to PyG/DGL/vanilla PyTorch objects without rebuilding graphs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from jcas.core.graph_splits import (
    apply_grouped_splits,
    read_group_metadata,
    validate_group_metadata_contract,
)


OBS_LEN = 50
FUTURE_START = 50
FUTURE_LEN = 60
DT = 0.1
EPS = 1e-6
GRAPH_SCHEMA_VERSION = 4
BUILD_CONTRACT_VERSION = 3
PROTECTED_LEGACY_OUTPUT_DIRS = (
    Path("graphs/av2mf_graph_v3"),
)
SOURCE_CONTRACT_FIELDS = (
    "source_scenario_relpath",
    "source_parquet_relpath",
    "source_parquet_sha256",
    "source_parquet_size_bytes",
    "source_map_relpath",
    "source_map_sha256",
    "source_map_size_bytes",
    "builder_code_relpath",
    "builder_code_sha256",
    "build_config_sha256",
    "source_contract_json",
    "source_contract_sha256",
)

VEHICLE_TYPES = {"vehicle", "bus"}

NODE_FEATURE_NAMES: list[str] = (
    [f"rel_pos_x_t{t:02d}" for t in range(OBS_LEN)]
    + [f"rel_pos_y_t{t:02d}" for t in range(OBS_LEN)]
    + [f"vel_x_t{t:02d}" for t in range(OBS_LEN)]
    + [f"vel_y_t{t:02d}" for t in range(OBS_LEN)]
    + [f"heading_sin_t{t:02d}" for t in range(OBS_LEN)]
    + [f"heading_cos_t{t:02d}" for t in range(OBS_LEN)]
    + [f"observed_valid_t{t:02d}" for t in range(OBS_LEN)]
    + [
        "speed_current",
        "speed_mean",
        "speed_std",
        "acc_std",
        "lateral_acc_std",
        "heading_rate_std",
        "observed_fraction",
        "is_focal",
    ]
)

EDGE_FEATURE_NAMES: list[str] = [
    "rel_x_current",
    "rel_y_current",
    "d_current",
    "rel_vx_current",
    "rel_vy_current",
    "closing_speed_current",
    "d_min_obs",
    "d_mean_obs",
    "d_std_obs",
    "d_trend",
    "d_trend_early",
    "d_trend_late",
    "closing_speed_mean",
    "closing_speed_max",
    "closing_speed_std",
    "closing_speed_late",
    "closing_speed_trend",
    "heading_diff_sin",
    "heading_diff_cos",
    "src_heading_to_dst_sin",
    "src_heading_to_dst_cos",
    "dst_heading_to_src_sin",
    "dst_heading_to_src_cos",
    "src_velocity_toward_dst",
    "dst_velocity_toward_src",
    "speed_delta_src_minus_dst",
    "src_speed_current",
    "dst_speed_current",
    "src_speed_mean",
    "dst_speed_mean",
    "src_acc_std",
    "dst_acc_std",
    "src_lateral_acc_std",
    "dst_lateral_acc_std",
]


MAP_UNKNOWN = 0
MAP_SAME_LANE = 1
MAP_LATERAL_NEIGHBOR = 2
MAP_TOPOLOGY_CONNECTED = 3
MAP_INTERSECTION_CONFLICT = 4
MAP_GEOMETRY_ONLY = 5
MAP_BLOCKED_OR_UNRELATED = 6

MAP_RELATION_NAMES = {
    MAP_UNKNOWN: "unknown",
    MAP_SAME_LANE: "same_lane",
    MAP_LATERAL_NEIGHBOR: "lateral_neighbor",
    MAP_TOPOLOGY_CONNECTED: "topology_connected",
    MAP_INTERSECTION_CONFLICT: "intersection_conflict",
    MAP_GEOMETRY_ONLY: "geometry_only",
    MAP_BLOCKED_OR_UNRELATED: "blocked_or_unrelated",
}


@dataclass(frozen=True)
class BuildConfig:
    data_root: str
    output_dir: str
    radius_m: float = 60.0
    future_risk_distance_m: float = 10.0
    min_observed: int = 40
    min_last_observed_timestep: int = 48
    min_pair_future: int = 30
    max_speed_mps: float = 40.0
    max_acc_mps2: float = 10.0
    max_pos_jumps: int = 2
    max_vel_jumps: int = 2
    max_heading_jumps: int = 3
    jump_sigma: float = 5.0
    seed: int = 20260621
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_group_metadata: str = (
        "record/v3/"
        "duplicate_scene_check_overlap_v4.parquet"
    )
    candidate_close_distance_m: float = 35.0
    candidate_recent_close_distance_m: float = 30.0
    candidate_far_distance_m: float = 40.0
    candidate_dynamic_distance_m: float = 50.0
    candidate_approach_trend_m_per_step: float = -0.02
    candidate_receding_trend_m_per_step: float = 0.03
    candidate_closing_speed_mps: float = 0.5
    candidate_closing_speed_max_mps: float = 1.5
    candidate_perpendicular_min_distance_m: float = 15.0
    strict_pair_future: int = 50
    strict_future_last_timestep: int = 100
    lane_max_distance_m: float = 8.0
    perpendicular_lane_min_deg: float = 45.0
    perpendicular_lane_max_deg: float = 135.0
    overwrite: bool = False


def semantic_build_config(cfg: BuildConfig) -> dict[str, Any]:
    """Configuration fields that change graph tensors or labels.

    Paths, split ratios, the split RNG seed, and overwrite behavior are excluded
    because they do not change an individual graph's contents.
    """
    ignored = {
        "data_root",
        "output_dir",
        "seed",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "split_group_metadata",
        "overwrite",
    }
    return {k: v for k, v in asdict(cfg).items() if k not in ignored}


def semantic_build_config_json(cfg: BuildConfig) -> str:
    return json.dumps(semantic_build_config(cfg), sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


BUILD_GRAPH_CODE_SHA256 = sha256_file(Path(__file__).resolve())


def source_file_contract(
    scenario_dir: Path, cfg: BuildConfig
) -> dict[str, str]:
    parquet_files = sorted(scenario_dir.glob("scenario_*.parquet"))
    if len(parquet_files) != 1:
        raise FileNotFoundError(
            f"expected exactly one scenario parquet in {scenario_dir}; "
            f"found={len(parquet_files)}"
        )
    map_files = sorted(scenario_dir.glob("log_map_archive_*.json"))
    if len(map_files) != 1:
        raise FileNotFoundError(
            f"expected exactly one map archive in {scenario_dir}; "
            f"found={len(map_files)}"
        )
    data_root = Path(cfg.data_root).resolve()
    parquet_path = parquet_files[0].resolve()
    map_path = map_files[0].resolve()
    try:
        scenario_relpath = scenario_dir.resolve().relative_to(data_root)
        parquet_relpath = parquet_path.relative_to(data_root)
        map_relpath = map_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            f"scenario sources must be contained by data_root={data_root}"
        ) from exc
    build_config_json = semantic_build_config_json(cfg)
    contract_payload = {
        "contract_version": BUILD_CONTRACT_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "scenario_id": scenario_dir.name,
        "scenario_relpath": scenario_relpath.as_posix(),
        "parquet": {
            "relative_path": parquet_relpath.as_posix(),
            "sha256": sha256_file(parquet_path),
            "size_bytes": int(parquet_path.stat().st_size),
        },
        "map": {
            "relative_path": map_relpath.as_posix(),
            "sha256": sha256_file(map_path),
            "size_bytes": int(map_path.stat().st_size),
        },
        "build": {
            "code_path": "jcas/workflows/graph_builder.py",
            "code_sha256": BUILD_GRAPH_CODE_SHA256,
            "config_sha256": hashlib.sha256(
                build_config_json.encode("utf-8")
            ).hexdigest(),
        },
    }
    contract_json = json.dumps(
        contract_payload, sort_keys=True, separators=(",", ":")
    )
    return {
        "source_scenario_relpath": scenario_relpath.as_posix(),
        "source_parquet_relpath": parquet_relpath.as_posix(),
        "source_parquet_sha256": contract_payload["parquet"]["sha256"],
        "source_parquet_size_bytes": str(
            contract_payload["parquet"]["size_bytes"]
        ),
        "source_map_relpath": map_relpath.as_posix(),
        "source_map_sha256": contract_payload["map"]["sha256"],
        "source_map_size_bytes": str(contract_payload["map"]["size_bytes"]),
        "builder_code_relpath": "jcas/workflows/graph_builder.py",
        "builder_code_sha256": BUILD_GRAPH_CODE_SHA256,
        "build_config_sha256": contract_payload["build"]["config_sha256"],
        "source_contract_json": contract_json,
        "source_contract_sha256": hashlib.sha256(
            contract_json.encode("utf-8")
        ).hexdigest(),
    }


@dataclass
class TrackData:
    track_id: str
    object_type: str
    object_category: int
    is_focal: bool
    obs_pos_raw: np.ndarray
    obs_vel_raw: np.ndarray
    obs_heading_raw: np.ndarray
    obs_valid: np.ndarray
    fut_pos_raw: np.ndarray
    fut_valid: np.ndarray
    obs_pos: np.ndarray
    obs_vel: np.ndarray
    obs_heading: np.ndarray
    node_feature: np.ndarray
    speed_current: float
    speed_mean: float
    acc_std: float
    lateral_acc_std: float
    skip_reason: str | None = None


@dataclass
class LaneData:
    lane_id: int
    points: np.ndarray
    headings: np.ndarray
    is_intersection: bool
    lateral_neighbors: set[int]
    topology_neighbors: set[int]


def detect_jumps(values: np.ndarray, threshold_sigma: float = 5.0) -> int:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0
    diffs = np.abs(np.diff(values))
    # A mean/std cutoff masks multiple large jumps because the jumps inflate the
    # very scale used to detect them.  Median/MAD remains stable under several
    # outliers and makes max_*_jumps an effective constraint.
    center = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - center)))
    q25, q75 = np.quantile(diffs, [0.25, 0.75])
    robust_sigma = max(1.4826 * mad, float(q75 - q25) / 1.349, 1e-3)
    cutoff = float(center + threshold_sigma * robust_sigma)
    return int(np.sum(diffs > cutoff))


def fill_1d(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values.astype(np.float64))
    filled = s.interpolate(limit_direction="both").ffill().bfill().to_numpy()
    if not np.isfinite(filled).all():
        return np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
    return filled


def fill_heading(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values, dtype=np.float64)
    unwrapped = values.copy()
    unwrapped[valid] = np.unwrap(unwrapped[valid])
    filled = fill_1d(unwrapped)
    return np.arctan2(np.sin(filled), np.cos(filled))


def safe_slope(t: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(t) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    x = t[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom < EPS:
        return 0.0
    return float(np.sum(x * (yy - float(np.mean(yy)))) / denom)


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(np.asarray(a) - np.asarray(b)), np.cos(np.asarray(a) - np.asarray(b)))


def abs_angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return np.abs(angle_diff(a, b))


def finite_or_zero(x: float) -> float:
    return float(x) if math.isfinite(float(x)) else 0.0


def load_map_data(scenario_dir: Path) -> dict[str, Any] | None:
    json_files = sorted(scenario_dir.glob("log_map_archive_*.json"))
    if not json_files:
        return None
    try:
        with json_files[0].open() as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_savez_compressed(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        # Loading the temporary file before publication prevents a truncated
        # archive from replacing a valid graph after a worker interruption.
        with np.load(temporary, allow_pickle=True) as verification:
            if "scenario_id" not in verification:
                raise RuntimeError("temporary graph archive lacks scenario_id")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_existing_graph_row(graph_path: Path, cfg: BuildConfig | None = None) -> dict[str, Any] | None:
    try:
        graph = np.load(graph_path, allow_pickle=True)
    except Exception:
        return None

    def scalar_str(key: str, default: str = "") -> str:
        if key not in graph:
            return default
        value = graph[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            return str(value.item())
        return str(value)

    def sum_int(key: str) -> int:
        if key not in graph:
            return 0
        return int(np.asarray(graph[key]).sum())

    def size_int(key: str) -> int:
        if key not in graph:
            return 0
        return int(np.asarray(graph[key]).shape[0])

    def mean_float(key: str) -> float:
        if key not in graph:
            return 0.0
        arr = np.asarray(graph[key], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else 0.0

    if cfg is not None:
        required_contract_keys = {
            "graph_schema_version",
            "build_contract_version",
            "build_config_json",
            *SOURCE_CONTRACT_FIELDS,
        }
        if not required_contract_keys.issubset(set(graph.files)):
            graph.close()
            return None
        schema_version = int(np.asarray(graph["graph_schema_version"]).item())
        contract_version = int(
            np.asarray(graph["build_contract_version"]).item()
        )
        stored_config = str(np.asarray(graph["build_config_json"]).item())
        if (
            schema_version != GRAPH_SCHEMA_VERSION
            or contract_version != BUILD_CONTRACT_VERSION
            or stored_config != semantic_build_config_json(cfg)
        ):
            graph.close()
            return None
        scenario_id = scalar_str("scenario_id", graph_path.stem)
        try:
            expected_source_contract = source_file_contract(
                Path(cfg.data_root) / scenario_id,
                cfg,
            )
        except (FileNotFoundError, OSError):
            graph.close()
            return None
        if any(
            scalar_str(key) != expected
            for key, expected in expected_source_contract.items()
        ):
            graph.close()
            return None

    proximity_pos = sum_int("edge_label_proximity")
    risk_pos = sum_int("edge_label_risk")
    supervision_edges = sum_int("supervision_edge_mask")
    map_known_edges = sum_int("map_known_mask")
    map_unknown_edges = sum_int("map_unknown_mask")
    map_blocked_edges = sum_int("map_blocked_mask")
    edge_count = size_int("edge_label")
    raw_mask = (
        np.asarray(graph["raw_edge_mask"], dtype=bool)
        if "raw_edge_mask" in graph
        else np.ones(edge_count, dtype=bool)
    )
    physical_mask = (
        np.asarray(graph["physical_possible_mask"], dtype=bool)
        if "physical_possible_mask" in graph
        else raw_mask.copy()
    )
    task_mask = (
        np.asarray(graph["task_candidate_mask"], dtype=bool)
        if "task_candidate_mask" in graph
        else np.asarray(graph["interaction_candidate_mask"], dtype=bool)
    )
    label_computable = (
        np.asarray(graph["label_computable_mask"], dtype=bool)
        if "label_computable_mask" in graph
        else np.ones(edge_count, dtype=bool)
    )
    computable_edges = int(label_computable.sum())
    interaction_mask = task_mask
    raw_edges = int(raw_mask.sum())
    physical_possible_edges = int(physical_mask.sum())
    hard_pruned_mask = raw_mask & ~physical_mask
    soft_task_excluded_mask = physical_mask & ~task_mask
    hard_pruned_edges = int(hard_pruned_mask.sum())
    soft_task_excluded_edges = int(soft_task_excluded_mask.sum())
    interaction_candidate_edges = int(task_mask.sum())
    interaction_computable_edges = int((interaction_mask & label_computable).sum())
    candidate_proximity_pos = (
        int(np.asarray(graph["edge_label_proximity"], dtype=np.int64)[interaction_mask & label_computable].sum())
        if "edge_label_proximity" in graph and "interaction_candidate_mask" in graph
        else 0
    )
    filtered_proximity_pos = max(proximity_pos - candidate_proximity_pos, 0)
    proximity_label = np.asarray(graph["edge_label_proximity"], dtype=np.int64)
    hard_pruned_proximity_pos = int(proximity_label[hard_pruned_mask & label_computable].sum())
    soft_excluded_proximity_pos = int(
        proximity_label[soft_task_excluded_mask & label_computable].sum()
    )

    row = {
        "scenario_id": scalar_str("scenario_id", graph_path.stem),
        "city": scalar_str("city", "unknown"),
        "status": "exists",
        "graph_path": str(Path("graphs") / graph_path.name),
        "num_nodes": size_int("node_track_ids"),
        "num_edges": edge_count,
        "num_raw_edges": raw_edges,
        "num_physical_possible_edges": physical_possible_edges,
        "num_message_passing_edges": physical_possible_edges,
        "num_hard_pruned_edges": hard_pruned_edges,
        "num_task_candidate_edges": interaction_candidate_edges,
        "num_soft_task_excluded_edges": soft_task_excluded_edges,
        "physical_possible_retention_rate": float(physical_possible_edges / max(raw_edges, 1)),
        "task_candidate_retention_rate": float(interaction_candidate_edges / max(raw_edges, 1)),
        "num_label_computable_edges": computable_edges,
        "num_proximity_positive_edges": proximity_pos,
        "num_proximity_negative_edges": max(computable_edges - proximity_pos, 0),
        "proximity_positive_rate": float(proximity_pos / max(computable_edges, 1)),
        "num_risk_positive_edges": risk_pos,
        "num_risk_negative_edges": max(computable_edges - risk_pos, 0),
        "risk_positive_rate": float(risk_pos / max(computable_edges, 1)),
        "num_filtered_proximity_positive_edges": filtered_proximity_pos,
        "filtered_proximity_positive_rate": float(filtered_proximity_pos / max(proximity_pos, 1)),
        "num_hard_pruned_proximity_positive_edges": hard_pruned_proximity_pos,
        "num_soft_excluded_proximity_positive_edges": soft_excluded_proximity_pos,
        "physical_positive_recall": float(
            (proximity_pos - hard_pruned_proximity_pos) / max(proximity_pos, 1)
        ),
        "task_positive_recall": float(candidate_proximity_pos / max(proximity_pos, 1)),
        "num_interaction_candidate_edges": interaction_candidate_edges,
        "num_interaction_candidate_computable_edges": interaction_computable_edges,
        "num_supervision_edges": supervision_edges,
        "num_interaction_candidate_proximity_positive_edges": candidate_proximity_pos,
        "num_interaction_candidate_proximity_negative_edges": max(
            interaction_computable_edges - candidate_proximity_pos, 0
        ),
        "interaction_candidate_proximity_positive_rate": float(
            candidate_proximity_pos / max(interaction_computable_edges, 1)
        ),
        "interaction_candidate_retention_rate": float(interaction_candidate_edges / max(edge_count, 1)),
        "num_kinematic_candidate_edges": sum_int("kinematic_candidate_mask"),
        "num_map_candidate_edges": sum_int("map_candidate_mask"),
        "num_map_known_edges": map_known_edges,
        "num_map_unknown_edges": map_unknown_edges,
        "num_map_blocked_edges": map_blocked_edges,
        "num_same_lane_edges": sum_int("edge_same_lane"),
        "num_lateral_neighbor_lane_edges": sum_int("edge_lateral_neighbor_lane"),
        "num_connected_lane_edges": sum_int("edge_connected_lane"),
        "num_intersection_edges": sum_int("edge_either_intersection"),
        "mean_d_current": mean_float("d_current"),
        "mean_future_min_distance": mean_float("future_min_distance"),
        "num_lanes": int(len(np.unique(np.asarray(graph["node_nearest_lane_id"])[np.asarray(graph["node_nearest_lane_id"]) >= 0])))
        if "node_nearest_lane_id" in graph
        else 0,
        "graph_sha256": sha256_file(graph_path),
        "source_parquet_sha256": scalar_str("source_parquet_sha256"),
        "source_map_sha256": scalar_str("source_map_sha256"),
        "build_contract_version": (
            int(np.asarray(graph["build_contract_version"]).item())
            if "build_contract_version" in graph
            else 0
        ),
    }
    row.update(
        {key: scalar_str(key) for key in SOURCE_CONTRACT_FIELDS}
    )
    graph.close()
    return row


def parse_lanes(map_data: dict[str, Any] | None) -> dict[int, LaneData]:
    if not map_data or "lane_segments" not in map_data:
        return {}
    lanes: dict[int, LaneData] = {}
    for raw_id, lane in map_data["lane_segments"].items():
        try:
            lane_id = int(lane.get("id", raw_id))
            if lane.get("lane_type", "VEHICLE") != "VEHICLE":
                continue
            centerline = lane.get("centerline", [])
            pts = np.asarray([[float(p["x"]), float(p["y"])] for p in centerline], dtype=np.float32)
            if pts.shape[0] < 2:
                continue
            seg = np.diff(pts, axis=0)
            seg_heading = np.arctan2(seg[:, 1], seg[:, 0])
            headings = np.empty(pts.shape[0], dtype=np.float32)
            headings[:-1] = seg_heading
            headings[-1] = seg_heading[-1]
            lateral_neighbors: set[int] = set()
            topology_neighbors: set[int] = set()
            for key in ("left_neighbor_id", "right_neighbor_id"):
                val = lane.get(key)
                if val is not None:
                    try:
                        lateral_neighbors.add(int(val))
                    except Exception:
                        pass
            for key in ("predecessors", "successors"):
                for val in lane.get(key, []) or []:
                    try:
                        topology_neighbors.add(int(val))
                    except Exception:
                        pass
            lanes[lane_id] = LaneData(
                lane_id=lane_id,
                points=pts,
                headings=headings,
                is_intersection=bool(lane.get("is_intersection", False)),
                lateral_neighbors=lateral_neighbors,
                topology_neighbors=topology_neighbors,
            )
        except Exception:
            continue
    return lanes


def nearest_lane_features(current_pos: np.ndarray, lanes: dict[int, LaneData]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = current_pos.shape[0]
    lane_ids = np.full(n, -1, dtype=np.int64)
    lane_dist = np.full(n, np.inf, dtype=np.float32)
    lane_heading = np.zeros(n, dtype=np.float32)
    lane_is_intersection = np.zeros(n, dtype=bool)
    if not lanes:
        return lane_ids, lane_dist, lane_heading, lane_is_intersection

    lane_items = list(lanes.items())
    for i, pos in enumerate(current_pos):
        best_dist = np.inf
        best_lane: LaneData | None = None
        best_idx = 0
        for _, lane in lane_items:
            delta = lane.points - pos[None, :]
            d2 = np.sum(delta * delta, axis=1)
            idx = int(np.argmin(d2))
            dist = float(math.sqrt(float(d2[idx])))
            if dist < best_dist:
                best_dist = dist
                best_lane = lane
                best_idx = idx
        if best_lane is not None:
            lane_ids[i] = best_lane.lane_id
            lane_dist[i] = best_dist
            lane_heading[i] = float(best_lane.headings[best_idx])
            lane_is_intersection[i] = best_lane.is_intersection
    return lane_ids, lane_dist, lane_heading, lane_is_intersection


def build_track(track_id: str, grp: pd.DataFrame, focal_track_id: str, cfg: BuildConfig) -> TrackData:
    object_type = str(grp["object_type"].iloc[0])
    object_category = int(grp["object_category"].iloc[0])
    is_focal = str(track_id) == str(focal_track_id)

    obs = grp[grp["observed"] == True].sort_values("timestep")
    fut = grp[grp["observed"] == False].sort_values("timestep")

    obs_pos = np.full((OBS_LEN, 2), np.nan, dtype=np.float64)
    obs_vel = np.full((OBS_LEN, 2), np.nan, dtype=np.float64)
    obs_heading = np.full(OBS_LEN, np.nan, dtype=np.float64)
    obs_valid = np.zeros(OBS_LEN, dtype=bool)

    fut_pos = np.full((FUTURE_LEN, 2), np.nan, dtype=np.float64)
    fut_valid = np.zeros(FUTURE_LEN, dtype=bool)

    for row in obs.itertuples(index=False):
        t = int(row.timestep)
        if 0 <= t < OBS_LEN:
            px, py = float(row.position_x), float(row.position_y)
            vx, vy = float(row.velocity_x), float(row.velocity_y)
            hd = float(row.heading)
            valid = all(math.isfinite(v) for v in (px, py, vx, vy, hd))
            if valid:
                obs_pos[t] = (px, py)
                obs_vel[t] = (vx, vy)
                obs_heading[t] = hd
                obs_valid[t] = True

    for row in fut.itertuples(index=False):
        t = int(row.timestep) - FUTURE_START
        if 0 <= t < FUTURE_LEN:
            px, py = float(row.position_x), float(row.position_y)
            if math.isfinite(px) and math.isfinite(py):
                fut_pos[t] = (px, py)
                fut_valid[t] = True

    skip = None
    if object_type not in VEHICLE_TYPES:
        skip = "non_vehicle"
    elif int(obs_valid.sum()) < cfg.min_observed:
        skip = "obs_too_short"
    elif np.flatnonzero(obs_valid).size == 0 or int(np.flatnonzero(obs_valid).max()) < cfg.min_last_observed_timestep:
        skip = "no_last_obs"
    elif not np.isfinite(obs_pos[obs_valid]).all() or not np.isfinite(obs_vel[obs_valid]).all() or not np.isfinite(obs_heading[obs_valid]).all():
        skip = "has_missing"

    obs_pos_f = np.column_stack([fill_1d(obs_pos[:, 0]), fill_1d(obs_pos[:, 1])])
    obs_vel_f = np.column_stack([fill_1d(obs_vel[:, 0]), fill_1d(obs_vel[:, 1])])
    obs_heading_f = fill_heading(obs_heading)

    if skip is None:
        valid_t = np.flatnonzero(obs_valid)
        pos_jumps = (
            detect_jumps(obs_pos[valid_t, 0], cfg.jump_sigma)
            + detect_jumps(obs_pos[valid_t, 1], cfg.jump_sigma)
        )
        vel_jumps = (
            detect_jumps(obs_vel[valid_t, 0], cfg.jump_sigma)
            + detect_jumps(obs_vel[valid_t, 1], cfg.jump_sigma)
        )
        heading_jumps = detect_jumps(np.unwrap(obs_heading[valid_t]), cfg.jump_sigma)
        speed = np.linalg.norm(obs_vel[valid_t], axis=1)
        speed_extreme = bool(speed.size and float(np.nanmax(speed)) > cfg.max_speed_mps)

        acc_extreme = False
        if valid_t.size >= 2:
            dt = np.maximum(np.diff(valid_t).astype(np.float64) * DT, EPS)
            acc = np.diff(obs_vel[valid_t], axis=0) / dt[:, None]
            acc_norm = np.linalg.norm(acc, axis=1)
            acc_extreme = bool(acc_norm.size and float(np.nanmax(acc_norm)) > cfg.max_acc_mps2)

        if pos_jumps > cfg.max_pos_jumps:
            skip = "pos_jumps"
        elif vel_jumps > cfg.max_vel_jumps:
            skip = "vel_jumps"
        elif heading_jumps > cfg.max_heading_jumps:
            skip = "heading_jumps"
        elif speed_extreme:
            skip = "speed_extreme"
        elif acc_extreme:
            skip = "acc_extreme"

    node_feature, speed_current, speed_mean, acc_std, lateral_acc_std = make_node_feature(
        obs_pos_f, obs_vel_f, obs_heading_f, obs_valid, is_focal
    )

    return TrackData(
        track_id=str(track_id),
        object_type=object_type,
        object_category=object_category,
        is_focal=is_focal,
        obs_pos_raw=obs_pos.astype(np.float32),
        obs_vel_raw=obs_vel.astype(np.float32),
        obs_heading_raw=obs_heading.astype(np.float32),
        obs_valid=obs_valid,
        fut_pos_raw=fut_pos.astype(np.float32),
        fut_valid=fut_valid,
        obs_pos=obs_pos_f.astype(np.float32),
        obs_vel=obs_vel_f.astype(np.float32),
        obs_heading=obs_heading_f.astype(np.float32),
        node_feature=node_feature.astype(np.float32),
        speed_current=speed_current,
        speed_mean=speed_mean,
        acc_std=acc_std,
        lateral_acc_std=lateral_acc_std,
        skip_reason=skip,
    )


def make_node_feature(
    pos: np.ndarray,
    vel: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    is_focal: bool,
) -> tuple[np.ndarray, float, float, float, float]:
    current_pos = pos[-1]
    rel_pos = pos - current_pos[None, :]
    speed = np.linalg.norm(vel, axis=1)
    valid_idx = np.flatnonzero(valid)

    if valid_idx.size >= 2:
        dt = np.maximum(np.diff(valid_idx).astype(np.float64) * DT, EPS)
        vel_valid = vel[valid_idx]
        heading_valid = heading[valid_idx]
        acc = np.diff(vel_valid, axis=0) / dt[:, None]
        acc_norm = np.linalg.norm(acc, axis=1)
        heading_rate = np.diff(np.unwrap(heading_valid)) / dt
        lateral_dirs = np.column_stack([-np.sin(heading_valid[1:]), np.cos(heading_valid[1:])])
        lateral_acc = np.sum(acc * lateral_dirs, axis=1)
    else:
        acc_norm = np.zeros(0, dtype=np.float64)
        heading_rate = np.zeros(0, dtype=np.float64)
        lateral_acc = np.zeros(0, dtype=np.float64)

    speed_current = float(speed[-1]) if speed.size else 0.0
    speed_mean = float(np.mean(speed[valid_idx])) if valid_idx.size else 0.0
    speed_std = float(np.std(speed[valid_idx])) if valid_idx.size else 0.0
    acc_std = float(np.std(acc_norm)) if acc_norm.size else 0.0
    lateral_acc_std = float(np.std(lateral_acc)) if lateral_acc.size else 0.0
    heading_rate_std = float(np.std(heading_rate)) if heading_rate.size else 0.0

    compact = np.asarray(
        [
            speed_current,
            speed_mean,
            speed_std,
            acc_std,
            lateral_acc_std,
            heading_rate_std,
            np.mean(valid.astype(np.float32)),
            1.0 if is_focal else 0.0,
        ],
        dtype=np.float64,
    )
    feature = np.concatenate(
        [
            rel_pos[:, 0],
            rel_pos[:, 1],
            vel[:, 0],
            vel[:, 1],
            np.sin(heading),
            np.cos(heading),
            valid.astype(np.float64),
            compact,
        ],
        axis=0,
    )
    return (
        np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0),
        finite_or_zero(speed_current),
        finite_or_zero(speed_mean),
        finite_or_zero(acc_std),
        finite_or_zero(lateral_acc_std),
    )


def pair_closing_speed(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(rel_pos, axis=-1)
    unit = rel_pos / np.maximum(dist[..., None], EPS)
    return -np.sum(rel_vel * unit, axis=-1)


def compute_edge_features(
    src: int,
    dst: int,
    obs_pos: np.ndarray,
    obs_vel: np.ndarray,
    obs_heading: np.ndarray,
    obs_valid: np.ndarray,
    node_speed_current: np.ndarray,
    node_speed_mean: np.ndarray,
    node_acc_std: np.ndarray,
    node_lateral_acc_std: np.ndarray,
) -> np.ndarray:
    rel_current = obs_pos[dst, -1] - obs_pos[src, -1]
    rel_vel_current = obs_vel[dst, -1] - obs_vel[src, -1]
    d_current = float(np.linalg.norm(rel_current))
    closing_current = float(pair_closing_speed(rel_current[None, :], rel_vel_current[None, :])[0])

    pair_valid = obs_valid[src] & obs_valid[dst]
    t_all = np.arange(OBS_LEN, dtype=np.float64)
    if pair_valid.sum() >= 2:
        rel_seq = obs_pos[dst] - obs_pos[src]
        vel_seq = obs_vel[dst] - obs_vel[src]
        d_seq = np.linalg.norm(rel_seq, axis=1)
        cl_seq = pair_closing_speed(rel_seq, vel_seq)
        valid_d = d_seq[pair_valid]
        valid_cl = cl_seq[pair_valid]
        d_min = float(np.min(valid_d))
        d_mean = float(np.mean(valid_d))
        d_std = float(np.std(valid_d))
        d_trend = safe_slope(t_all[pair_valid], d_seq[pair_valid])
        early_mask = pair_valid & (t_all < OBS_LEN // 2)
        valid_ts = t_all[pair_valid]
        if valid_ts.size >= 20:
            late_ts = valid_ts[-20:]
            late_mask = np.isin(t_all, late_ts) & pair_valid
        else:
            late_mask = pair_valid
        d_trend_early = safe_slope(t_all[early_mask], d_seq[early_mask])
        d_trend_late = safe_slope(t_all[late_mask], d_seq[late_mask])
        cl_mean = float(np.mean(valid_cl))
        cl_max = float(np.max(valid_cl))
        cl_std = float(np.std(valid_cl))
        cl_late = float(np.mean(cl_seq[late_mask])) if late_mask.sum() else 0.0
        cl_trend = safe_slope(t_all[pair_valid], cl_seq[pair_valid])
    else:
        d_min = d_mean = d_current
        d_std = d_trend = d_trend_early = d_trend_late = 0.0
        cl_mean = cl_max = cl_std = cl_late = cl_trend = 0.0

    hdiff = float(angle_diff(obs_heading[dst, -1], obs_heading[src, -1]))
    unit_src_to_dst = rel_current / max(d_current, EPS)
    unit_dst_to_src = -unit_src_to_dst
    bearing_src_to_dst = float(math.atan2(rel_current[1], rel_current[0]))
    bearing_dst_to_src = float(math.atan2(-rel_current[1], -rel_current[0]))
    src_heading_to_dst = float(angle_diff(bearing_src_to_dst, float(obs_heading[src, -1])))
    dst_heading_to_src = float(angle_diff(bearing_dst_to_src, float(obs_heading[dst, -1])))
    src_velocity_toward_dst = float(np.dot(obs_vel[src, -1], unit_src_to_dst))
    dst_velocity_toward_src = float(np.dot(obs_vel[dst, -1], unit_dst_to_src))

    return np.asarray(
        [
            rel_current[0],
            rel_current[1],
            d_current,
            rel_vel_current[0],
            rel_vel_current[1],
            closing_current,
            d_min,
            d_mean,
            d_std,
            d_trend,
            d_trend_early,
            d_trend_late,
            cl_mean,
            cl_max,
            cl_std,
            cl_late,
            cl_trend,
            math.sin(hdiff),
            math.cos(hdiff),
            math.sin(src_heading_to_dst),
            math.cos(src_heading_to_dst),
            math.sin(dst_heading_to_src),
            math.cos(dst_heading_to_src),
            src_velocity_toward_dst,
            dst_velocity_toward_src,
            node_speed_current[src] - node_speed_current[dst],
            node_speed_current[src],
            node_speed_current[dst],
            node_speed_mean[src],
            node_speed_mean[dst],
            node_acc_std[src],
            node_acc_std[dst],
            node_lateral_acc_std[src],
            node_lateral_acc_std[dst],
        ],
        dtype=np.float32,
    )


def future_min_distance(
    src: int,
    dst: int,
    fut_pos: np.ndarray,
    fut_valid: np.ndarray,
) -> tuple[float, int, int, int, int, int, float]:
    pair_valid = fut_valid[src] & fut_valid[dst]
    count = int(pair_valid.sum())
    if count == 0:
        return float("inf"), -1, 0, -1, -1, 0, 0.0
    rel = fut_pos[dst, pair_valid] - fut_pos[src, pair_valid]
    dist = np.linalg.norm(rel, axis=1)
    arg = int(np.argmin(dist))
    timesteps = np.flatnonzero(pair_valid) + FUTURE_START
    first_valid = int(timesteps[0])
    last_valid = int(timesteps[-1])
    valid_span = int(last_valid - first_valid + 1)
    valid_fraction = float(count / FUTURE_LEN)
    return float(dist[arg]), int(timesteps[arg]), count, first_valid, last_valid, valid_span, valid_fraction


def lane_pair_features(
    src: int,
    dst: int,
    edge_d_current: float,
    node_lane_id: np.ndarray,
    node_lane_distance: np.ndarray,
    node_lane_heading: np.ndarray,
    node_lane_is_intersection: np.ndarray,
    lanes: dict[int, LaneData],
    cfg: BuildConfig,
) -> tuple[bool, bool, bool, float, bool, bool, bool, bool, int]:
    src_lane = int(node_lane_id[src])
    dst_lane = int(node_lane_id[dst])
    src_known = src_lane >= 0 and float(node_lane_distance[src]) <= cfg.lane_max_distance_m
    dst_known = dst_lane >= 0 and float(node_lane_distance[dst]) <= cfg.lane_max_distance_m
    same_lane = bool(src_known and dst_known and src_lane == dst_lane)
    lateral_neighbor_lane = False
    connected_lane = False
    if src_known and dst_known and src_lane in lanes and dst_lane in lanes:
        lateral_neighbor_lane = (
            dst_lane in lanes[src_lane].lateral_neighbors
            or src_lane in lanes[dst_lane].lateral_neighbors
        )
        connected_lane = (
            dst_lane in lanes[src_lane].topology_neighbors
            or src_lane in lanes[dst_lane].topology_neighbors
        )
    lane_heading_diff = float(abs_angle_diff(float(node_lane_heading[src]), float(node_lane_heading[dst])))
    either_intersection = bool(node_lane_is_intersection[src] or node_lane_is_intersection[dst])

    if not (src_known and dst_known):
        # Missing or distant lane assignment is ambiguous, not proof of non-interaction.
        return (
            same_lane,
            lateral_neighbor_lane,
            connected_lane,
            lane_heading_diff,
            either_intersection,
            True,
            False,
            True,
            MAP_UNKNOWN,
        )

    low = math.radians(cfg.perpendicular_lane_min_deg)
    high = math.radians(cfg.perpendicular_lane_max_deg)
    perpendicular_unrelated = (
        low <= lane_heading_diff <= high
        and not same_lane
        and not lateral_neighbor_lane
        and not connected_lane
        and not either_intersection
        and edge_d_current > cfg.candidate_perpendicular_min_distance_m
    )
    if same_lane:
        map_relation_type = MAP_SAME_LANE
        map_candidate_flag = True
    elif lateral_neighbor_lane:
        map_relation_type = MAP_LATERAL_NEIGHBOR
        map_candidate_flag = True
    elif connected_lane:
        map_relation_type = MAP_TOPOLOGY_CONNECTED
        map_candidate_flag = True
    elif either_intersection:
        map_relation_type = MAP_INTERSECTION_CONFLICT
        map_candidate_flag = True
    elif perpendicular_unrelated:
        map_relation_type = MAP_BLOCKED_OR_UNRELATED
        map_candidate_flag = False
    else:
        map_relation_type = MAP_GEOMETRY_ONLY
        map_candidate_flag = True
    return (
        same_lane,
        lateral_neighbor_lane,
        connected_lane,
        lane_heading_diff,
        either_intersection,
        map_candidate_flag,
        True,
        False,
        map_relation_type,
    )


def kinematic_candidate_details(
    edge_feature: np.ndarray, cfg: BuildConfig
) -> tuple[bool, bool, bool, bool]:
    """Motion-based interaction feasibility filter.

    Units:
      d_current, d_min_obs, candidate_close_distance_m         → meters
      d_trend, d_trend_late                                    → meters / timestep (≈ 0.1 s)
        e.g. -0.02 m/ts ≈ -0.2 m/s
      closing_current, closing_max, candidate_closing_speed_*  → m/s
    """
    names = EDGE_FEATURE_NAMES
    d_current = float(edge_feature[names.index("d_current")])
    d_min_obs = float(edge_feature[names.index("d_min_obs")])
    d_trend = float(edge_feature[names.index("d_trend")])
    d_trend_late = float(edge_feature[names.index("d_trend_late")])
    closing_current = float(edge_feature[names.index("closing_speed_current")])
    closing_max = float(edge_feature[names.index("closing_speed_max")])

    near_gate = (
        d_current <= cfg.candidate_close_distance_m
        or d_min_obs <= cfg.candidate_recent_close_distance_m
    )
    dynamic_gate = (
        d_current <= cfg.candidate_dynamic_distance_m
        and (
            d_trend <= cfg.candidate_approach_trend_m_per_step
            or d_trend_late <= cfg.candidate_approach_trend_m_per_step
            or closing_current >= cfg.candidate_closing_speed_mps
            or closing_max >= cfg.candidate_closing_speed_max_mps
        )
    )
    obviously_far_receding = (
        d_current > cfg.candidate_far_distance_m
        and d_trend_late > cfg.candidate_receding_trend_m_per_step
        and closing_current < -0.2
        and closing_max < cfg.candidate_closing_speed_mps
    )
    candidate = bool((near_gate or dynamic_gate) and not obviously_far_receding)
    return candidate, bool(near_gate), bool(dynamic_gate), bool(obviously_far_receding)


def kinematic_candidate(edge_feature: np.ndarray, cfg: BuildConfig) -> bool:
    """Backward-compatible boolean view of the soft task-candidate gate."""
    return kinematic_candidate_details(edge_feature, cfg)[0]


def build_scenario_graph(scenario_dir: Path, cfg: BuildConfig) -> dict[str, Any]:
    parquet_files = sorted(scenario_dir.glob("scenario_*.parquet"))
    if not parquet_files:
        return {"scenario_id": scenario_dir.name, "status": "skipped", "reason": "missing_parquet"}

    scenario_id = scenario_dir.name
    out_path = Path(cfg.output_dir) / "graphs" / f"{scenario_id}.npz"
    if out_path.exists() and not cfg.overwrite:
        existing_row = load_existing_graph_row(out_path, cfg)
        if existing_row is not None:
            return existing_row
        return {
            "scenario_id": scenario_id,
            "status": "error",
            "reason": "incompatible_existing_graph",
            "error": (
                f"{out_path} was built with an older schema or different semantic configuration; "
                "use a new output directory or pass --overwrite"
            ),
        }

    source_contract = source_file_contract(scenario_dir, cfg)
    df = pd.read_parquet(parquet_files[0])
    if df.empty:
        return {"scenario_id": scenario_id, "status": "skipped", "reason": "empty_parquet"}
    map_data = load_map_data(scenario_dir)
    lanes = parse_lanes(map_data)

    focal_track_id = str(df["focal_track_id"].dropna().iloc[0]) if "focal_track_id" in df else ""
    city = str(df["city"].dropna().iloc[0]) if "city" in df and df["city"].notna().any() else "unknown"

    tracks: list[TrackData] = []
    filter_counts: dict[str, int] = {}
    for track_id, grp in df.groupby("track_id", sort=False):
        tr = build_track(str(track_id), grp, focal_track_id, cfg)
        if tr.skip_reason is None:
            tracks.append(tr)
        else:
            filter_counts[tr.skip_reason] = filter_counts.get(tr.skip_reason, 0) + 1

    if len(tracks) < 2:
        return {
            "scenario_id": scenario_id,
            "city": city,
            "status": "skipped",
            "reason": "fewer_than_two_nodes",
            "num_nodes": len(tracks),
            **{f"filter_{k}": v for k, v in filter_counts.items()},
        }

    x_node = np.stack([t.node_feature for t in tracks]).astype(np.float32)
    obs_pos = np.stack([t.obs_pos for t in tracks]).astype(np.float32)
    obs_vel = np.stack([t.obs_vel for t in tracks]).astype(np.float32)
    obs_heading = np.stack([t.obs_heading for t in tracks]).astype(np.float32)
    obs_valid = np.stack([t.obs_valid for t in tracks])
    obs_pos_raw = np.stack([t.obs_pos_raw for t in tracks]).astype(np.float32)
    obs_vel_raw = np.stack([t.obs_vel_raw for t in tracks]).astype(np.float32)
    obs_heading_raw = np.stack([t.obs_heading_raw for t in tracks]).astype(np.float32)
    fut_pos = np.stack([t.fut_pos_raw for t in tracks]).astype(np.float32)
    fut_valid = np.stack([t.fut_valid for t in tracks])

    node_speed_current = np.asarray([t.speed_current for t in tracks], dtype=np.float32)
    node_speed_mean = np.asarray([t.speed_mean for t in tracks], dtype=np.float32)
    node_acc_std = np.asarray([t.acc_std for t in tracks], dtype=np.float32)
    node_lateral_acc_std = np.asarray([t.lateral_acc_std for t in tracks], dtype=np.float32)

    current_pos = obs_pos[:, -1, :]
    node_lane_id, node_lane_distance, node_lane_heading, node_lane_is_intersection = nearest_lane_features(
        current_pos, lanes
    )
    diff = current_pos[None, :, :] - current_pos[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    n = len(tracks)
    candidates = np.argwhere((dist <= cfg.radius_m) & (~np.eye(n, dtype=bool)))

    edge_index: list[tuple[int, int]] = []
    edge_attr: list[np.ndarray] = []
    edge_label_proximity: list[int] = []
    edge_label_computable_mask: list[bool] = []
    edge_future_min_distance: list[float] = []
    edge_future_min_timestep: list[int] = []
    edge_pair_future_count: list[int] = []
    edge_future_first_valid_timestep: list[int] = []
    edge_future_last_valid_timestep: list[int] = []
    edge_future_valid_span: list[int] = []
    edge_future_valid_fraction: list[float] = []
    edge_interaction_candidate_mask: list[bool] = []
    edge_supervision_mask: list[bool] = []
    edge_kinematic_candidate_mask: list[bool] = []
    edge_kinematic_near_gate_mask: list[bool] = []
    edge_kinematic_dynamic_gate_mask: list[bool] = []
    edge_kinematic_far_receding_mask: list[bool] = []
    edge_map_candidate_mask: list[bool] = []
    edge_map_known_mask: list[bool] = []
    edge_map_unknown_mask: list[bool] = []
    edge_map_blocked_mask: list[bool] = []
    edge_map_relation_type: list[int] = []
    edge_same_lane: list[bool] = []
    edge_lateral_neighbor_lane: list[bool] = []
    edge_connected_lane: list[bool] = []
    edge_lane_heading_diff: list[float] = []
    edge_either_intersection: list[bool] = []

    for src, dst in candidates:
        (
            fmin,
            fmin_t,
            pair_future_count,
            future_first_valid_timestep,
            future_last_valid_timestep,
            future_valid_span,
            future_valid_fraction,
        ) = future_min_distance(int(src), int(dst), fut_pos, fut_valid)
        label_computable_flag = pair_future_count >= cfg.min_pair_future
        features = compute_edge_features(
            int(src),
            int(dst),
            obs_pos,
            obs_vel,
            obs_heading,
            obs_valid,
            node_speed_current,
            node_speed_mean,
            node_acc_std,
            node_lateral_acc_std,
        )
        (
            kinematic_candidate_flag,
            kinematic_near_gate_flag,
            kinematic_dynamic_gate_flag,
            kinematic_far_receding_flag,
        ) = kinematic_candidate_details(features, cfg)
        d_current = float(features[EDGE_FEATURE_NAMES.index("d_current")])
        (
            same_lane,
            lateral_neighbor_lane,
            connected_lane,
            lane_hdiff,
            either_intersection,
            map_candidate_flag,
            map_known_flag,
            map_unknown_flag,
            map_relation_type,
        ) = lane_pair_features(
            int(src),
            int(dst),
            d_current,
            node_lane_id,
            node_lane_distance,
            node_lane_heading,
            node_lane_is_intersection,
            lanes,
            cfg,
        )
        # Map feasibility is the only hard topology gate. Kinematics remains a
        # soft task gate so an edge can still carry GNN messages and be audited.
        physical_possible_flag = bool(map_candidate_flag)
        task_candidate_flag = bool(physical_possible_flag and kinematic_candidate_flag)
        interaction_candidate_flag = task_candidate_flag  # compatibility alias
        # The message-passing topology must be available at inference time, so
        # it is built only from observed history. Future availability controls
        # label supervision, never whether an edge exists in edge_index.
        supervision_flag = bool(interaction_candidate_flag and label_computable_flag)
        edge_index.append((int(src), int(dst)))
        edge_attr.append(features)
        edge_label_proximity.append(
            1 if label_computable_flag and fmin <= cfg.future_risk_distance_m else 0
        )
        edge_label_computable_mask.append(label_computable_flag)
        edge_future_min_distance.append(fmin)
        edge_future_min_timestep.append(fmin_t)
        edge_pair_future_count.append(pair_future_count)
        edge_future_first_valid_timestep.append(future_first_valid_timestep)
        edge_future_last_valid_timestep.append(future_last_valid_timestep)
        edge_future_valid_span.append(future_valid_span)
        edge_future_valid_fraction.append(future_valid_fraction)
        edge_kinematic_candidate_mask.append(kinematic_candidate_flag)
        edge_kinematic_near_gate_mask.append(kinematic_near_gate_flag)
        edge_kinematic_dynamic_gate_mask.append(kinematic_dynamic_gate_flag)
        edge_kinematic_far_receding_mask.append(kinematic_far_receding_flag)
        edge_map_candidate_mask.append(map_candidate_flag)
        edge_interaction_candidate_mask.append(interaction_candidate_flag)
        edge_supervision_mask.append(supervision_flag)
        edge_map_known_mask.append(map_known_flag)
        edge_map_unknown_mask.append(map_unknown_flag)
        edge_map_blocked_mask.append(bool(map_relation_type == MAP_BLOCKED_OR_UNRELATED))
        edge_map_relation_type.append(map_relation_type)
        edge_same_lane.append(same_lane)
        edge_lateral_neighbor_lane.append(lateral_neighbor_lane)
        edge_connected_lane.append(connected_lane)
        edge_lane_heading_diff.append(lane_hdiff)
        edge_either_intersection.append(either_intersection)

    if not edge_index:
        return {
            "scenario_id": scenario_id,
            "city": city,
            "status": "skipped",
            "reason": "no_eligible_edges",
            "num_nodes": n,
            **{f"filter_{k}": v for k, v in filter_counts.items()},
        }

    edge_index_arr = np.asarray(edge_index, dtype=np.int64).T
    edge_attr_arr = np.stack(edge_attr).astype(np.float32)
    edge_label_proximity_arr = np.asarray(edge_label_proximity, dtype=np.int8)
    label_computable_mask_arr = np.asarray(edge_label_computable_mask, dtype=bool)
    future_min_arr = np.asarray(edge_future_min_distance, dtype=np.float32)
    future_t_arr = np.asarray(edge_future_min_timestep, dtype=np.int16)
    future_count_arr = np.asarray(edge_pair_future_count, dtype=np.int16)
    future_first_arr = np.asarray(edge_future_first_valid_timestep, dtype=np.int16)
    future_last_arr = np.asarray(edge_future_last_valid_timestep, dtype=np.int16)
    future_span_arr = np.asarray(edge_future_valid_span, dtype=np.int16)
    future_fraction_arr = np.asarray(edge_future_valid_fraction, dtype=np.float32)
    interaction_candidate_mask_arr = np.asarray(edge_interaction_candidate_mask, dtype=bool)
    supervision_mask_arr = np.asarray(edge_supervision_mask, dtype=bool)
    kinematic_candidate_arr = np.asarray(edge_kinematic_candidate_mask, dtype=bool)
    kinematic_near_gate_arr = np.asarray(edge_kinematic_near_gate_mask, dtype=bool)
    kinematic_dynamic_gate_arr = np.asarray(edge_kinematic_dynamic_gate_mask, dtype=bool)
    kinematic_far_receding_arr = np.asarray(edge_kinematic_far_receding_mask, dtype=bool)
    map_candidate_arr = np.asarray(edge_map_candidate_mask, dtype=bool)
    map_known_arr = np.asarray(edge_map_known_mask, dtype=bool)
    map_unknown_arr = np.asarray(edge_map_unknown_mask, dtype=bool)
    map_blocked_arr = np.asarray(edge_map_blocked_mask, dtype=bool)
    map_relation_type_arr = np.asarray(edge_map_relation_type, dtype=np.int8)
    raw_edge_mask_arr = np.ones(edge_label_proximity_arr.shape[0], dtype=bool)
    physical_possible_mask_arr = map_candidate_arr.copy()
    message_passing_edge_mask_arr = physical_possible_mask_arr.copy()
    task_candidate_mask_arr = physical_possible_mask_arr & kinematic_candidate_arr
    # Keep the pre-v3 key as an exact alias so downstream analysis can migrate
    # without silently changing edge indexing.
    interaction_candidate_mask_arr = task_candidate_mask_arr.copy()
    supervision_mask_arr = task_candidate_mask_arr & label_computable_mask_arr
    hard_pruned_edge_mask_arr = ~physical_possible_mask_arr
    soft_task_excluded_mask_arr = physical_possible_mask_arr & ~task_candidate_mask_arr
    edge_label_arr = edge_label_proximity_arr.astype(np.int8)
    edge_label_risk_arr = (edge_label_proximity_arr.astype(bool) & task_candidate_mask_arr).astype(np.int8)
    # In v3, "impossible" means hard physical exclusion only. Soft kinematic
    # exclusions are represented separately by soft_task_excluded_mask.
    impossible_interaction_mask_arr = hard_pruned_edge_mask_arr.copy()
    strict_label_mask_arr = (
        label_computable_mask_arr
        & (future_count_arr >= cfg.strict_pair_future)
        & (future_last_arr >= cfg.strict_future_last_timestep)
    )
    same_lane_arr = np.asarray(edge_same_lane, dtype=bool)
    lateral_neighbor_lane_arr = np.asarray(edge_lateral_neighbor_lane, dtype=bool)
    connected_lane_arr = np.asarray(edge_connected_lane, dtype=bool)
    lane_heading_diff_arr = np.asarray(edge_lane_heading_diff, dtype=np.float32)
    either_intersection_arr = np.asarray(edge_either_intersection, dtype=bool)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez_compressed(
        out_path,
        scenario_id=np.asarray(scenario_id),
        city=np.asarray(city),
        x_node=x_node,
        node_track_ids=np.asarray([t.track_id for t in tracks], dtype=object),
        node_object_type=np.asarray([t.object_type for t in tracks], dtype=object),
        node_object_category=np.asarray([t.object_category for t in tracks], dtype=np.int16),
        node_is_focal=np.asarray([t.is_focal for t in tracks], dtype=bool),
        observed_positions=obs_pos_raw,
        observed_velocities=obs_vel_raw,
        observed_headings=obs_heading_raw,
        observed_positions_filled=obs_pos,
        observed_velocities_filled=obs_vel,
        observed_headings_filled=obs_heading,
        future_positions=fut_pos,
        observed_valid_mask=obs_valid,
        future_valid_mask=fut_valid,
        node_nearest_lane_id=node_lane_id,
        node_nearest_lane_distance=node_lane_distance,
        node_nearest_lane_heading=node_lane_heading,
        node_nearest_lane_is_intersection=node_lane_is_intersection,
        num_lanes=np.asarray(len(lanes), dtype=np.int32),
        edge_index=edge_index_arr,
        edge_attr=edge_attr_arr,
        edge_label=edge_label_arr,
        edge_label_proximity=edge_label_proximity_arr,
        edge_label_risk=edge_label_risk_arr,
        graph_schema_version=np.asarray(GRAPH_SCHEMA_VERSION, dtype=np.int16),
        build_contract_version=np.asarray(BUILD_CONTRACT_VERSION, dtype=np.int16),
        build_config_json=np.asarray(semantic_build_config_json(cfg)),
        source_scenario_relpath=np.asarray(
            source_contract["source_scenario_relpath"]
        ),
        source_parquet_relpath=np.asarray(
            source_contract["source_parquet_relpath"]
        ),
        source_parquet_sha256=np.asarray(
            source_contract["source_parquet_sha256"]
        ),
        source_parquet_size_bytes=np.asarray(
            int(source_contract["source_parquet_size_bytes"]), dtype=np.int64
        ),
        source_map_relpath=np.asarray(source_contract["source_map_relpath"]),
        source_map_sha256=np.asarray(source_contract["source_map_sha256"]),
        source_map_size_bytes=np.asarray(
            int(source_contract["source_map_size_bytes"]), dtype=np.int64
        ),
        builder_code_relpath=np.asarray(
            source_contract["builder_code_relpath"]
        ),
        builder_code_sha256=np.asarray(
            source_contract["builder_code_sha256"]
        ),
        build_config_sha256=np.asarray(
            source_contract["build_config_sha256"]
        ),
        source_contract_json=np.asarray(
            source_contract["source_contract_json"]
        ),
        source_contract_sha256=np.asarray(
            source_contract["source_contract_sha256"]
        ),
        label_computable_mask=label_computable_mask_arr,
        label_strict_mask=strict_label_mask_arr,
        raw_edge_mask=raw_edge_mask_arr,
        physical_possible_mask=physical_possible_mask_arr,
        message_passing_edge_mask=message_passing_edge_mask_arr,
        hard_pruned_edge_mask=hard_pruned_edge_mask_arr,
        physically_impossible_mask=hard_pruned_edge_mask_arr,
        task_candidate_mask=task_candidate_mask_arr,
        soft_task_excluded_mask=soft_task_excluded_mask_arr,
        task_excluded_mask=~task_candidate_mask_arr,
        supervision_edge_mask=supervision_mask_arr,
        interaction_candidate_mask=interaction_candidate_mask_arr,
        impossible_interaction_mask=impossible_interaction_mask_arr,
        kinematic_candidate_mask=kinematic_candidate_arr,
        kinematic_near_gate_mask=kinematic_near_gate_arr,
        kinematic_dynamic_gate_mask=kinematic_dynamic_gate_arr,
        kinematic_far_receding_mask=kinematic_far_receding_arr,
        map_candidate_mask=map_candidate_arr,
        map_known_mask=map_known_arr,
        map_unknown_mask=map_unknown_arr,
        map_blocked_mask=map_blocked_arr,
        map_relation_type=map_relation_type_arr,
        map_relation_type_names=np.asarray([MAP_RELATION_NAMES[i] for i in range(len(MAP_RELATION_NAMES))], dtype=object),
        edge_same_lane=same_lane_arr,
        edge_lateral_neighbor_lane=lateral_neighbor_lane_arr,
        edge_connected_lane=connected_lane_arr,
        edge_lane_heading_diff=lane_heading_diff_arr,
        edge_either_intersection=either_intersection_arr,
        edge_src_track_id=np.asarray([tracks[s].track_id for s, _ in edge_index], dtype=object),
        edge_dst_track_id=np.asarray([tracks[d].track_id for _, d in edge_index], dtype=object),
        future_min_distance=future_min_arr,
        future_min_distance_timestep=future_t_arr,
        pair_future_valid_count=future_count_arr,
        future_first_valid_timestep=future_first_arr,
        future_last_valid_timestep=future_last_arr,
        future_valid_span=future_span_arr,
        future_valid_fraction=future_fraction_arr,
        d_current=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("d_current")],
        d_min_obs=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("d_min_obs")],
        d_mean_obs=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("d_mean_obs")],
        d_trend=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("d_trend")],
        d_trend_early=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("d_trend_early")],
        d_trend_late=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("d_trend_late")],
        closing_speed_current=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("closing_speed_current")],
        closing_speed_mean=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("closing_speed_mean")],
        closing_speed_max=edge_attr_arr[:, EDGE_FEATURE_NAMES.index("closing_speed_max")],
        node_feature_names=np.asarray(NODE_FEATURE_NAMES, dtype=object),
        edge_feature_names=np.asarray(EDGE_FEATURE_NAMES, dtype=object),
    )

    proximity_pos_edges = int(edge_label_proximity_arr.sum())
    risk_pos_edges = int(edge_label_risk_arr.sum())
    computable_edges = int(label_computable_mask_arr.sum())
    raw_edges = int(raw_edge_mask_arr.sum())
    physical_possible_edges = int(physical_possible_mask_arr.sum())
    hard_pruned_edges = int(hard_pruned_edge_mask_arr.sum())
    task_candidate_edges = int(task_candidate_mask_arr.sum())
    soft_task_excluded_edges = int(soft_task_excluded_mask_arr.sum())
    interaction_candidate_edges = task_candidate_edges
    interaction_candidate_computable_edges = int(
        (task_candidate_mask_arr & label_computable_mask_arr).sum()
    )
    supervision_edges = int(supervision_mask_arr.sum())
    candidate_proximity_pos_edges = (
        int(edge_label_proximity_arr[task_candidate_mask_arr & label_computable_mask_arr].sum())
        if task_candidate_edges
        else 0
    )
    filtered_proximity_pos_edges = int(proximity_pos_edges - candidate_proximity_pos_edges)
    hard_pruned_proximity_pos_edges = int(
        edge_label_proximity_arr[hard_pruned_edge_mask_arr & label_computable_mask_arr].sum()
    )
    soft_excluded_proximity_pos_edges = int(
        edge_label_proximity_arr[soft_task_excluded_mask_arr & label_computable_mask_arr].sum()
    )
    return {
        "scenario_id": scenario_id,
        "city": city,
        "status": "built",
        "graph_path": str(Path("graphs") / out_path.name),
        "graph_sha256": sha256_file(out_path),
        **source_contract,
        "build_contract_version": BUILD_CONTRACT_VERSION,
        "num_nodes": n,
        "num_edges": int(edge_label_risk_arr.size),
        "num_raw_edges": raw_edges,
        "num_physical_possible_edges": physical_possible_edges,
        "num_message_passing_edges": physical_possible_edges,
        "num_hard_pruned_edges": hard_pruned_edges,
        "num_task_candidate_edges": task_candidate_edges,
        "num_soft_task_excluded_edges": soft_task_excluded_edges,
        "physical_possible_retention_rate": float(physical_possible_edges / max(raw_edges, 1)),
        "task_candidate_retention_rate": float(task_candidate_edges / max(raw_edges, 1)),
        "num_label_computable_edges": computable_edges,
        "num_proximity_positive_edges": proximity_pos_edges,
        "num_proximity_negative_edges": int(computable_edges - proximity_pos_edges),
        "proximity_positive_rate": float(proximity_pos_edges / max(computable_edges, 1)),
        "num_risk_positive_edges": risk_pos_edges,
        "num_risk_negative_edges": int(computable_edges - risk_pos_edges),
        "risk_positive_rate": float(risk_pos_edges / max(computable_edges, 1)),
        "num_supervision_edges": supervision_edges,
        "num_filtered_proximity_positive_edges": filtered_proximity_pos_edges,
        "filtered_proximity_positive_rate": float(filtered_proximity_pos_edges / max(proximity_pos_edges, 1)),
        "num_hard_pruned_proximity_positive_edges": hard_pruned_proximity_pos_edges,
        "num_soft_excluded_proximity_positive_edges": soft_excluded_proximity_pos_edges,
        "physical_positive_recall": float(
            (proximity_pos_edges - hard_pruned_proximity_pos_edges) / max(proximity_pos_edges, 1)
        ),
        "task_positive_recall": float(candidate_proximity_pos_edges / max(proximity_pos_edges, 1)),
        "num_interaction_candidate_edges": interaction_candidate_edges,
        "num_interaction_candidate_computable_edges": interaction_candidate_computable_edges,
        "num_interaction_candidate_proximity_positive_edges": candidate_proximity_pos_edges,
        "num_interaction_candidate_proximity_negative_edges": int(
            interaction_candidate_computable_edges - candidate_proximity_pos_edges
        ),
        "interaction_candidate_proximity_positive_rate": float(
            candidate_proximity_pos_edges / max(interaction_candidate_computable_edges, 1)
        ),
        "interaction_candidate_retention_rate": float(interaction_candidate_edges / max(edge_label_risk_arr.size, 1)),
        "num_kinematic_candidate_edges": int(kinematic_candidate_arr.sum()),
        "num_map_candidate_edges": int(map_candidate_arr.sum()),
        "num_map_known_edges": int(map_known_arr.sum()),
        "num_map_unknown_edges": int(map_unknown_arr.sum()),
        "num_map_blocked_edges": int(map_blocked_arr.sum()),
        "num_same_lane_edges": int(same_lane_arr.sum()),
        "num_lateral_neighbor_lane_edges": int(lateral_neighbor_lane_arr.sum()),
        "num_connected_lane_edges": int(connected_lane_arr.sum()),
        "num_intersection_edges": int(either_intersection_arr.sum()),
        "mean_d_current": float(np.mean(edge_attr_arr[:, EDGE_FEATURE_NAMES.index("d_current")])),
        "mean_future_min_distance": (
            float(np.mean(future_min_arr[label_computable_mask_arr])) if computable_edges else 0.0
        ),
        "num_lanes": int(len(lanes)),
        **{f"filter_{k}": v for k, v in filter_counts.items()},
    }


def scenario_dirs(data_root: Path, limit: int | None = None) -> list[Path]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    return dirs[:limit] if limit is not None else dirs


def manifest_output_names(limit: int | None) -> tuple[str, str]:
    """Keep partial-build indexes separate from the complete manifest."""
    if limit is None:
        return "manifest.csv", "summary.json"
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    return f"manifest.limit_{int(limit)}.csv", f"summary.limit_{int(limit)}.json"


def effective_output_dir(output_dir: str | Path, limit: int | None) -> Path:
    """Isolate every smoke/partial build from the complete graph directory."""
    base = Path(output_dir)
    if limit is None:
        return base
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    return base / "partial_builds" / f"limit_{int(limit)}"


def validate_generation_output_dir(output_dir: str | Path) -> Path:
    """Prevent the v4 builder from writing into frozen legacy graph roots."""
    resolved = Path(output_dir).resolve()
    for legacy in PROTECTED_LEGACY_OUTPUT_DIRS:
        protected = legacy.resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                f"refusing to write the new graph generation under frozen "
                f"legacy directory: {protected}"
            )
    return resolved


def assign_splits(
    rows: list[dict[str, Any]], cfg: BuildConfig
) -> dict[str, Any]:
    """Assign order-independent candidate-recording/content-group splits."""
    validate_group_metadata_contract(cfg.split_group_metadata)
    group_metadata = read_group_metadata(cfg.split_group_metadata)
    assigned_rows, audit = apply_grouped_splits(
        rows,
        group_metadata,
        seed=cfg.seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )
    rows[:] = sorted(
        assigned_rows, key=lambda row: str(row.get("scenario_id", ""))
    )
    return audit


def _atomic_csv_write(
    rows: list[dict[str, Any]], fieldnames: list[str], path: Path
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(
    rows: list[dict[str, Any]],
    output_dir: Path,
    cfg: BuildConfig,
    *,
    manifest_name: str = "manifest.csv",
    summary_name: str = "summary.json",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_audit = assign_splits(rows, cfg)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    manifest_path = output_dir / manifest_name
    _atomic_csv_write(rows, fieldnames, manifest_path)

    built = [r for r in rows if r.get("status") in {"built", "exists"}]
    summary = {
        "config": asdict(cfg),
        "num_scenarios_total": len(rows),
        "num_graphs": len(built),
        "num_skipped": len(rows) - len(built),
        "num_nodes": int(sum(int(r.get("num_nodes", 0) or 0) for r in built)),
        "num_edges": int(sum(int(r.get("num_edges", 0) or 0) for r in built)),
        "num_raw_edges": int(sum(int(r.get("num_raw_edges", r.get("num_edges", 0)) or 0) for r in built)),
        "num_physical_possible_edges": int(
            sum(int(r.get("num_physical_possible_edges", r.get("num_edges", 0)) or 0) for r in built)
        ),
        "num_message_passing_edges": int(
            sum(int(r.get("num_message_passing_edges", r.get("num_edges", 0)) or 0) for r in built)
        ),
        "num_hard_pruned_edges": int(sum(int(r.get("num_hard_pruned_edges", 0) or 0) for r in built)),
        "num_task_candidate_edges": int(
            sum(int(r.get("num_task_candidate_edges", r.get("num_interaction_candidate_edges", 0)) or 0) for r in built)
        ),
        "num_soft_task_excluded_edges": int(
            sum(int(r.get("num_soft_task_excluded_edges", 0) or 0) for r in built)
        ),
        "num_label_computable_edges": int(
            sum(int(r.get("num_label_computable_edges", r.get("num_edges", 0)) or 0) for r in built)
        ),
        "num_proximity_positive_edges": int(sum(int(r.get("num_proximity_positive_edges", 0) or 0) for r in built)),
        "num_proximity_negative_edges": int(sum(int(r.get("num_proximity_negative_edges", 0) or 0) for r in built)),
        "num_risk_positive_edges": int(sum(int(r.get("num_risk_positive_edges", 0) or 0) for r in built)),
        "num_risk_negative_edges": int(sum(int(r.get("num_risk_negative_edges", 0) or 0) for r in built)),
        "num_supervision_edges": int(sum(int(r.get("num_supervision_edges", 0) or 0) for r in built)),
        "num_filtered_proximity_positive_edges": int(
            sum(int(r.get("num_filtered_proximity_positive_edges", 0) or 0) for r in built)
        ),
        "num_hard_pruned_proximity_positive_edges": int(
            sum(int(r.get("num_hard_pruned_proximity_positive_edges", 0) or 0) for r in built)
        ),
        "num_soft_excluded_proximity_positive_edges": int(
            sum(int(r.get("num_soft_excluded_proximity_positive_edges", 0) or 0) for r in built)
        ),
        "num_interaction_candidate_edges": int(sum(int(r.get("num_interaction_candidate_edges", 0) or 0) for r in built)),
        "num_interaction_candidate_computable_edges": int(
            sum(int(r.get("num_interaction_candidate_computable_edges", 0) or 0) for r in built)
        ),
        "num_interaction_candidate_proximity_positive_edges": int(
            sum(int(r.get("num_interaction_candidate_proximity_positive_edges", 0) or 0) for r in built)
        ),
        "num_interaction_candidate_proximity_negative_edges": int(
            sum(int(r.get("num_interaction_candidate_proximity_negative_edges", 0) or 0) for r in built)
        ),
        "num_kinematic_candidate_edges": int(sum(int(r.get("num_kinematic_candidate_edges", 0) or 0) for r in built)),
        "num_map_candidate_edges": int(sum(int(r.get("num_map_candidate_edges", 0) or 0) for r in built)),
        "num_map_known_edges": int(sum(int(r.get("num_map_known_edges", 0) or 0) for r in built)),
        "num_map_unknown_edges": int(sum(int(r.get("num_map_unknown_edges", 0) or 0) for r in built)),
        "num_map_blocked_edges": int(sum(int(r.get("num_map_blocked_edges", 0) or 0) for r in built)),
        "num_same_lane_edges": int(sum(int(r.get("num_same_lane_edges", 0) or 0) for r in built)),
        "num_lateral_neighbor_lane_edges": int(sum(int(r.get("num_lateral_neighbor_lane_edges", 0) or 0) for r in built)),
        "num_connected_lane_edges": int(sum(int(r.get("num_connected_lane_edges", 0) or 0) for r in built)),
        "num_intersection_edges": int(sum(int(r.get("num_intersection_edges", 0) or 0) for r in built)),
        "split_counts": {},
        "city_counts": {},
        "skip_reasons": {},
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "split_audit": split_audit,
    }
    if summary["num_edges"]:
        summary["interaction_candidate_retention_rate"] = summary["num_interaction_candidate_edges"] / summary["num_edges"]
        summary["physical_possible_retention_rate"] = (
            summary["num_physical_possible_edges"] / summary["num_raw_edges"]
        )
        summary["task_candidate_retention_rate"] = (
            summary["num_task_candidate_edges"] / summary["num_raw_edges"]
        )
        summary["filtered_proximity_positive_rate"] = (
            summary["num_filtered_proximity_positive_edges"] / max(summary["num_proximity_positive_edges"], 1)
        )
    if summary["num_label_computable_edges"]:
        summary["proximity_positive_rate"] = (
            summary["num_proximity_positive_edges"] / summary["num_label_computable_edges"]
        )
        summary["risk_positive_rate"] = (
            summary["num_risk_positive_edges"] / summary["num_label_computable_edges"]
        )
        summary["physical_positive_recall"] = (
            (summary["num_proximity_positive_edges"] - summary["num_hard_pruned_proximity_positive_edges"])
            / max(summary["num_proximity_positive_edges"], 1)
        )
        summary["task_positive_recall"] = (
            summary["num_interaction_candidate_proximity_positive_edges"]
            / max(summary["num_proximity_positive_edges"], 1)
        )
    if summary["num_interaction_candidate_computable_edges"]:
        summary["interaction_candidate_proximity_positive_rate"] = (
            summary["num_interaction_candidate_proximity_positive_edges"]
            / summary["num_interaction_candidate_computable_edges"]
        )
    for r in rows:
        summary["split_counts"][r.get("split", "")] = summary["split_counts"].get(r.get("split", ""), 0) + 1
        summary["city_counts"][r.get("city", "unknown")] = summary["city_counts"].get(r.get("city", "unknown"), 0) + 1
        if r.get("status") == "skipped":
            reason = r.get("reason", "unknown")
            summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1

    summary_path = output_dir / summary_name
    _atomic_json_write(summary, summary_path)
    return manifest_path, summary_path


def worker(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    scenario_dir, cfg_dict = args
    cfg = BuildConfig(**cfg_dict)
    try:
        return build_scenario_graph(Path(scenario_dir), cfg)
    except Exception as exc:  # keep batch builds running; inspect manifest later
        return {
            "scenario_id": Path(scenario_dir).name,
            "status": "error",
            "reason": type(exc).__name__,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AV2-MF vehicle interaction graph-v4 files.")
    parser.add_argument("--data-root", default="data/train", help="Directory containing AV2 scenario folders.")
    parser.add_argument("--output-dir", default="graphs/av2mf_graph_v4", help="Output graph directory.")
    parser.add_argument("--radius-m", type=float, default=60.0)
    parser.add_argument("--future-risk-distance-m", type=float, default=10.0)
    parser.add_argument("--min-observed", type=int, default=40)
    parser.add_argument("--min-last-observed-timestep", type=int, default=48)
    parser.add_argument("--min-pair-future", type=int, default=30)
    parser.add_argument("--candidate-close-distance-m", type=float, default=35.0)
    parser.add_argument("--candidate-recent-close-distance-m", type=float, default=30.0)
    parser.add_argument("--candidate-far-distance-m", type=float, default=40.0)
    parser.add_argument("--candidate-dynamic-distance-m", type=float, default=50.0)
    parser.add_argument("--candidate-approach-trend-m-per-step", type=float, default=-0.02)
    parser.add_argument("--candidate-receding-trend-m-per-step", type=float, default=0.03)
    parser.add_argument("--candidate-closing-speed-mps", type=float, default=0.5)
    parser.add_argument("--candidate-closing-speed-max-mps", type=float, default=1.5)
    parser.add_argument("--candidate-perpendicular-min-distance-m", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=None, help="Build only the first N scenarios.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument(
        "--split-group-metadata",
        default=(
            "record/v3/"
            "duplicate_scene_check_overlap_v4.parquet"
        ),
        help=(
            "CSV/Parquet with scenario_id, city, recording/content/focal group "
            "keys. Required for leakage-resistant group splits."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_output_dir = Path(args.output_dir)
    validate_generation_output_dir(requested_output_dir)
    isolated_output_dir = effective_output_dir(requested_output_dir, args.limit)
    cfg = BuildConfig(
        data_root=args.data_root,
        output_dir=str(isolated_output_dir),
        radius_m=args.radius_m,
        future_risk_distance_m=args.future_risk_distance_m,
        min_observed=args.min_observed,
        min_last_observed_timestep=args.min_last_observed_timestep,
        min_pair_future=args.min_pair_future,
        candidate_close_distance_m=args.candidate_close_distance_m,
        candidate_recent_close_distance_m=args.candidate_recent_close_distance_m,
        candidate_far_distance_m=args.candidate_far_distance_m,
        candidate_dynamic_distance_m=args.candidate_dynamic_distance_m,
        candidate_approach_trend_m_per_step=args.candidate_approach_trend_m_per_step,
        candidate_receding_trend_m_per_step=args.candidate_receding_trend_m_per_step,
        candidate_closing_speed_mps=args.candidate_closing_speed_mps,
        candidate_closing_speed_max_mps=args.candidate_closing_speed_max_mps,
        candidate_perpendicular_min_distance_m=args.candidate_perpendicular_min_distance_m,
        seed=args.seed,
        split_group_metadata=args.split_group_metadata,
        overwrite=args.overwrite,
    )

    data_root = Path(cfg.data_root)
    output_dir = Path(cfg.output_dir)
    dirs = scenario_dirs(data_root, args.limit)
    cfg_dict = asdict(cfg)
    rows: list[dict[str, Any]] = []

    if not cfg.overwrite:
        for scenario_dir in dirs:
            existing_path = output_dir / "graphs" / f"{scenario_dir.name}.npz"
            if not existing_path.exists():
                continue
            if load_existing_graph_row(existing_path, cfg) is None:
                raise RuntimeError(
                    f"Existing graph {existing_path} is incompatible with graph schema "
                    f"v{GRAPH_SCHEMA_VERSION} or the requested configuration. Use a new "
                    "--output-dir or pass --overwrite; old graphs will not be reused silently."
                )
            break

    print(
        f"[build_graph] data_root={data_root} scenarios={len(dirs)} "
        f"output={output_dir}"
    )
    if args.num_workers <= 1:
        iterator: Iterable[dict[str, Any]] = (worker((str(d), cfg_dict)) for d in dirs)
        for idx, row in enumerate(iterator, start=1):
            rows.append(row)
            if idx % 100 == 0 or idx == len(dirs):
                print(f"[build_graph] processed {idx}/{len(dirs)}")
    else:
        with Pool(args.num_workers) as pool:
            for idx, row in enumerate(pool.imap_unordered(worker, [(str(d), cfg_dict) for d in dirs]), start=1):
                rows.append(row)
                if idx % 100 == 0 or idx == len(dirs):
                    print(f"[build_graph] processed {idx}/{len(dirs)}")

    # A smoke/partial build lives under an isolated graph root and must never
    # replace either complete graph files or the complete dataset index.
    manifest_name, summary_name = manifest_output_names(args.limit)
    built = sum(1 for r in rows if r.get("status") in {"built", "exists"})
    skipped = sum(1 for r in rows if r.get("status") == "skipped")
    errors = sum(1 for r in rows if r.get("status") == "error")
    if errors:
        manifest_name = str(Path(manifest_name).with_suffix(".failed.csv"))
        summary_name = str(Path(summary_name).with_suffix(".failed.json"))
    manifest_path, summary_path = write_manifest(
        rows,
        output_dir,
        cfg,
        manifest_name=manifest_name,
        summary_name=summary_name,
    )
    print(f"[build_graph] done built={built} skipped={skipped} errors={errors}")
    print(f"[build_graph] manifest={manifest_path}")
    print(f"[build_graph] summary={summary_path}")
    if errors:
        raise RuntimeError(
            f"graph build produced {errors} error rows; active manifest was not "
            "published"
        )


if __name__ == "__main__":
    main()
