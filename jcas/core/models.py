#!/usr/bin/env python3
"""GENConv model for dynamic vehicle-pair risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GENConv
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    Tensor = Any
    nn = None
    F = None
    GENConv = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


MODEL_NAME = "genconv"


@dataclass(frozen=True)
class ModelConfig:
    node_dim: int
    edge_dim: int
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.10
    norm: str = "layer"
    decoder_hidden_dim: int = 128
    decoder_num_layers: int = 2
    gen_aggr: str = "softmax"
    gen_learn_t: bool = True
    gen_msg_norm: bool = True
    gen_learn_msg_scale: bool = True
    gen_mlp_layers: int = 2
    gen_expansion: int = 2


def model_kwargs_from_result(
    result: dict[str, Any],
    node_dim: int,
    edge_dim: int,
    fallback_model_name: str | None = None,
) -> dict[str, Any]:
    cfg = dict(result.get("model_config") or result.get("config") or {})
    model_name = str(
        cfg.get("model_name", fallback_model_name or MODEL_NAME)
    ).lower()
    if model_name != MODEL_NAME:
        raise ValueError("the main experiment supports only GENConv")
    defaults = ModelConfig(node_dim=int(node_dim), edge_dim=int(edge_dim))
    keys = (
        "hidden_dim",
        "num_layers",
        "dropout",
        "norm",
        "decoder_hidden_dim",
        "decoder_num_layers",
        "gen_aggr",
        "gen_learn_t",
        "gen_msg_norm",
        "gen_learn_msg_scale",
        "gen_mlp_layers",
        "gen_expansion",
    )
    kwargs: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "node_dim": int(node_dim),
        "edge_dim": int(edge_dim),
    }
    for key in keys:
        kwargs[key] = cfg.get(key, getattr(defaults, key))
    return kwargs


def _raise_missing_dependency() -> None:
    raise ModuleNotFoundError(
        "models.py requires torch and torch_geometric: "
        f"{_IMPORT_ERROR!r}"
    )


if nn is not None:

    def _make_norm(norm: str, hidden_dim: int) -> nn.Module:
        normalized = str(norm).lower()
        if normalized == "layer":
            return nn.LayerNorm(hidden_dim)
        if normalized == "batch":
            return nn.BatchNorm1d(hidden_dim)
        if normalized == "identity":
            return nn.Identity()
        raise ValueError(
            "norm must be one of: layer, batch, identity"
        )


    class MLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2,
            dropout: float = 0.0,
            norm: str = "layer",
        ) -> None:
            super().__init__()
            if int(num_layers) < 1:
                raise ValueError("num_layers must be at least one")
            layers: list[nn.Module] = []
            if int(num_layers) == 1:
                layers.append(nn.Linear(input_dim, output_dim))
            else:
                dimensions = (
                    [input_dim]
                    + [hidden_dim] * (int(num_layers) - 1)
                    + [output_dim]
                )
                for index in range(len(dimensions) - 1):
                    layers.append(
                        nn.Linear(
                            dimensions[index], dimensions[index + 1]
                        )
                    )
                    if index < len(dimensions) - 2:
                        layers.append(
                            _make_norm(norm, dimensions[index + 1])
                        )
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)

        @property
        def in_features(self) -> int:
            return int(self.net[0].in_features)

        def forward(self, values: Tensor) -> Tensor:
            return self.net(values)


    class EdgeDecoder(nn.Module):
        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.mlp = MLP(
                input_dim=cfg.hidden_dim * 5,
                hidden_dim=cfg.decoder_hidden_dim,
                output_dim=1,
                num_layers=cfg.decoder_num_layers,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )

        def forward(
            self,
            node_embedding: Tensor,
            edge_index: Tensor,
            edge_embedding: Tensor,
        ) -> Tensor:
            src, dst = edge_index[0], edge_index[1]
            source = node_embedding[src]
            destination = node_embedding[dst]
            pair = torch.cat(
                [
                    source,
                    destination,
                    source - destination,
                    source * destination,
                    edge_embedding,
                ],
                dim=-1,
            )
            return self.mlp(pair).squeeze(-1)


    class GENConvEdgeRiskModel(nn.Module):
        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.node_encoder = MLP(
                cfg.node_dim,
                cfg.hidden_dim,
                cfg.hidden_dim,
                num_layers=2,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )
            self.edge_encoder = MLP(
                cfg.edge_dim,
                cfg.hidden_dim,
                cfg.hidden_dim,
                num_layers=2,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )
            self.convs = nn.ModuleList(
                [
                    GENConv(
                        in_channels=cfg.hidden_dim,
                        out_channels=cfg.hidden_dim,
                        aggr=cfg.gen_aggr,
                        learn_t=cfg.gen_learn_t,
                        msg_norm=cfg.gen_msg_norm,
                        learn_msg_scale=cfg.gen_learn_msg_scale,
                        norm=(
                            None
                            if cfg.norm == "identity"
                            else cfg.norm
                        ),
                        num_layers=cfg.gen_mlp_layers,
                        expansion=cfg.gen_expansion,
                        edge_dim=cfg.hidden_dim,
                    )
                    for _ in range(cfg.num_layers)
                ]
            )
            self.norms = nn.ModuleList(
                [
                    _make_norm(cfg.norm, cfg.hidden_dim)
                    for _ in range(cfg.num_layers)
                ]
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.decoder = EdgeDecoder(cfg)

        def encode_nodes(self, values: Tensor) -> Tensor:
            return self.node_encoder(values)

        def encode_edges(self, values: Tensor) -> Tensor:
            return self.edge_encoder(values)

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            message_edge_mask: Tensor | None = None,
        ) -> Tensor:
            node_hidden = self.encode_nodes(x)
            edge_hidden = self.encode_edges(edge_attr)
            if message_edge_mask is None:
                message_edge_index = edge_index
                message_edge_hidden = edge_hidden
            else:
                if (
                    message_edge_mask.ndim != 1
                    or message_edge_mask.numel() != edge_index.shape[1]
                ):
                    raise ValueError(
                        "message_edge_mask must contain one value per edge"
                    )
                mask = message_edge_mask.to(
                    device=edge_index.device, dtype=torch.bool
                )
                message_edge_index = edge_index[:, mask]
                message_edge_hidden = edge_hidden[mask]
            embedding = node_hidden
            for convolution, normalization in zip(
                self.convs, self.norms
            ):
                update = convolution(
                    embedding,
                    message_edge_index,
                    message_edge_hidden,
                )
                update = self.dropout(
                    F.relu(normalization(update))
                )
                embedding = embedding + update
            return self.decoder(embedding, edge_index, edge_hidden)


    def build_model(
        model_name: str,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.10,
        norm: str = "layer",
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        gen_aggr: str = "softmax",
        gen_learn_t: bool = True,
        gen_msg_norm: bool = True,
        gen_learn_msg_scale: bool = True,
        gen_mlp_layers: int = 2,
        gen_expansion: int = 2,
    ) -> nn.Module:
        if str(model_name).lower() != MODEL_NAME:
            raise ValueError("the main experiment supports only GENConv")
        config = ModelConfig(
            node_dim=int(node_dim),
            edge_dim=int(edge_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            norm=str(norm),
            decoder_hidden_dim=int(decoder_hidden_dim),
            decoder_num_layers=int(decoder_num_layers),
            gen_aggr=str(gen_aggr),
            gen_learn_t=bool(gen_learn_t),
            gen_msg_norm=bool(gen_msg_norm),
            gen_learn_msg_scale=bool(gen_learn_msg_scale),
            gen_mlp_layers=int(gen_mlp_layers),
            gen_expansion=int(gen_expansion),
        )
        return GENConvEdgeRiskModel(config)


else:

    class GENConvEdgeRiskModel:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_missing_dependency()


    def build_model(  # pragma: no cover
        *args: Any, **kwargs: Any
    ) -> Any:
        _raise_missing_dependency()
