import logging
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass
class PrefixTreeMergingContext:
    """Dummy context for wiring Prefix Tree Merging through RL control flow."""

    num_sequences: int
    padded_seq_length: int
    active_token_count: int
    max_active_tokens_per_sequence: int
    first_token_unique_count: int


def build_dummy_prefix_tree_context(
    trajs: torch.Tensor,
    generation_masks: Optional[torch.Tensor],
) -> PrefixTreeMergingContext:
    """Build a dummy PTM context and log that the build path was exercised."""

    if trajs.ndim != 2:
        raise ValueError(f"expected 2D trajectories tensor, got shape={tuple(trajs.shape)}")

    if generation_masks is not None:
        active_lengths = generation_masks.sum(dim=1)
        active_token_count = int(active_lengths.sum().item())
        max_active_tokens_per_sequence = int(active_lengths.max().item()) if active_lengths.numel() else 0
    else:
        active_token_count = int(trajs.numel())
        max_active_tokens_per_sequence = int(trajs.shape[1]) if trajs.numel() else 0

    first_token_unique_count = int(torch.unique(trajs[:, 0]).numel()) if trajs.shape[0] > 0 else 0

    context = PrefixTreeMergingContext(
        num_sequences=int(trajs.shape[0]),
        padded_seq_length=int(trajs.shape[1]),
        active_token_count=active_token_count,
        max_active_tokens_per_sequence=max_active_tokens_per_sequence,
        first_token_unique_count=first_token_unique_count,
    )

    log_single_rank(
        logger,
        logging.INFO,
        "[Prefix Tree Merging][dummy] built context: "
        f"num_sequences={context.num_sequences}, "
        f"padded_seq_length={context.padded_seq_length}, "
        f"active_token_count={context.active_token_count}, "
        f"max_active_tokens_per_sequence={context.max_active_tokens_per_sequence}, "
        f"first_token_unique_count={context.first_token_unique_count}",
    )
    return context


def log_dummy_prefix_tree_path(
    stage: str,
    context: PrefixTreeMergingContext,
    tokens: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> None:
    """Log that a dummy PTM path was exercised for the given stage."""

    token_shape = tuple(tokens.shape) if tokens is not None else None
    position_shape = tuple(position_ids.shape) if position_ids is not None else None
    log_single_rank(
        logger,
        logging.INFO,
        "[Prefix Tree Merging][dummy] "
        f"stage={stage}, "
        f"context=(num_sequences={context.num_sequences}, "
        f"padded_seq_length={context.padded_seq_length}, "
        f"active_token_count={context.active_token_count}), "
        f"tokens_shape={token_shape}, "
        f"position_ids_shape={position_shape}",
    )
