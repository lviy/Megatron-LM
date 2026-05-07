# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass

import torch.distributed as dist
from torch import Tensor


@dataclass
class PackedSeqParams:
    '''
    parameters to TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format
    '''

    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None
    # Optional PTM TreeMask metadata for MagiAttention path.
    ptm_q_ranges: Tensor = None
    ptm_k_ranges: Tensor = None
    ptm_attn_type_map: Tensor = None
    ptm_q_ranges_non_overlapped: bool | None = None
    ptm_magi_dist_key: object = None
    ptm_magi_cp_enabled: bool | None = None
    ptm_magi_global_seqlen: int | None = None
    ptm_magi_pad_size: int | None = None
    ptm_magi_chunk_size: int | None = None
    explicit_position_ids: bool | None = None
