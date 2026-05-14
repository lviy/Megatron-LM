# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False

from tests.unit_tests.test_utilities import Utils


class TestMultimodalRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_gpu_init = MultimodalRotaryEmbedding(self.kv_channels, self.rotary_percent)

    def teardown_method(self, method):
        del self.rope_gpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_gpu_init, MultimodalRotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(torch.Tensor(3, 1, 64), mrope_section=[16, 24, 24])
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'


class TestRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.kv_channels = 8
        self.rotary_percent = 1.0
        self.rope_cpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=True
        )
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        del self.rope_cpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_cpu_init, RotaryEmbedding)
        assert self.rope_cpu_init.inv_freq.device.type == 'cpu'
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_forward(self):
        output = self.rope_cpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'


def test_thd_rope_explicit_position_ids_use_token_aligned_freqs():
    config = TransformerConfig(
        num_attention_heads=1,
        num_layers=1,
        apply_rope_fusion=False,
        rotary_interleaved=False,
    )
    t = torch.randn(6, 2, 8, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    freqs = torch.randn(6, 1, 1, 8, dtype=torch.float32)

    out = apply_rotary_pos_emb(
        t,
        freqs,
        config,
        cu_seqlens=cu_seqlens,
        cp_group=None,
        force_unfused=True,
        explicit_position_ids=True,
    )
    expected = _apply_rotary_pos_emb_bshd(t.unsqueeze(1), freqs).squeeze(1)

    assert out.shape == t.shape
    assert torch.allclose(out, expected)


def test_thd_rope_explicit_position_ids_use_tensor_parallel_sequence_shard(monkeypatch):
    from megatron.core.models.common.embeddings import rope_utils

    class _FakeTPGroup:
        def size(self):
            return 2

        def rank(self):
            return 1

    monkeypatch.setattr(
        rope_utils.parallel_state,
        "get_tensor_model_parallel_group",
        lambda check_initialized=False: _FakeTPGroup(),
    )

    config = TransformerConfig(
        num_attention_heads=1,
        num_layers=1,
        apply_rope_fusion=False,
        rotary_interleaved=False,
    )
    t = torch.randn(3, 2, 8, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
    freqs = torch.randn(6, 1, 1, 8, dtype=torch.float32)

    out = apply_rotary_pos_emb(
        t,
        freqs,
        config,
        cu_seqlens=cu_seqlens,
        cp_group=None,
        force_unfused=True,
        explicit_position_ids=True,
    )
    expected = _apply_rotary_pos_emb_bshd(t.unsqueeze(1), freqs[3:6]).squeeze(1)

    assert out.shape == t.shape
    assert torch.allclose(out, expected)


def test_thd_rope_explicit_position_ids_does_not_restore_global_cp_group(monkeypatch):
    from megatron.core.models.common.embeddings import rope_utils

    class _UnexpectedCPGroup:
        def size(self):
            return 2

        def rank(self):
            return 0

    monkeypatch.setattr(
        rope_utils.parallel_state,
        "get_context_parallel_group",
        lambda: _UnexpectedCPGroup(),
    )

    config = TransformerConfig(
        num_attention_heads=1,
        num_layers=1,
        apply_rope_fusion=False,
        rotary_interleaved=False,
    )
    t = torch.randn(6, 2, 8, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    freqs = torch.randn(6, 1, 1, 8, dtype=torch.float32)

    out = apply_rotary_pos_emb(
        t,
        freqs,
        config,
        cu_seqlens=cu_seqlens,
        cp_group=None,
        force_unfused=True,
        explicit_position_ids=True,
    )
    expected = _apply_rotary_pos_emb_bshd(t.unsqueeze(1), freqs).squeeze(1)

    assert out.shape == t.shape
    assert torch.allclose(out, expected)


def test_rotary_embedding_uses_magi_position_ids_for_nonpacked_cp(monkeypatch):
    from megatron.core.models.common.embeddings import rotary_pos_embedding as rotary_mod

    class _FakeCPGroup:
        def size(self):
            return 2

    monkeypatch.setattr(
        rotary_mod.parallel_state,
        "get_context_parallel_group",
        lambda check_initialized=False: _FakeCPGroup(),
    )

    rope = RotaryEmbedding(8, 1.0, use_cpu_initialization=True)
    packed_seq_params = PackedSeqParams(
        qkv_format="sbhd",
        ptm_magi_dist_key=object(),
    )

    fake_indices = torch.tensor([1, 3, 5], dtype=torch.long)
    expected = rope.get_emb(8)[fake_indices]

    monkeypatch.setattr(
        rotary_mod,
        "get_pos_emb_on_this_cp_rank_magi",
        lambda pos_emb, magi_attention_key: pos_emb[fake_indices],
    )

    out = rope(8, packed_seq_params=packed_seq_params)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected)


class TestQKVRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.seq_len = 64
        self.num_heads = 1
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )
        self.transformer_config = TransformerConfig(
            num_attention_heads=self.num_heads, num_layers=1, apply_rope_fusion=True
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAVE_FUSED_QKV_ROPE, reason="Fused QKV RoPE not available.")
    def test_gpu_forward(self):
        pos_embed = self.rope_gpu_init(self.seq_len)
        assert pos_embed.shape[0] == self.seq_len
        assert pos_embed.shape[1] == 1
        assert pos_embed.shape[2] == 1
        assert pos_embed.shape[3] == self.kv_channels
        assert pos_embed.dtype == torch.float32
        assert pos_embed.device.type == 'cuda'

        qkv_split_arg_list = [self.kv_channels * 4, self.kv_channels, self.kv_channels]
        # Create input tensors
        qkv = torch.randn(self.seq_len, 1, self.num_heads, self.kv_channels * 6, device="cuda")
        (query_in, key_in, value_in) = torch.split(qkv, qkv_split_arg_list, dim=3)

        query_in = query_in.reshape(query_in.shape[0], query_in.shape[1], -1, self.kv_channels)
        q_out_ref = apply_rotary_pos_emb(query_in, pos_embed, self.transformer_config)
        k_out_ref = apply_rotary_pos_emb(key_in, pos_embed, self.transformer_config)
        q_out, k_out, _ = apply_fused_qkv_rotary_pos_emb(
            qkv, pos_embed, pos_embed, qkv_split_arg_list
        )

        assert (
            q_out_ref.numel() == q_out.numel()
        ), f"Output sizes do not match for Q: {q_out.shape} != {q_out_ref.shape}"
        assert (
            k_out_ref.numel() == k_out.numel()
        ), f"Output sizes do not match for K: {k_out.shape} != {k_out_ref.shape}"
        assert torch.allclose(q_out_ref, q_out), f"Outputs do not match for Q"
        assert torch.allclose(k_out_ref, k_out), f"Outputs do not match for K"
