# GRPO Agentic Rollout Mock Dataset (AI Generated)

This folder contains synthetic rollout data generated locally to simulate GRPO training data.

## Files
- `grpo_grouped_rollouts_4x16.jsonl`: 4 JSONL rows, each row is one rollout group (`group_size=16`).
- `grpo_flat_rollouts_64.jsonl`: 64 JSONL rows, one row per rollout sample.
- `metadata.json`: generation metadata and references.

## Schema Highlights
Each grouped row includes:
- `group_id`, `group_size`, `prompt`, `rollouts`
- group-level reward stats: `reward_mean`, `reward_std`, `advantages`

Each rollout includes fields inspired by public agentic RL formats:
- Megatron-style rollout basics: `trajectory`, `reward`, `env_id`, `problem_id`, `policy_epoch`, `kv_cache_epoch`, `num_evictions`
- Agentic traces: `messages`, `tool_trace`, `response`
- GRPO/rollout collector style fields: `prompt_ids`, `completion_ids`, `logprobs`, `raw_rewards`, `group_advantage`

## Important
`AI Generated`: All content is synthetic and intended only for local simulation/testing.

- Added explicit `tool_definitions` + sandbox-aware `request_prompt` in every rollout sample.
