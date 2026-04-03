# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any, Iterable

from megatron.rl.agent.reward_only_agent import RewardOnlyAgent


class SmokeAgent(RewardOnlyAgent):
    env_id: str = "smoke"

    def __init__(self, prompt: str = "Answer with HELLO.", expected_substring: str = "HELLO", **kwargs):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.expected_substring = expected_substring
        self._dataset = [
            {
                "problem_id": "smoke-0",
                "prompt": self.prompt,
                "expected_substring": self.expected_substring,
            }
        ]

    def get_dataset(self, validation: bool = False):
        return self._dataset

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> Iterable[tuple[str, Any]]:
        dataset = self.get_dataset(validation)
        sample = dataset[0]
        return [(sample["prompt"], sample) for _ in range(num_prompts)]

    async def get_prompt(self, validation: bool = False) -> tuple[str, dict]:
        sample = self.get_dataset(validation)[0]
        return sample["prompt"], sample

    async def get_reward(self, response: str, golden: dict) -> float:
        return 1.0 if golden["expected_substring"] in response else 0.0
