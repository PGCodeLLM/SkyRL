import logging
from typing import Any

from mindforge_harness.agent.agent import RemoteAgent


RUNTIME_URL = "http://lux-2-cyber-03:9401"

class OfflineRolloutAgent(RemoteAgent):
    
    def __init__(self, 
                 logger: logging.Logger,
                 instance: dict[str, Any], 
                 trajectory_id: int, 
                 max_prompt_length: int, 
                 tokenizer: Any, 
                 infer_engine: Any, 
                 sampling_params: Any, 
                 qwen3_enable_thinking: bool,
                 max_iter: int):
        super().__init__(
            request_url=RUNTIME_URL,
            instance=instance,
            logger=logger.getChild(f"{instance['instance_id']}-{trajectory_id}"),
            max_iter=max_iter,
            context_length=max_prompt_length,
            trajectory_id=trajectory_id,
        )   
        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer
        self.infer_engine = infer_engine
        self.sampling_params = sampling_params
        self.qwen3_enable_thinking = qwen3_enable_thinking

    async def generate(self, messages: list[dict]):
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=self.qwen3_enable_thinking
        )
        if len(input_ids) >= self.max_prompt_length:
            raise ValueError("maximum context length")
        res = await self.infer_engine.async_generate(
            input_ids=input_ids,
            sampling_params=self.sampling_params
        )
        return res["text"], None