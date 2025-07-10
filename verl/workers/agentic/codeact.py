import json
import asyncio
from pathlib import Path
from typing import Any, Dict
import torch
import os
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import logging


from .agent import OfflineRolloutAgent


def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(name)s-%(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger(__name__)


# this is for the tokenizer.apply_chat_template to be able to generate assistant masks directly
# todo: this is a hack, we should find a better way to do this
chat_template = (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{{message['content'] + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    )

# chat template for qwen3 thinking mode to remove think tokens similar to generation phase
chat_template_qwen3_thinking = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% generation %}"
    "{% set full_content = message['content'] %}"
    "{% set mycontent = message['content'] %}"
    "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
    "{% if '</think>' in full_content and not is_last_message %}"
    "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
    "{% endif %}"
    "{{mycontent + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\n'}}"
    "{% endif %}"
    "{% endfor %}"
)

def convert_right_padding_to_left(tokenizer, input_ids, attention_mask, device, max_len=None):
    """
    Converts right-padded tensors to left-padded tensors with optional custom length.
    
    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        input_ids (torch.Tensor): Right-padded input IDs tensor of shape [batch_size, seq_length]
        attention_mask (torch.Tensor): Right-padded attention mask tensor of shape [batch_size, seq_length]
        device: The device to place the new tensors on
        max_len (int, optional): The desired maximum length of the returned tensors.
                                If None, uses the original sequence length.
    
    Returns:
        tuple: (left_padded_input_ids, left_padded_attention_mask)
    """
    batch_size, orig_seq_length = input_ids.size()
    
    # Use original length if max_len is not specified
    seq_length = max_len if max_len is not None else orig_seq_length
    
    # Create new tensors with the desired size
    left_padded_input_ids = torch.full((batch_size, seq_length), 
                                     tokenizer.pad_token_id, 
                                     dtype=input_ids.dtype, 
                                     device=device)
    left_padded_attention_mask = torch.zeros((batch_size, seq_length), 
                                           dtype=attention_mask.dtype, 
                                           device=device)
    
    for i in range(batch_size):
        # Get the non-padded length of this sequence
        seq_len = attention_mask[i].sum().item()
        
        # Trim sequence if it's longer than max_len
        if seq_len > seq_length:
            logger.warning(f"Trimming sequence length from {seq_len} to {seq_length}")
            seq_len = seq_length
        
        # Calculate the offset for left padding
        offset = seq_length - seq_len
        
        # Copy the non-padded tokens to the end
        left_padded_input_ids[i, offset:] = input_ids[i, :seq_len]
        left_padded_attention_mask[i, offset:] = 1  # Set attention mask for non-padding tokens
    
    return left_padded_input_ids, left_padded_attention_mask

def pad_to_max_length_right(tokenizer, encodings, max_length, device):
    """
    Pads tokenizer outputs to a specific maximum length with configurable padding side.
    
    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        encodings (dict): Dictionary containing 'input_ids', 'attention_mask', and optionally 'assistant_masks'
        max_length (int): The desired maximum length to pad to
        device: The device to place the tensors on
        
    Returns:
        dict: Dictionary with padded tensors for 'input_ids', 'attention_mask', and 'assistant_masks' if present
    """
    batch_size = len(encodings['input_ids'])
    
    # Initialize output tensors
    padded_input_ids = torch.full((batch_size, max_length), 
                                tokenizer.pad_token_id, 
                                dtype=torch.long, 
                                device=device)
    padded_attention_mask = torch.zeros((batch_size, max_length), 
                                      dtype=torch.long, 
                                      device=device)
    padded_assistant_mask = torch.zeros((batch_size, max_length), 
                                          dtype=torch.long, 
                                          device=device)
    
    # Fill tensors with actual values
    num_trimmed = 0
    for i in range(batch_size):
        seq_len = encodings["attention_mask"][i].sum().item() if isinstance(encodings["attention_mask"][i], torch.Tensor) else sum(encodings["attention_mask"][i])
        # Trim if longer than max_length
        actual_len = min(seq_len, max_length)
        if seq_len > max_length:
            logger.warning(
                f"Trimming sequence length from {seq_len} to {actual_len} for batch item {i}"
            )
            num_trimmed += 1
        
        # Right padding - copy sequence data to the beginning
        padded_input_ids[i, :actual_len] = torch.tensor(encodings['input_ids'][i][:actual_len], device=device)
        padded_attention_mask[i, :actual_len] = torch.tensor(encodings['attention_mask'][i][:actual_len], device=device)
        padded_assistant_mask[i, :actual_len] = torch.tensor(encodings['assistant_masks'][i][:actual_len], device=device)
    
    logger.info(f"Trimmed {num_trimmed*100 / max(batch_size, 1)}% of samples in the batch of size {batch_size}")
    return padded_input_ids, padded_attention_mask, padded_assistant_mask


class CodeActAgentGroup:
    """
    A class that manages multiple CodeActAgent instances to generate trajectories in parallel.
    """
    
    def __init__(
        self,
        batch: DataProto,
        num_trajectories: int,
        infer_engine: Any,
        max_prompt_length: int = 1024,
        max_response_length: int = 1024,
        max_starting_message_length: int = 12000,
        max_parallel_agents: int = 1,
        max_eval_parallel_agents: int = 1,
        max_iterations: int = 10,
        tokenizer: Any = None,
        sampling_params: Any = None,
        device: Any = None,
        log_messages_dir: str = None,
        remove_think_tokens: bool = False,
        qwen3_enable_thinking: bool = True
    ) -> None:
        """
        Initialize the CodeActAgentGroup to manage multiple agent instances.
        
        Args:
            batch: DataProto containing the batch of data
            num_trajectories: Number of trajectories to generate per instance
            infer_engine: The infer engine for generation
            max_prompt_length: Maximum prompt length
            max_parallel_agents: Maximum number of agents to run in parallel
            max_iterations: Maximum number of iterations per agent
            tokenizer: Tokenizer to use for text encoding/decoding
            max_batch_size: Maximum batch size for LLM generation
        """
        self.batch = batch
        self.infer_engine = infer_engine
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.total_len = self.max_prompt_length + self.max_response_length
        # todo: make it a config
        self.max_starting_message_length = max_starting_message_length
        self.max_parallel_agents = max_parallel_agents
        self.max_eval_parallel_agents = max_eval_parallel_agents
        logger.info(f"max eval parallel agents: {self.max_eval_parallel_agents}")
        if max_eval_parallel_agents <= 0: 
            logger.info(f"`max_eval_parallel_agents` has not been set. Setting it to `max_parallel_agents` i.e {max_parallel_agents}")
            self.max_eval_parallel_agents = max_parallel_agents
        self.max_iterations = max_iterations
        self.num_trajectories = num_trajectories
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.device = device
        
        # Map of instance ID to agent instance
        self.agents = {}
        
        # Map of instance ID to agent results
        self.results = {}
        
        self.qwen3_enable_thinking = qwen3_enable_thinking
        self.log_messages_dir = None
        if log_messages_dir:
            self.log_messages_dir = Path(log_messages_dir)
            logger.info(f"Logging all messages to {self.log_messages_dir}")

        self.remove_think_tokens = remove_think_tokens
        if self.remove_think_tokens:
            logger.info("Removing think tokens....")


    def _convert_results_to_dataproto(self) -> DataProto:
        """
        Convert results to DataProto format for training.
        
        Args:
            results: Dictionary of results, with structure {instance_id: {trajectory_id: result_dict}}
            input_dataproto: The input DataProto that contains the original batch data
            tokenizer: The tokenizer to use for encoding messages
            
        Returns:
            DataProto: A DataProto object with the converted results
        """

        # Non-tensor data
        git_patch_list = []
        success_list = []
        error_list = []
        resolved_list = []
        has_finish_action_list = []
        stop_reason_list = []
        
        # Create the final results in the same order as the batch
        matched_results: list[dict[str, Any]] = []
        instance_list: list[dict[str, Any]] = []
        for batch_item in self.batch:
            instance = json.loads(batch_item.non_tensor_batch['instance'])
            instance_id = instance['instance_id']
            if instance_id in self.results:
                # Add all trajectories for this instance
                matched_results.extend(list(self.results[instance_id].values()))
                instance_list.extend([instance] * len(self.results[instance_id]))
        
        assert len(matched_results) == self.num_trajectories * len(self.batch), f"Expected number of results {self.num_trajectories * len(self.batch)}, got {len(matched_results)}"
        
        SAVE_PATH = "samples"
        os.makedirs(SAVE_PATH, exist_ok=True)
        for iid, results in self.results.items():
            with open(os.path.join(SAVE_PATH, f"{iid}.json"), "w") as f:
                json.dump(results, f)

        # Handle empty messages by copying from another trajectory of the same instance
        for offset in range(len(self.batch)):
            start_idx = offset * self.num_trajectories
            end_idx = (offset + 1) * self.num_trajectories
            instance_trajectories = matched_results[start_idx:end_idx]
            
            # Find first valid result to use as fallback
            valid_result = next((result for result in instance_trajectories 
                               if result.get('messages') and len(result.get('messages', [])) > 0), None)
            

            for i, result in enumerate(instance_trajectories):
                if not result.get('messages') or len(result.get('messages', [])) == 0:
                    instance_id = instance_list[start_idx + i]['instance_id']
                    if valid_result:
                        logger.warning(f"Got empty messages for instance_id {instance_id}, trajectory {i}. Copying messages array from a valid trajectory.")
                        matched_results[start_idx + i].update({
                            'messages': valid_result['messages'].copy(),
                            'git_patch': valid_result.get('git_patch'),
                            'resolved': valid_result.get('resolved', False),
                            'error': valid_result.get('error'),
                            'finish': valid_result.get('finish', False),
                            'stop_reason': valid_result.get('stop_reason', '')
                        })
                    else:
                        raise ValueError(f"Got empty messages for instance_id {instance_id}, trajectory {i}. No valid trajectory found.")
        
        # Get batch of messages
        all_messages = []
        all_prompts = []
        all_responses = []
        for result in matched_results:
            messages = result.get('messages', [])
            all_messages.append(messages)
            # get the response: starting from the first assistant message
            starting_index = 0
            for i, msg in enumerate(messages):
                if msg["role"] == 'assistant':
                    starting_index = i
                    break
            if starting_index == 0:
                # If we don't find an assistant, all messages are prompts and there are no responses
                logger.error(f'Found no assistant message {instance_id}. len(messages) == {len(messages)} and roles are {[msg["role"] for msg in messages]}')
                starting_index = len(messages)
            prompt = messages[:starting_index]
            all_prompts.append(prompt)
            response = messages[starting_index:]
            all_responses.append(response)

            # Also add non-tensor data
            git_patch_list.append(result.get('git_patch', None))
            success_list.append(result.get('success', False))
            error_list.append(result.get('error', None))
            resolved_list.append(result.get('resolved', False))
            has_finish_action_list.append(result.get('finish', False))
            stop_reason_list.append(result.get('stop_reason', ''))

        # Encode messages, get assitant mask and position ids
        prompt_encodings = self.tokenizer.apply_chat_template(
            all_prompts, 
            # return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
            padding=True
        )
        prompt_input_ids = torch.tensor(prompt_encodings['input_ids'], device=self.device)
        prompt_attention_mask = torch.tensor(prompt_encodings['attention_mask'], device=self.device)
        prompt_input_ids, prompt_attention_mask = convert_right_padding_to_left(self.tokenizer, prompt_input_ids, prompt_attention_mask, self.device, self.max_starting_message_length)

        response_encodings = self.tokenizer.apply_chat_template(
            all_responses,
            chat_template=chat_template_qwen3_thinking if self.remove_think_tokens else chat_template,
            # return_tensors="pt",
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_dict=True,
            padding=True
        )
        
        response_ids, response_attention_mask, response_assistant_mask = pad_to_max_length_right(
            self.tokenizer, response_encodings, self.total_len, self.device)
            
        
        input_ids = torch.cat([prompt_input_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Create tensor dictionary
        logger.info(f"input_ids shape: {input_ids.shape}, response_ids shape: {response_ids.shape}, max_starting_message_length: {self.max_starting_message_length}, max_response_length: {self.total_len}")
        assert input_ids.shape[1] == attention_mask.shape[1] == position_ids.shape[1], f"input_ids shape {input_ids.shape}, attention_mask shape {attention_mask.shape}, position_ids shape {position_ids.shape} do not match"
        assert response_ids.shape[1] == response_assistant_mask.shape[1], f"response_ids shape {response_ids.shape}, response_assistant_mask shape {response_assistant_mask.shape} do not match"
        tensor_dict = {
            'input_ids': input_ids,
            'responses': response_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': response_assistant_mask,
        }

        # Create non-tensor dictionary
        non_tensor_dict = {
            'git_patch': git_patch_list,
            'success': success_list,
            'error': error_list,
            'instance': instance_list,
            'resolved': resolved_list,
            'finish': has_finish_action_list,
            'stop_reason': stop_reason_list,
        }
        
        # Create and return DataProto
        result_dataproto = DataProto.from_dict(
            tensors=tensor_dict,
            non_tensors=non_tensor_dict
        )
        
        return result_dataproto
        
    def close(self):
        """Clean up resources"""
            
        # Close all agent instances
        for instance_id in self.agents:
            for trajectory_id in self.agents[instance_id]:
                self._cleanup_agent(instance_id, trajectory_id)
    
    def _cleanup_agent(self, instance_id, trajectory_id):
        pass
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        self.close()

    async def generate_trajectories_pipeline(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        
        init_queue = asyncio.Queue()
        complete_queue = asyncio.Queue()
        self.results = {}
        for data_item in self.batch:
            instance = json.loads(data_item.non_tensor_batch['instance'])
            for i in range(self.num_trajectories):
                await init_queue.put((instance, i))
            self.results[instance['instance_id']] = {}
        
        for _ in range(self.max_parallel_agents):
            await init_queue.put(None)

        MAX_RETRIES = 3 
        async def initialize_one_agent():
            while True:
                instance_n_tid = await init_queue.get()
                if not instance_n_tid:
                    await complete_queue.put(None)
                    break
                instance, trajectory_id = instance_n_tid
                assert isinstance(instance, dict), f"instance {instance} is not a dict"
                assert 'data_source' in instance, f"data_source not found in instance {instance}"
                logger.info(f"Initializing agent for {instance['instance_id']}-{trajectory_id}")
                agent = OfflineRolloutAgent(
                        logger=logger,
                        instance=instance,
                        trajectory_id=trajectory_id,
                        max_prompt_length=self.max_prompt_length,
                        tokenizer=self.tokenizer,
                        infer_engine=self.infer_engine,
                        sampling_params=self.sampling_params,
                        qwen3_enable_thinking=self.qwen3_enable_thinking,
                        max_iter=self.max_iterations
                )
                current_retries = 0
                while current_retries < MAX_RETRIES:
                    try:
                        await agent.initialize()
                        history, stop_reason = await agent.run()   # It will automatically retry if it fails
                        await complete_queue.put((agent, instance, trajectory_id, history, stop_reason))
                        break # Break out of the retry loop
                    except Exception as e:
                        current_retries += 1
                        if current_retries >= MAX_RETRIES:
                            raise e
                        logger.error(f"Error in agent {instance['instance_id']}-{trajectory_id}: {e}. Retrying {current_retries} / {MAX_RETRIES}...")
                        await asyncio.sleep(60)

                await asyncio.sleep(0.1)
        
        async def complete_one_agent():
            while True:
                params = await complete_queue.get()
                if not params:
                    break
                agent, instance, trajectory_id, history, stop_reason = params
                logger.info(f"Completing agent for {instance['instance_id']}-{trajectory_id}")

                current_retries = 0
                while current_retries < MAX_RETRIES:
                    try:
                        patch, report = await agent.complete()
                        break
                    except Exception as e:
                        current_retries += 1
                        if current_retries >= MAX_RETRIES:
                            raise e
                        logger.error(f"Error in agent {instance['instance_id']}-{trajectory_id}: {e}. Retrying {current_retries} / {MAX_RETRIES}...")
                        await asyncio.sleep(60)
                        await agent.redo_previous_actions() # Redo previous actions to recover from the error

                if report['status'] != 'success':
                    error_msg = report.get('error', "")
                    error = report['status'] + (f": {error_msg}" if error_msg else "")
                else:
                    error = None
                
                self.results[instance['instance_id']][trajectory_id] = {
                    'messages': history,
                    'git_patch': patch,
                    'resolved': report.get('resolved', False),
                    'error': error,
                    'finish': stop_reason == "Model finished",
                    'stop_reason': stop_reason,
                    'success': report['status'] != 'error'
                }
                await asyncio.sleep(0.1)

        init_tasks = [initialize_one_agent() for _ in range(self.max_parallel_agents)]
        complete_tasks = [complete_one_agent() for _ in range(self.max_parallel_agents)]
        task_refs = [asyncio.create_task(task) for task in init_tasks + complete_tasks]

        total_trajectories = len(self.batch) * self.num_trajectories
        while not all(task.done() for task in task_refs):
            finished = sum(len(result) for result in self.results.values())
            pending = init_queue.qsize()
            running = total_trajectories - finished - pending
            logger.info(f"Finished {finished} / {total_trajectories} trajectories, pending {pending}, running {running}")
            await asyncio.sleep(10)
        results_dataproto = self._convert_results_to_dataproto()
        return results_dataproto
    
    def run(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Run the agent group synchronously by creating a new event loop if necessary.
        
        Returns:
            Dict mapping instance ID to a dict of trajectory ID to results
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists in this thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the generate_trajectories coroutine in the event loop
        try:
            return loop.run_until_complete(self.generate_trajectories_pipeline())
        finally:
            # Close the batch manager to ensure cleanup
            self.close()
            # loop.close()