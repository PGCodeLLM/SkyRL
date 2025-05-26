import os
import ray 
 
os.environ["ALLHANDS_API_KEY"] = "sycao-sandbox-remote"
os.environ["SANDBOX_REMOTE_RUNTIME_API_URL"] = "http://146.235.215.207:8000"

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.reward_manager.swebench import SWEBenchRewardManager
import numpy as np
# from torch.utils.data import DataLoader

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils import hf_tokenizer

DATASET_PATH = "/shared_workspace/datasets/SkyRL-mindforge-harness-data/train.parquet"

from verl.utils import hf_tokenizer
tokenizer = hf_tokenizer("Qwen/Qwen2.5-Coder-7B-Instruct")

dataset = RLHFDataset(parquet_files=DATASET_PATH, tokenizer=tokenizer, prompt_key='prompt', max_prompt_length=256)
# dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

# Fetch the first 1 instances
first = dataset[0]['instance']
second = dataset[1]['instance']

n = 128
non_tensor_batch = {
    "data_source": np.array(['swe-gym', 'swe-gym']*n, dtype=object),
    "ability": np.array(["coding", "coding"]*n, dtype=object),
    "instance": np.array([{"instance_id": first['instance_id']}, {"instance_id": second['instance_id']}]*n, dtype=object),
    "git_patch": np.asarray(["", second['patch']]*n, dtype=object)
}

data = DataProto(
    batch=TensorDict({"responses": torch.randn(2*n)}, batch_size=2*n),
    non_tensor_batch=non_tensor_batch
)

@ray.remote(num_cpus=1)
def task(data):
    manager = SWEBenchRewardManager(None, None, None)
    score, metric = manager.verify_ray(data)
    # print(score)
    assert sum(score) == 1.0*n, "Not passed."
    print(f"Pass reward manager test.")

ray.get(task.remote(data))