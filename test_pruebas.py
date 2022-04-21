import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

# Args
output = '2d-nav/results.npz'
config = '2d-nav/config.json'
policy_path = '2d-nav/policy.th'
seed = 1
num_workers = 11
device = 'cpu'
num_batches = 10
meta_batch_size = 20
grad_steps = 3

with open(config, 'r') as f:
    config = json.load(f)

if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

env = gym.make(config['env-name'], **config['env-kwargs'])
env.close()

# Policy
policy = get_policy_for_env(env,
                            hidden_sizes=config['hidden-sizes'],
                            nonlinearity=config['nonlinearity'])
with open(policy_path, 'rb') as f:
    state_dict = torch.load(f, map_location=torch.device(device))
    policy.load_state_dict(state_dict)
policy.share_memory()

# Baseline
baseline = LinearFeatureBaseline(get_input_size(env))

# Sampler
sampler = MultiTaskSampler(config['env-name'],
                           env_kwargs=config['env-kwargs'],
                           batch_size=config['fast-batch-size'],
                           policy=policy,
                           baseline=baseline,
                           env=env,
                           seed=seed,
                           num_workers=num_workers)

logs = {'tasks': []}
train_returns, valid_returns = [], []
for batch in trange(num_batches):
    tasks = sampler.sample_tasks(num_tasks=meta_batch_size)
    train_episodes, valid_episodes = sampler.sample(tasks,
                                                    num_steps=grad_steps,
                                                    fast_lr=config['fast-lr'],
                                                    gamma=config['gamma'],
                                                    gae_lambda=config['gae-lambda'],
                                                    device=device)

    logs['tasks'].extend(tasks)
    train_returns.append(get_returns(train_episodes[0]))
    valid_returns.append(get_returns(valid_episodes))

logs['train_returns'] = np.concatenate(train_returns, axis=0)
logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

with open(output, 'wb') as f:
    np.savez(f, **logs)