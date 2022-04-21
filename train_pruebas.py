import gym
import torch
import json
import gym_miniworld
import os
import yaml
from tqdm import trange

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

# Args
config_file = 'configs/maml/2d-navigation.yaml'
output_folder = '2d-nav2'
seed = 1
num_workers = 11
device = 'cpu'

dict = {
    'config_file': config_file,
    'output_folder': output_folder,
    'seed': seed,
    'num_workers': num_workers,
    'device': device
}

with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if output_folder is not None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    policy_filename = os.path.join(output_folder, 'policy.th')
    config_filename = os.path.join(output_folder, 'config.json')

    with open(config_filename, 'w') as f:
        config.update(dict)
        json.dump(config, f, indent=2)

if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
env.close()

# Policy
policy = get_policy_for_env(env,
                            hidden_sizes=config['hidden-sizes'],
                            nonlinearity=config['nonlinearity'])
policy.share_memory()

# Baseline
baseline = LinearFeatureBaseline(get_input_size(env))

# Sampler
sampler = MultiTaskSampler(config['env-name'],
                           env_kwargs=config.get('env-kwargs', {}),
                           batch_size=config['fast-batch-size'],
                           policy=policy,
                           baseline=baseline,
                           env=env,
                           seed=seed,
                           num_workers=num_workers)

metalearner = MAMLTRPO(policy,
                       fast_lr=config['fast-lr'],
                       first_order=config['first-order'],
                       device=device)

num_iterations = 0
for batch in trange(config['num-batches']):
    tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
    futures = sampler.sample_async(tasks,
                                   num_steps=config['num-steps'],
                                   fast_lr=config['fast-lr'],
                                   gamma=config['gamma'],
                                   gae_lambda=config['gae-lambda'],
                                   device=device)
    logs = metalearner.step(*futures,
                            max_kl=config['max-kl'],
                            cg_iters=config['cg-iters'],
                            cg_damping=config['cg-damping'],
                            ls_max_steps=config['ls-max-steps'],
                            ls_backtrack_ratio=config['ls-backtrack-ratio'])

    train_episodes, valid_episodes = sampler.sample_wait(futures)
    num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
    num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
    logs.update(tasks=tasks,
                num_iterations=num_iterations,
                train_returns=get_returns(train_episodes[0]),
                valid_returns=get_returns(valid_episodes))

    # Save policy
    if output_folder is not None:
        with open(policy_filename, 'wb') as f:
            torch.save(policy.state_dict(), f)

