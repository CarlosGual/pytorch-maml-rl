import matplotlib.pyplot as plt
import numpy as np
import pickle

from maml_rl.utils.torch_utils import weighted_mean, to_numpy

prefixes = ['maml', 'pretrain']

n_itr = 4

with open('2d-nav3/results.pkl', 'rb') as f:
    file = pickle.load(f)

fig = plt.figure(figsize=(10, 10))
plt.clf()
itr_line_styles = [':', '-.', '--', '-']
maml_colors = ['dodgerblue', None, None, 'darkblue']
pretrain_colors = ['limegreen', None, None, 'darkgreen']

#plt.figure(figsize=(9.0,4.5))
batch = 0
meta_batch = 12
rollout = 8
if batch == 0 and meta_batch == 0:
    task = 0
elif batch == 0 and meta_batch != 0:
    task = meta_batch
elif batch != 0 and meta_batch == 0:
    task = batch * 20
else:
    task = batch * meta_batch


# get last non zero index
last_non_zero_idx_obs = int(np.max(np.nonzero(to_numpy(file['valid_episodes'][batch][meta_batch].actions[:, rollout, :]))))

# Calculate last state (last observation of the agent, e.g., last position)
last_state = np.clip(to_numpy(file['valid_episodes'][batch][meta_batch].actions[last_non_zero_idx_obs, rollout, :]), -0.1, 0.1) + \
             to_numpy(file['valid_episodes'][batch][meta_batch].observations[last_non_zero_idx_obs, rollout, :])

paths1 = to_numpy(file['train_episodes'][batch][meta_batch].observations[:, rollout, :])
plt.plot(paths1[:, 0], paths1[:, 1], itr_line_styles[0], color=maml_colors[0], linewidth=2)
paths2 = to_numpy(file['valid_episodes'][batch][meta_batch].observations[:, rollout, :])
paths2[last_non_zero_idx_obs + 1] = last_state
paths2 = paths2[0:last_non_zero_idx_obs + 2, :]
plt.plot(paths2[:, 0], paths2[:, 1], itr_line_styles[1], color=maml_colors[1], linewidth=2)

plt.plot(file['tasks'][task]['goal'][0], file['tasks'][task]['goal'][1], 'r*', markersize=28, markeredgewidth=0)
plt.title('MAML', fontsize=25)
plt.legend(['pre-update',  '3 steps', 'goal position'], fontsize=15, bbox_to_anchor=(1,0), loc="lower right",
                bbox_transform=fig.transFigure, ncol=3) #, 'pretrain preupdate', 'pretrain 3 steps'])
# plt.xlim([-0.7, 0.2])
# plt.ylim([-0.15, 0.3])
plt.tight_layout()
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.show()
# plt.savefig('maml_paths_viz.png')


