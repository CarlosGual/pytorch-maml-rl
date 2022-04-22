import matplotlib.pyplot as plt
import numpy as np
import pickle

from maml_rl.utils.torch_utils import weighted_mean, to_numpy

prefixes = ['maml', 'pretrain']

n_itr = 4

with open('2d-nav3/results.pkl', 'rb') as f:
    file = pickle.load(f)

itr_line_styles = [':', '-.', '--', '-']
maml_colors = ['dodgerblue', None, None, 'darkblue']
pretrain_colors = ['limegreen', None, None, 'darkgreen']

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
last_non_zero_idx_obs = int(
    np.max(np.nonzero(to_numpy(file['valid_episodes'][batch][meta_batch].actions[:, rollout, :]))))

# Calculate last state (last observation of the agent, e.g., last position)
last_state = np.clip(to_numpy(file['valid_episodes'][batch][meta_batch].actions[last_non_zero_idx_obs, rollout, :]),
                     -0.1, 0.1) + \
             to_numpy(file['valid_episodes'][batch][meta_batch].observations[last_non_zero_idx_obs, rollout, :])

paths1 = to_numpy(file['train_episodes'][batch][meta_batch].observations[:, rollout, :])

paths2 = to_numpy(file['valid_episodes'][batch][meta_batch].observations[:, rollout, :])
paths2[last_non_zero_idx_obs + 1] = last_state
paths2 = paths2[0:last_non_zero_idx_obs + 2, :]

filenames = []

for frame in range(2, len(paths1)):
    fig = plt.figure(figsize=(10, 10))
    plt.clf()

    plt.plot(paths1[:frame, 0], paths1[:frame, 1], itr_line_styles[0], color=maml_colors[0], linewidth=2)
    plt.plot(paths2[0, 0], paths2[0, 1], itr_line_styles[1], color=maml_colors[1], linewidth=2)
    plt.plot(file['tasks'][task]['goal'][0], file['tasks'][task]['goal'][1], 'r*', markersize=28, markeredgewidth=0)

    plt.title('MAML', fontsize=25)
    plt.legend(['pre-update', '3 steps', 'goal position'], fontsize=15, bbox_to_anchor=(1, 0), loc="lower right",
               bbox_transform=fig.transFigure, ncol=3)  # , 'pretrain preupdate', 'pretrain 3 steps'])

    plt.xlim([-0.7, 0.2])
    plt.ylim([-0.15, 0.3])
    plt.tight_layout()
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.savefig('images/maml_paths_viz_' + str(frame) + '.png')
    filenames.append('images/maml_paths_viz_' + str(frame) + '.png')
    plt.close(fig)

for frame in range(2, len(paths2) + 1):
    fig = plt.figure(figsize=(10, 10))
    plt.clf()

    plt.plot(paths1[:, 0], paths1[:, 1], itr_line_styles[0], color=maml_colors[0], linewidth=2)
    plt.plot(paths2[:frame, 0], paths2[:frame, 1], itr_line_styles[1], color=maml_colors[1], linewidth=2)
    plt.plot(file['tasks'][task]['goal'][0], file['tasks'][task]['goal'][1], 'r*', markersize=28, markeredgewidth=0)

    plt.title('MAML', fontsize=25)
    plt.legend(['pre-update', '3 steps', 'goal position'], fontsize=15, bbox_to_anchor=(1, 0), loc="lower right",
               bbox_transform=fig.transFigure, ncol=3)  # , 'pretrain preupdate', 'pretrain 3 steps'])

    plt.xlim([-0.7, 0.2])
    plt.ylim([-0.15, 0.3])
    plt.tight_layout()
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.savefig('images/maml_paths_viz_' + str(frame + len(paths1)) + '.png')
    filenames.append('images/maml_paths_viz_' + str(frame + len(paths1)) + '.png')
    plt.close(fig)

import imageio

with imageio.get_writer('maml.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
