import argparse

import dm_control.rl.control
from dm_control import suite
import numpy as np
from dm_control.suite.wrappers import pixels
from tqdm import tqdm
from typing import *


DATASET_KEYS = ['is_first', 'image', 'reward', 'is_last']
VIEWPOINT_MAPPING = {'birdseye': 0, 'firstperson': 1}

def append_to_episode(episode: Dict[str, List], time_step) -> None:
    episode['is_first'].append(time_step.first())
    episode['image'].append(time_step.observation['pixels'])
    episode['reward'].append(time_step.reward)
    episode['is_last'].append(time_step.last())

def create_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='cueball')
    parser.add_argument('--task', type=str, default='sink_single')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--file', type=str,
                        default='cueball_random_dataset.npz')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=72)
    parser.add_argument('--viewpoint', type=str, default='birdseye')
    args = parser.parse_args()

    # Find camera index.
    assert args.viewpoint in VIEWPOINT_MAPPING.keys(), \
        f"--viewpoint should be in {VIEWPOINT_MAPPING.keys()}, but found " \
        f"unknown viewpoint '{args.viewpoint}'."

    # Initialize dataset.
    dataset = {k: [] for k in DATASET_KEYS}
    episode = {k: [] for k in DATASET_KEYS}

    # Load and reset environment.
    env = suite.load(domain_name=args.domain, task_name=args.task)
    env = pixels.Wrapper(
        env, pixels_only=True, render_kwargs=dict(
            height=args.height, width=args.width, camera_id=VIEWPOINT_MAPPING[
                args.viewpoint]))
    action_spec = env.action_spec()
    time_step = env.reset()
    append_to_episode(episode, time_step)

    for episode_count in tqdm(range(args.num_episodes)):
        while True:
            # Sample random action and step the environment.
            action = np.random.uniform(action_spec.minimum,
                                       action_spec.maximum,
                                       size=action_spec.shape)
            time_step = env.step(action)
            append_to_episode(episode, time_step)

            if time_step.last():
                for k in DATASET_KEYS:
                    dataset[k].append(np.array(episode[k]))
                time_step = env.reset()
                episode = {k: [] for k in DATASET_KEYS}
                append_to_episode(episode, time_step)
                episode_count += 1
                break

    for k in DATASET_KEYS:
        dataset[k] = np.stack(dataset[k])

    np.savez_compressed(args.file, **dataset)

    dataset_reloaded = np.load(args.file)

    for k in dataset_reloaded.keys():
        print(f"dataset_reloaded[{k}].shape:", dataset_reloaded[k].shape)

        import matplotlib.pyplot as plt
        for i in range(dataset['image'].shape[1]):
            plt.imshow(dataset['image'][0, i])
            plt.show()


if __name__ == '__main__':
    create_dataset()
