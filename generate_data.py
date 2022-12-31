import numpy as np
import torch

from time import time
from pathlib import Path
import argparse

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN

from tqdm.rich import trange

# GAMES = ['BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'QbertNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'AsteroidsNoFrameskip-v4', 'BeamRiderNoFrameskip-v4']
GAMES = ['Breakout', 'Pong', 'SpaceInvaders', 'Seaquest',
         'Qbert', 'Enduro', 'MsPacman', 'Asteroids', 'BeamRider']

# this is definitely slow as it does two pass throughs


def predict(model, observation):
    # Get the activations
    model.policy.set_training_mode(False)
    with torch.no_grad():
        obs, vectorized_env = model.policy.obs_to_tensor(observation)
        activations = model.q_net.extract_features(obs)
        qvalues = model.q_net.q_net(activations)
    # Get the action
    actions, _ = model.predict(observation, deterministic=False)
    return actions, activations, qvalues


def generate_data(name, n_steps, seed, save_observations=False, save_actions=False, save_activations=True, save_qvalues=True, save_rewards=True, save_images=True):
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    env = make_atari_env(name + 'NoFrameskip-v4', n_envs=1,
                         seed=seed, env_kwargs={'render_mode': 'rgb_array'})
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load('models/dqn-' + name + 'NoFrameskip-v4', env=env,
                     custom_objects=custom_objects, buffer_size=1, optimize_memory_usage=False)

    data = dict(observations=[], actions=[], activations=[],
                qvalues=[], rewards=[], images=[])

    obs = env.reset()
    for i in trange(n_steps):
        actions, activations, qvalues = predict(model, obs)
        obs, rewards, dones, info = env.step(actions)
        if save_observations:
            data['observations'].append(obs[j])
        if save_actions:
            data['actions'].append(actions[0])
        if save_activations:
            data['activations'].append(activations[0].cpu())
        if save_qvalues:
            data['qvalues'].append(qvalues[0].cpu())
        if save_rewards:
            data['rewards'].append(rewards[0])
        if save_images:
            data['images'].append(info[0]['rgb'])

    print("Saving data...")
    Path(f"data/{name}").mkdir(exist_ok=True)

    if save_observations:
        np.save('data/{}/observations_{}.npy'.format(name, seed),
                np.stack(data['observations']))
    if save_actions:
        np.save('data/{}/actions_{}.npy'.format(name, seed),
                np.stack(data['actions']))
    if save_activations:
        np.save('data/{}/activations_{}.npy'.format(name, seed),
                np.stack(data['activations']))
    if save_qvalues:
        np.save('data/{}/qvalues_{}.npy'.format(name, seed),
                np.stack(data['qvalues']))
    if save_rewards:
        np.save('data/{}/rewards_{}.npy'.format(name, seed),
                np.stack(data['rewards']))
    if save_images:
        np.save('data/{}/images_{}.npy'.format(name, seed),
                np.stack(data['images']))


def view_game(name, n_steps, seed):
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    env = make_atari_env(name + 'NoFrameskip-v4', n_envs=1,
                         seed=seed, env_kwargs={'render_mode': 'human'})
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load('models/dqn-' + name + 'NoFrameskip-v4', env=env,
                     custom_objects=custom_objects, buffer_size=1, optimize_memory_usage=False)

    obs = env.reset()
    for i in trange(n_steps):
        actions, activations, qvalues = predict(model, obs)
        obs, rewards, dones, info = env.step(actions)


def process(games, n_steps, view, seed, **kwargs):
    if view:
        view_game(games[0], n_steps, seed)
    else:
        for game in games:
            generate_data(game, n_steps, seed, **kwargs)


def add_complementary_options(parser, option, short_option=None, default=True, dest=None, help=None):
    group = parser.add_mutually_exclusive_group(required=False)

    if dest is None:
        dest = option

    pa = ['--' + option]
    if short_option is not None:
        pa.insert(0, '-' + short_option)
    ka = {'dest': dest, 'help': help, 'action': 'store_const',
          'const': True, 'default': default}

    group.add_argument(*pa, **ka)

    pa = ['--no-' + option]
    if short_option is not None:
        pa.insert(0, '-n' + short_option)
    ka = {'dest': dest, 'action': 'store_const',
          'const': False, 'default': default}

    group.add_argument(*pa, **ka)


def main():
    root_parser = argparse.ArgumentParser(
        description='Generate data by running the models on the Atari games')

    root_parser.add_argument('games', type=str, nargs='+', metavar='GAME',
                             help="List of the names of the games to run: %(choices)s", choices=GAMES)
    root_parser.add_argument('--n-steps', '-num', dest='n_steps', metavar='N', type=int,
                             default=10000, help='The number of steps to run the game for (default: 10000)')

    root_parser.add_argument('-v', '--view', action='store_true',
                             help='View the agent play the first game in the list instead of saving data')
    root_parser.add_argument('-s', '--seed', type=int, default=int(time()),
                             help='The random seed to use for generating the data')

    add_complementary_options(root_parser, 'observations', 'o', False,
                              'save_observations', 'Save the observations (default: False)')
    add_complementary_options(root_parser, 'actions', 'a', False,
                              'save_actions', 'Save the actions (default: False)')
    add_complementary_options(root_parser, 'activations', 'c', True,
                              'save_activations', 'Save the activations (default: True)')
    add_complementary_options(root_parser, 'qvalues', 'q', True,
                              'save_qvalues', 'Save the q-values (default: True)')
    add_complementary_options(root_parser, 'rewards', 'r',
                              True, 'save_rewards', 'Save the rewards (default: True)')
    add_complementary_options(root_parser, 'images', 'i',
                              True, 'save_images', 'Save the images (default: True)')

    args = root_parser.parse_args()

    process(**vars(args))


if __name__ == '__main__':
    main()
