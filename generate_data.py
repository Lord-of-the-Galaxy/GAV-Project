import numpy as np
import torch

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN

from tqdm import trange

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

def main(name, n_steps, n_envs, seed):
    custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    env = make_atari_env(name, n_envs=n_envs, seed=seed, env_kwargs={'render_mode':'rgb_array'})
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load('data/{}'.format(name), env=env, custom_objects=custom_objects, buffer_size=1, optimize_memory_usage=False)

    data = [dict(observations=[], actions=[], activations=[], qvalues=[], images=[]) for i in range(n_envs)]

    obs = env.reset()
    for i in trange(n_steps):
        actions, activations, qvalues = predict(model, obs)
        obs, rewards, dones, info = env.step(actions)
        for j in range(n_envs):
            data[j]['observations'].append(obs[j])
            data[j]['actions'].append(actions[j])
            data[j]['activations'].append(activations[j].cpu())
            data[j]['qvalues'].append(qvalues[j].cpu())
            data[j]['images'].append(info[j]['rgb'])
    
    print("Compiling data...")
    save = {}
    for j in range(n_envs):
        save['observations_{}'.format(j)] = np.stack(data[j]['observations'])
        save['actions_{}'.format(j)] = np.stack(data[j]['actions'])
        save['activations_{}'.format(j)] = np.stack(data[j]['activations'])
        save['qvalues_{}'.format(j)] = np.stack(data[j]['qvalues'])
        save['images_{}'.format(j)] = np.stack(data[j]['images'])
    
    print("Saving data...")
    np.savez_compressed('data/{}_{}.npz'.format(name, seed), **save)


if __name__ == '__main__':
    main('BreakoutNoFrameskip-v4', 2000, 1, 1)