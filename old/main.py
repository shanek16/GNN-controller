import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter

import gym
import numpy as np
from gym import wrappers
import gym_flock

import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from old.param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="Flocking-v0",
                    help='name of the environment to run')
parser.add_argument('--algo', default='DDPG',
                    help='algorithm to use: DDPG | NAF')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=200, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

env = NormalizedActions(gym.make(args.env_name))
writer = SummaryWriter()

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.algo == "NAF":
    agent = NAF(args.gamma, args.tau, args.hidden_size,
                      env.observation_space.shape[0], env.action_space, device)
else:
    agent = DDPG(args.gamma, args.tau, args.hidden_size,
                      env.observation_space.shape[0], env.action_space, device)

memory = ReplayMemory(args.replay_size)

ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, 
    desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05, device=device) if args.param_noise else None

rewards = []
total_numsteps = 0
updates = 0

state = env.reset()
n_agents = state.shape[0]

action = np.zeros((n_agents, env.action_space.shape[0]))

for i_episode in range(args.num_episodes):

    state = env.reset() # TODO

    if args.ou_noise: 
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                      i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if args.param_noise and args.algo == "DDPG":
        agent.perturb_actor_parameters(param_noise)

    episode_reward = 0
    while True:

        for n in range(n_agents):
            arr_agent_state = state[n,:].flatten()
            agent_state = torch.Tensor([state[n,:].flatten()]).to(device)
            agent_action = agent.select_action(agent_state, ounoise, param_noise)
            action[n, :] = agent_action.cpu().numpy()

        (amax, next_state), reward, done, _ = env.step(action)
        total_numsteps += 1
        episode_reward += reward
        mask = torch.Tensor([not done])
        reward = torch.Tensor([reward])

        agent_action = torch.Tensor([action[amax,:].flatten()]).to(device)
        agent_next_state = torch.Tensor([next_state[amax,:].flatten()]).to(device)
        agent_state = torch.Tensor([state[amax,:].flatten()]).to(device)
        memory.push(agent_state, agent_action, mask, agent_next_state, reward)

        # for n in range(n_agents):
        #     agent_action = torch.Tensor([action[n,:].flatten()]).to(device)
        #     agent_next_state = torch.Tensor([next_state[n,:].flatten()]).to(device)
        #     agent_state = torch.Tensor([state[n,:].flatten()]).to(device)
        #     memory.push(agent_state, agent_action, mask, agent_next_state, reward)


        state = next_state

        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_parameters(batch)

                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)

                updates += 1
        if done:
            break

    writer.add_scalar('reward/train', episode_reward, i_episode)

    # Update param_noise based on distance metric
    if args.param_noise:
        episode_transitions = memory.memory[memory.position-t:memory.position]
        states = torch.cat([transition[0] for transition in episode_transitions], 0).to(device)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

        ddpg_dist = ddpg_distance_metric(perturbed_actions.cpu().numpy(), unperturbed_actions.cpu().numpy())
        param_noise.adapt(ddpg_dist)

    rewards.append(episode_reward)
    if i_episode % 10 == 0:
        state = env.reset() #torch.Tensor([env.reset()]) # TODO
        episode_reward = 0
        while True:
            for n in range(n_agents):
                agent_state = torch.Tensor([state[n,:].flatten()]).to(device)
                agent_action = agent.select_action(agent_state, ounoise, param_noise)
                action[n, :] = agent_action.cpu().numpy()
            #action = agent.select_action(state)

            (amax, next_state), reward, done, _ = env.step(action) # TODO
            episode_reward += reward

            state = next_state
            if done:
                break

        writer.add_scalar('reward/test', episode_reward, i_episode)

        rewards.append(episode_reward)
        print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))
    if i_episode % 50 == 0:       
        agent.save_model(args.env_name, suffix=str(i_episode))

env.close()