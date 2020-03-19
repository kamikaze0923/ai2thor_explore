"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

Contains the train code run by each A3C process on AI2ThorEnv.
For initialisation, we set up the environment, seeds, shared model and optimizer.
In the main training loop, we always ensure the weights of the current model are equal to the
shared model. Then the algorithm interacts with the environment args.num_steps at a time,
i.e it sends an action to the env for each state and stores predicted values, rewards, log probs
and entropies to be used for loss calculation and backpropagation.
After args.num_steps has passed, we calculate advantages, value losses and policy losses using
Generalized Advantage Estimation (GAE) with the entropy loss added onto policy loss to encourage
exploration. Once these losses have been calculated, we add them all together, backprop to find all
gradients and then optimise with Adam and we go back to the start of the main training loop.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import sys

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from algorithms.a3c.model import ActorCritic

# import matplotlib.pyplot as plt

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer):
    torch.manual_seed(args.seed + rank)
    env = AI2ThorEnv(config_dict=args.config_dict)
    env.seed(args.seed + rank)

    if args.point_cloud_model:
        model = ActorCritic(env.action_space.n)
    else:
        args.frame_dim = env.config['resolution'][-1]
        model = ActorCritic(env.action_space.n, env.observation_space.shape[0], args.frame_dim)

    if args.cuda:
        model = model.cuda()
    model.train()

    state = env.reset()
    done = True

    # monitoring
    total_reward_for_num_steps_list = []
    episode_total_rewards_list = []
    avg_reward_for_num_steps_list = []

    total_length = 0
    episode_length = 0
    n_episode = 0
    total_reward_for_episode = 0
    all_rewards_in_episode = []
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            total_length += 1
            if args.cuda:
                if args.point_cloud_model:
                    state = (state[0].cuda(), state[1].cuda())
                else:
                    state = state.cuda()
                cx = cx.cuda()
                hx = hx.cuda()
            value, logit, (hx, cx) = model((state, (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            log_probs.append(log_prob)

            action_int = action.cpu().numpy()[0][0].item()

            state, reward, done, _ = env.step(action_int, verbose=False)
            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
                total_length -= 1
                total_reward_for_episode = sum(all_rewards_in_episode)
                episode_total_rewards_list.append(total_reward_for_episode)
                all_rewards_in_episode = []
                state = env.reset()
                print('Process {} Episode {} Over with Length: {} and Reward: {: .2f}. Total Trained Length: {}'.format(
                    rank, n_episode, episode_length, total_reward_for_episode, total_length))
                sys.stdout.flush()
                episode_length = 0
                n_episode += 1

            values.append(value)
            rewards.append(reward)
            all_rewards_in_episode.append(reward)

            if done:
                break

        if args.synchronous:
            if total_reward_for_episode >= args.solved_reward:
                print("Process {} Solved with Reward {}".format(rank, total_reward_for_episode))
                env.close()
                break


        total_reward_for_num_steps = sum(rewards)
        total_reward_for_num_steps_list.append(total_reward_for_num_steps)
        avg_reward_for_num_steps = total_reward_for_num_steps / len(rewards)
        avg_reward_for_num_steps_list.append(avg_reward_for_num_steps)

        # Backprop and optimisation
        R = torch.zeros(1, 1)
        gae = torch.zeros(1, 1)
        if args.cuda:
            if args.point_cloud_model:
                state = (state[0].cuda(), state[1].cuda())
            else:
                state = state.cuda()
            R = R.cuda()
            gae = gae.cuda()
        if not done:  # to change last reward to predicted value to ....
            value, _, _ = model((state, (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        # import pdb;pdb.set_trace() # good place to breakpoint to see training cycle

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                          args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

