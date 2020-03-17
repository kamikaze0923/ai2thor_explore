"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/test.py

Contains the testing loop of the shared model within A3C (no optimisation/backprop needed)
Usually this is run concurrently while training occurs and is useful for tracking progress. But to
save resources we can choose to only test every args.test_sleep_time seconds.
"""

import time
from collections import deque

import torch
import torch.nn.functional as F

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from gym_ai2thor.utils import CSVLogger
from algorithms.a3c.model import ActorCritic

import os
from copy import deepcopy

def test(rank, args, shared_model, counter):
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

    model.eval()

    state = env.reset()
    reward_sum = 0
    done = True

    save = '{}-steps{}-process{}-lr{}-entropy_coef{}'.format("point" if args.point_cloud_model else "conv",
        args.num_steps, args.num_processes, args.lr, args.entropy_coef)
    save = os.path.join('logs', save)
    os.makedirs(save, exist_ok=True)

    if args.model:
        shared_model.load_state_dict(torch.load(os.path.join(save, "solved_ai2thor.pth")))
    else:
        logger = CSVLogger(os.path.join(save, 'test.csv'))
        fileds = ['episode_reward', 'frames_rendered']
        logger.log(fileds)

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    # actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(deepcopy(shared_model.state_dict()))
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            if args.cuda:
                if args.point_cloud_model:
                    state = (state[0].cuda(), state[1].cuda())
                else:
                    state = state.cuda()
                cx = cx.cuda()
                hx = hx.cuda()
            value, logit, (hx, cx) = model((state, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        # log_prob = F.log_softmax(logit, dim=-1)
        # print(prob)
        # entropy = -(log_prob * prob).sum(1, keepdim=True)
        # print(prob.max(1, keepdim=True)[0].cpu().numpy())
        # print(entropy)

        action = prob.max(1, keepdim=True)[1].cpu().numpy()
        state, reward, done, _ = env.step(action[0, 0], verbose=False)
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        # i.e. in test mode an agent can repeat an action ad infinitum
        # actions.append(action[0, 0])
        # if actions.count(actions[0]) == actions.maxlen:
        #     print('In test. Episode over because agent repeated action {} times'.format(
        #                                                                         actions.maxlen))
        #     done = True

        if done:
            print("Time {}, num steps over all threads {}, FPS {:.0f}, episode reward {: .2f}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            if not args.model:
                logger.log(["{: .2f}".format(reward_sum), counter.value])

            if reward_sum >= args.solved_reward:
                print("Solved Testing with Reward {}".format(reward_sum))
                torch.save(model.state_dict(), os.path.join(save, "solved_ai2thor.pth"))
                env.close()
                logger.close()
                break

            reward_sum = 0
            episode_length = 0
            # actions.clear()
            state = env.reset()
            time.sleep(args.test_sleep_time)

