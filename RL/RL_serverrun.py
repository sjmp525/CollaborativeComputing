import pickle
import torch
import tqdm
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys

sys.path.append('../')
from RLEnv import Env
from adapt_multiplehead_attention_with_nystrom import AdaptiveMultiHeadAttentionWithNystrom


import config

import PPO

if config.random:
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    logger.info('Random seed: {}'.format(config.random_seed))

# Creating environment
env = Env(0, config.SERVER_ADDR, config.SERVER_PORT, config.CLIENTS_LIST, config.model_name, config.model_cfg,
          config.rl_b)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Creating PPO agent
state_dim = env.state_dim
action_dim = env.action_dim
memory = PPO.Memory()
ppo = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs,
              config.eps_clip)


state_derivative = AdaptiveMultiHeadAttentionWithNystrom(
    input_dim=state_dim,  # assuming state_dim is defined elsewhere
    base_num_heads=4,  # you can adjust the number of heads
    delta=0.1,  # initial delta value
    max_k=3,  # maximum number of derived states
    dropout_prob=0.1  # dropout probability for regularization
)
# RL training
logger.info('==> RL Training Start.')
time_step = 0
update_epoch = 1

res = {'rewards': [], 'maxtime': [], 'state': [], 'actions': [], 'split_layers': [], 'std': [], 'loss': []}

for i_episode in tqdm.tqdm(range(1, config.max_episodes + 1)):
    done = False  # Flag controling finish of one episode
    if i_episode == 1:  # We run two times of initial state to get stable training time
        first = True
        state = env.reset(done, first)
    else:
        first = False
        state = env.reset(done, first)

    # Generate the first derived state
    derived_states = state_derivative.generate_derivative_states(state, [state])

    for t in range(config.max_timesteps):
        logger.info('====================================>')
        time_step += 1

        if derived_states:
            state = derived_states.pop(0)
        action, action_mean, std = ppo.select_action(state, memory)
        state, reward, maxtime, done, split_layers = env.step(action, done)
        # Generate next derived state
        derived_states = state_derivative.generate_derivative_states(state, [state])

        logger.info('Current reward: ' + str(reward))
        logger.info('Current maxtime: ' + str(maxtime))

        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # Update
        avg_loss = 0.0
        if time_step % config.update_timestep == 0:
            avg_loss = ppo.update(memory)  # 获取平均损失
            logger.info('Agent has been updated: ' + str(update_epoch))
            if update_epoch > config.exploration_times:
                ppo.explore_decay(update_epoch - config.exploration_times)

            memory.clear_memory()
            time_step = 0
            update_epoch += 1

            # Record the results for each update epoch
            with open('MobileNetV3_Linear_Kmeans++.pkl', 'wb') as f:
                pickle.dump(res, f)

            # save the agent every updates
            torch.save(ppo.policy.state_dict(), './MobileNetV3_Linear_Kmeans++.pth')

        res['rewards'].append(reward)
        res['maxtime'].append(maxtime)
        res['actions'].append((action, action_mean))
        res['std'].append(std)
        res['state'].append(state)
        res['split_layers'].append(split_layers)
        res['loss'].append(avg_loss)

        if done:
            break

        # stop when get control update epoch
        if update_epoch > 50:
            break
