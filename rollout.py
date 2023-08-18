import tqdm
import os
import gym
from PIL import Image
import numpy as np
import numpy.random as nr
import numpy as np
from PIL import Image
import imageio

class CarRacing_rollouts():
    def __init__(self, seed_num=0):
        self.env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=False)
        self.env.reset(seed=seed_num)
        # self.file_dir = './data/CarRacing/'
        self.file_dir = './content/data/CarRacing/'

    def get_rollouts(self, num_rollouts=10000, reflesh_rate=5, max_episode=300):
        start_idx = 0
        if os.path.exists(self.file_dir):
            start_idx = len(os.listdir(self.file_dir)) 
        for i in tqdm.tqdm(range(start_idx, num_rollouts+1)):
            state_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []
            state = self.env.reset()
            done = False
            iter = 0
            # while (not done) and iter < max_episode:
            while iter < max_episode:
                if iter < 20:
                    action = np.array([-0.1, 1, 0])
                elif iter % reflesh_rate == 0:
                    steering, acceleration, brake = self.env.action_space.sample()
                    action = np.array([steering, acceleration, brake])
                    rn = nr.randint(0, 9)
                    if rn in [0]:
                        action = np.array([0, 0, 0])
                    elif rn in [1, 2, 3, 4]:
                        action = np.array([0, nr.uniform(0.0, 1.0), 0])
                    elif rn in [5, 6]:
                        action = np.array([nr.uniform(-1.0, 0.0), 0, 0])
                    elif rn in [7, 8]:
                        action = np.array([nr.uniform(0.0, 1.0), 0, 0])
                    elif rn in [9]:
                        action = np.array([0, 0, nr.uniform(0.0, 1.0)])
                    else:
                        pass

                state, reward, done, _, _ = self.env.step(action)
                state = self.reshape_state(state)
                state_sequence.append(state)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)
                iter += 1
            np.savez_compressed(os.path.join(self.file_dir, 'rollout_{}.npz'.format(i)), state=state_sequence, action=action_sequence, reward=reward_sequence, done=done_sequence)
            # np.savez(os.path.join(self.file_dir, 'rollout_{}.npz'.format(i)), state=state_sequence, action=action_sequence, reward=reward_sequence, done=done_sequence)

    def load_rollout(self, idx_rolloout):
        data = np.load(os.path.join(self.file_dir, 'rollout_{}.npz'.format(idx_rolloout)))
        return data['state'], data['action'], data['reward'], data['done']
    
    def load_rollouts(self, idx_rolloouts):
        states = []
        actions = []
        rewards = []
        dones = []
        for idx_rolloout in idx_rolloouts:
            data = np.load(os.path.join(self.file_dir, 'rollout_{}.npz'.format(idx_rolloout)))
            states.append(data['state'])
            actions.append(data['action'])
            rewards.append(data['reward'])
            dones.append(data['done'])
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        return states, actions, rewards, dones

    def reshape_state(self, state):
        # state（画像）をリサイズして64x64にする．値を0-1に正規化する処理は行っていない．
        HEIGHT = 64
        WIDTH = 64
        state = state[0:84, :, :]
        state = Image.fromarray(state).resize((HEIGHT, WIDTH))
        return state
    
    def make_gif(self, idx_rolloout):
        state, _, _, _ = self.load_rollout(idx_rolloout)
        images = []
        for i in range(len(state)):
            pil_image = Image.fromarray(state[i].astype("uint8"))
            images.append(pil_image)
        imageio.mimsave('./rollout.gif'.format(idx_rolloout), images, duration=10)
        