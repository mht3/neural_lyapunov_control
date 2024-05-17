import torch
import numpy as np
import gymnasium as gym
from cartpole_lqr import LQR

class CartpoleDatasetBuilder():
    def __init__(self, state_min, state_max, trajectory_length, N, inverse_noise_factor=3.):
        self.state_min = state_min
        self.state_max = state_max

        # lqr policy
        lqr = LQR()
        K = lqr.K
        self.lqr_val = -K

        self.env = gym.make('CartPole-v1')

        # scale standard deviation based on min/max values
        if inverse_noise_factor == 0:
            self.scaled_sigma = 0
        else:
            self.scaled_sigma = (np.array(state_max) - np.array(state_min)) / (2*inverse_noise_factor)
        print(self.scaled_sigma)
        self.trajectory_length = trajectory_length
        self.N = N

    def build(self):

        # TODO make this function return numpy array instead
        X = self.load_state(self.state_min, self.state_max).numpy()

        # get 3 (s, pi(s), s') pairs
        # trajectory_size = self.trajectory_length * len(self.sta)
        trajectories = []
        for i in range(self.N):
            # length 4 vector for state
            x_i = X[i, :]
            # list of tuples of (s, pi(s), s') pairs
            trajectory = self.get_trajectory(x_i)
            trajectories.append(trajectory)

        return trajectories
    
    def get_action(self, x_i):
        '''
        Take in single state x_i -> plicy -> get action
        '''
        return self.lqr_val @ x_i

    def get_next_state(self, x_i, pi_i):
        '''
        Generates all X_primes needed given current state and current action
        X: current position, velocity, pole angle, and pole angular velocity
        u: input for cartpole 
        '''
        # take step in environment based upon current state and action
        u_i = np.clip(pi_i, -10, 10)[0]
        # set environment as x_i
        observation, info = self.env.reset()
        self.env.unwrapped.state = x_i

        action = 0
        if u_i > 0:
            # move cart right
            action = 1
        else:
            # move cart left
            action = 0

        # set magnitude of force as input
        self.env.unwrapped.force_mag = abs(u_i)
        # take step in environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        return observation

    def get_trajectory(self, x_i):
        
        trajectory = []
        # get action
        pi_s = self.get_action(x_i)
        # get next state
        x_prime = self.get_next_state(x_i, pi_s)
        trajectory.append((x_i, pi_s, x_prime))

        observation, info = self.env.reset()
        for j in range(self.trajectory_length - 1):
            # perturb state and use as next trajectory
            x = x_prime + np.random.normal(0, self.scaled_sigma, len(x_prime))
            # get action
            pi_s = self.get_action(x)
            # get next state
            x_prime = self.get_next_state(x, pi_s)

            # add to trajectories
            trajectory.append((x, pi_s, x_prime))
            

        return trajectory

    def load_state(self, target_mins, target_maxs):
        # X: Nx3 tensor of initial states
        X = torch.empty(self.N, 0)
        for i in range(len(target_mins)):
            t_min = target_mins[i]
            t_max = target_maxs[i]
            x = torch.Tensor(self.N, 1).uniform_(t_min, t_max)
            X = torch.cat([X, x], dim=1)

        return X

    def save(self, data, filename='trajectories.npz'):
        # TODO: Save dataset/compress
        # Could store each trajectory in flattened list each s, a, s' pair is separated by 11 entries
        pass

    def load(self, filename):
        # TODO load compressed dataset in original form
        pass

    def convert_data(data):
        # TODO flatten trajectories to s, a, s' 
        pass

if __name__ == '__main__':

    # bounds for position, velocity, angle, and angular velocity
    state_min = [-2.4, -2.0, -0.2094, -2]
    state_max = [2.4, 2.0, 0.2094, 2]

    dataset = CartpoleDatasetBuilder(state_min, state_max, trajectory_length=2,
                                     N=5, inverse_noise_factor=3.)
    trajectories = dataset.build()

    # dataset.save(trajectories, filename='trajectories')