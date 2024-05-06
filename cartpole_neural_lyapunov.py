'''
See gymnasium cartpole environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
'''

import gymnasium as gym
import numpy as np
import torch
from cartpole_lqr import LQR
from model import NeuralLyapunovController
# from train_model import Trainer
import utils

class NeuralLyaponovControl():
    '''
    Neural Lyaponov Control For CartPole Environment
    '''
    def __init__(self, env, controller):
        # Gymnasium environment
        self.env = env
        # x has 4 dimensions, x, xdot, theta, thetadot
        self.controller = controller

    def get_input(self, x):
        self.controller.eval()
        x_tensor = torch.Tensor(x)
        V, u_tensor = self.controller(x_tensor)
        u_numpy = u_tensor.cpu().detach().numpy()
        # set min and max bounds
        u = np.clip(u_numpy, -10, 10)
        # input is scalar
        return u[0]

    def control(self, noisy_observer=False):
        observation, info = self.env.reset()
        for _ in range(1000):
            # get action and input (force) required 
            u = self.get_input(observation)
            action = 0
                    # returns action and input force
            if u > 0:
                # move cart right
                action = 1
            else:
                # move cart left
                action = 0

            # set magnitude of force as input
            env.unwrapped.force_mag = abs(u)

            observation, reward, terminated, truncated, info = self.env.step(action)

            # add small noise to observer (position velocity, pole angle, pole angular velocity)
            if noisy_observer:
                observation += np.random.normal(0, 0.15, 4)

            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()

def load_model():
    lqr = LQR()
    K = lqr.K
    lqr_val = -torch.Tensor(K)
    d_in, n_hidden, d_out = 4, 6, 1
    controller = NeuralLyapunovController(d_in, n_hidden, d_out, lqr_val)
    # TODO: load weights of trained model
    return controller

if __name__ == '__main__':
    show_gui = True
    env = gym.make('CartPole-v1', render_mode="human")


    controller = load_model()

    nlc = NeuralLyaponovControl(env, controller)
    if show_gui:
        nlc.control(noisy_observer=False)


    # TODO Plot results
    # utils.plot_roa(V_lqr, f)


