'''
See gymnasium cartpole environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
'''

import gymnasium as gym
import numpy as np
from scipy import linalg
import utils

class LQR():
    '''
    LQR For CartPole Environment
    '''
    def __init__(self, env=None):
        # Gymnasium environment
        self.env = env

        # mass of cart
        self.M = 1. 
        # mass of pole/weight on end
        self.m = 0.1
        m_total = self.m + self.M
        # gravity
        self.g = 9.81
        # cart dampening
        self.delta = 0.

        # length of pole
        self.l = 1.5


        # State space model found Steve Brunton's book page 353
        b = -1 # configuration for upward control
        # https://databookuw.com/databook.pdf
        self.A = np.array([[0, 1., 0, 0],
                        [0, -self.delta/self.M, b*self.m*self.g / self.m, 0],
                        [0, 0, 0, 1],
                        [0, -b*self.delta / (self.M * self.l), -b * m_total*self.g / (self.M * self.l), 0]])

        # input matrix
        self.B = np.array([[0], [1/self.M], [0], [b/(self.M * self.l)]])

        # penalties for lqr
        self.R = 0.1 * np.eye(1)
        self.Q = np.diag([1., 1., 10., 1.])
        self.K = LQR.lqr(self.A, self.B, self.Q, self.R)

    def f(self, x):
        '''
        gets x_dot (dx/dt) for dynamic system 
        xdot = Ax + BU
        U --Kx
        '''
        s = self.A - self.B @ self.K
        xdot = s @ x
        return xdot
    
    def get_system(self):
        return self.A, self.B, self.Q, self.R, self.K

    def lyapunov_function(self, x):
        # algebraic ricatti equation
        P=linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        return x.T @ P @ x

    @staticmethod
    def lqr(A,B,Q,R):
        '''
        Method taked from my undergrad class (AE 353 at UIUC with Tim Bretl)
        https://tbretl.github.io/ae353-sp22/reference#lqr
        '''
        # algebraic ricatti equation
        P=linalg.solve_continuous_are(A,B,Q,R)
        # side note: v = xTPx is the lyapunov function
        K=linalg.inv(R) @ B.T @ P
        return K

    @staticmethod
    def get_input(K, x):
        # u = -Kx
        u = -np.dot(K, x)

        # set min and max bounds
        u = np.clip(u, -10, 10)

        # input is scalar
        return u[0]

    def control(self, noisy_observer=False):
        observation, info = self.env.reset()
        for _ in range(1000):
            # get action and input (force) required 
            u = LQR.get_input(self.K, observation)
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

def checkStability(A, B, K):
    '''
    Prove closed loop stability
    '''
    s = A - B@K
    eigenvalues, eigenvectors = np.linalg.eig(s)
    assert(np.all(eigenvalues.real < 0))

if __name__ == '__main__':
    show_gui = True
    env = gym.make('CartPole-v1', render_mode="human")
    lqr = LQR(env)
    A, B, Q, R, K = lqr.get_system()

    checkStability(A, B, K)

    if show_gui:
        lqr.control(noisy_observer=False)

    # plotting
    # algebraic ricatti equation gives p and V = x^TPx is lyapunov function
    V_lqr = lqr.lyapunov_function

    f = lqr.f
    # TODO Plot results
    # utils.plot_roa(V_lqr, f)


