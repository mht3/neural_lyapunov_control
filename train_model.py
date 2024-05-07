
import torch
from cartpole_lqr import LQR
from model import NeuralLyapunovController
from loss import LyapunovRisk
from utils import CheckLyapunov, AddCounterexamples
import gymnasium as gym 

class Trainer():
    def __init__(self, model, lr, optimizer, loss_fn):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.lyapunov_loss = loss_fn
    
    def get_lie_derivative(self, X, V_candidate, f):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ
        '''
        w1 = self.model.layer1.weight
        b1 = self.model.layer1.bias
        w2 = self.model.layer2.weight
        b2 = self.model.layer2.bias
        # running through model again 
        z1 = X @ w1.t() + b1
        a1 = torch.tanh(z1)
        z2 = a1 @ w2.t() + b2
        d_z2 = 1. - V_candidate**2
        partial_z2_a1 = w2
        partial_a1_z1 = 1 - torch.tanh(z1)**2
        partial_z1_x = w1

        d_a1 = (d_z2 @ partial_z2_a1)
        d_z1 = d_a1 * partial_a1_z1

        # gets final ∂V/∂x
        d_x = d_z1 @ partial_z1_x

        lie_derivative = torch.diagonal((d_x @ f.t()), 0)
        return lie_derivative

    def train(self, X, x_0, epochs=2000, verbose=False, every_n_epochs=10, check_approx=False):
        model.train()
        valid = False
        loss_list = []

        if check_approx == True:
            env = gym.make('CartPole-v1')

        for epoch in range(1, epochs+1):
            if valid == True:
                if verbose:
                    print('Found valid solution.')
                break

            # zero gradients
            optimizer.zero_grad()

            # get lyapunov function and input from model
            V_candidate, u = self.model(X)
            # get lyapunov function evaluated at equilibrium point
            V_X0, u_X0 = self.model(x_0)
            # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
            f = f_value(X, u)
            L_V = self.get_lie_derivative(X, V_candidate, f)
            # get loss
            loss = self.lyapunov_loss(V_candidate, L_V, V_X0)

            # compute approximate f_dot and compare to true f
            if check_approx == True:
                X_prime = step(X, u, env)
                f_approx = approx_f_value(X, X_prime, dt=0.02)

                # check dx/dt estimates are close
                # epsilon for x_dot. cart velocity and angular velocity are easier to approximate than accelerations.
                # TODO is there a better way to approximate without running throught the simulator multiple times?
                epsilon = torch.tensor([1e-4, 10., 1e-4, 10.])

                assert(torch.all(abs(f - f_approx) < epsilon))

                # could replace loss function 
                L_V_approx = self.get_lie_derivative(X, V_candidate, f_approx)
                    
            

            loss_list.append(loss.item)
            loss.backward()
            self.optimizer.step() 
            if verbose and (epoch % every_n_epochs == 0):
                print('Epoch:\t{}\tLyapunov Risk: {:.4f}'.format(epoch, loss.item()))

            # TODO Add in falsifier here
            # add counterexamples

        return loss_list
    
def load_model():
    lqr = LQR()
    K = lqr.K
    lqr_val = -torch.Tensor(K)
    d_in, n_hidden, d_out = 4, 6, 1
    controller = NeuralLyapunovController(d_in, n_hidden, d_out, lqr_val)
    return controller

def step(X, u, env):
    '''
    Generates all X_primes needed given current state and current action
    X: current position, velocity, pole angle, and pole angular velocity
    u: input for cartpole 
    '''
    # take step in environment based upon current state and action
    N = X.shape[0]
    u = torch.clip(u, -10, 10)

    X_prime = torch.empty_like(X)
    observation, info = env.reset()
    for i in range(N):
        x_i = X[i, :].detach().numpy()
        # set environment as x_i
        observation, info = env.reset()
        env.unwrapped.state = x_i

        # get current action to take 
        u_i = u[i][0].detach().numpy()
        action = 0
        if u_i > 0:
            # move cart right
            action = 1
        else:
            # move cart left
            action = 0

        # set magnitude of force as input
        env.unwrapped.force_mag = abs(u_i)
        # take step in environment
        # TODO: Error when taking step 
        observation, reward, terminated, truncated, info = env.step(action)
        # add sample to X_prime
        X_prime[i, :] = torch.tensor(observation)

    return X_prime

def approx_f_value(X, X_prime, dt=0.02):
    # Approximate f value with S, a, S'
    y = (X_prime - X) / dt
    return y

def f_value(X, u):
    y = []
    N = X.shape[0]
    # Get system dynamics for cartpole 
    lqr = LQR()
    A, B, Q, R, K = lqr.get_system()
    u = torch.clip(u, -10, 10)
    for i in range(0, N): 
        x_i = X[i, :].detach().numpy()

        u_i = u[i].detach().numpy()
        # xdot = Ax + Bu
        f = A@x_i + B@u_i
        
        y.append(f.tolist()) 

    y = torch.tensor(y)
    return y
        

if __name__ == '__main__':
    ### load model and training pipeline with initialized LQR weights ###
    model = load_model()
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1.)
    trainer = Trainer(model, lr, optimizer, loss_fn)

    ### Generate random training data ###
    # number of samples
    N = 500
    # bounds for position, velocity, angle, and angular velocity
    X_p = torch.Tensor(N, 1).uniform_(-2.4, 2.4)
    X_v = torch.Tensor(N, 1).uniform_(-2, 2)  
    # -12 to 12 degrees
    X_theta = torch.Tensor(N, 1).uniform_(-0.2094, 0.2094)  
    X_theta_dot = torch.Tensor(N, 1).uniform_(-2, 2)
    # concatenate all tensors as N x 4
    X = torch.cat([X_p, X_v, X_theta, X_theta_dot], dim=1)
    # stable conditions (used for V(x_0) = 0)
    # note that X_p is a free variable and can be at any position
    x_p_eq, x_v_eq, x_theta_eq, x_theta_dot_eq = 0., 0., 0., 0.
    X_0 = torch.Tensor([x_p_eq, x_v_eq, x_theta_eq, x_theta_dot_eq])

    ### Start training process ##
    approx = True # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    trainer.train(X, X_0, epochs=200, verbose=True, check_approx=approx)