
import torch
from cartpole_lqr import LQR
from model import NeuralLyapunovController
from loss import LyapunovRisk
from utils import dtanh, CheckLyapunov, AddCounterexamples

class Trainer():
    def __init__(self, model, lr, optimizer, loss_fn, f_value):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.lyapunov_loss = loss_fn

        # f=Xdot: how inputs change as a function of time
        self.f_value = f_value
    
    def get_lie_derivative(self, X, V_candidate, f):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ
        Could alternatively use finite difference methods to do this if f is unknown.
        '''
        w1 = self.model.layer1.weight
        b1 = self.model.layer1.bias
        w2 = self.model.layer2.weight
        b2 = self.model.layer2.bias
        # running through model again 
        z1 = X @ w1.t() + b1
        a1 = torch.tanh(z1)
        z2 = a1 @ w2.t() + b2
        d_z2 = dtanh(z2) # originally dtanh(V_candidate) in Ya-Chien's code
        partial_z2_a1 = w2
        partial_a1_z1 = dtanh(z1)
        partial_z1_x = w1

        d_a1 = (d_z2 @ partial_z2_a1)
        d_z1 = d_a1 * partial_a1_z1

        # gets final ∂V/∂x
        d_x = d_z1 @ partial_z1_x

        lie_derivative = torch.diagonal((d_x @ f.t()), 0)
        return lie_derivative

    def train(self, X, x_0, epochs=2000, verbose=False, every_n_epochs=10):
        model.train()
        valid = False
        for epoch in range(1, epochs+1):
            if valid == True:
                if verbose:
                    print('Found valid solution.')
                break
            
            # get lyapunov function and input from model
            V_candidate, u = self.model(X)
            # get lyapunov function evaluated at equilibrium point
            V_X0, u_X0 = self.model(x_0)
            # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ

            f = self.f_value(X, u)
            L_V = self.get_lie_derivative(X, V_candidate, f)

            optimizer.zero_grad()

            loss = self.lyapunov_loss(V_candidate, L_V, V_X0)
                    
            loss.backward()
            self.optimizer.step() 
            if verbose and (epoch % every_n_epochs == 0):
                print('Epoch:\t{}\tLyapunov Risk: {:.4f}'.format(epoch, loss.item()))

    

def load_model():
    lqr = LQR()
    K = lqr.K
    lqr_val = -torch.Tensor(K)
    d_in, n_hidden, d_out = 4, 6, 1
    controller = NeuralLyapunovController(d_in, n_hidden, d_out, lqr_val)
    return controller

def f_value(X, u):
    #Dynamics
    y = []
    N = X.shape[0]
    for i in range(0, N): 
        x, x_dot, theta, theta_dot = X[i, :]

        # TODO: Get dynamics
        v = 0. # acceleration
        w_dot = 0. # angular acceleration

        f = [ x_dot, 
              v,
              theta_dot,
              w_dot]
        
        y.append(f) 

    y = torch.tensor(y)
    return y
        

if __name__ == '__main__':
    ### load model and training pipeline with initialized LQR weights ###
    model = load_model()
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1.)
    trainer = Trainer(model, lr, optimizer, loss_fn, f_value)

    ### Generate random training data ###
    # number of samples
    N = 500
    # bounds for position, velocity, angle, and angular velocity
    X_p = torch.Tensor(N, 1).uniform_(-4.8, 4.8)
    X_v = torch.Tensor(N, 1).uniform_(-6, 6)  
    # -24 to 24 degrees
    X_theta = torch.Tensor(N, 1).uniform_(-0.4189, 0.4189)  
    X_theta_dot = torch.Tensor(N, 1).uniform_(-6, 6)
    # concatenate all tensors as N x 4
    X = torch.cat([X_p, X_v, X_theta, X_theta_dot], dim=1)
    # stable conditions (used for V(x_0) = 0)
    # note that X_p is a free variable and can be at any position
    x_p_eq, x_v_eq, x_theta_eq, x_theta_dot_eq = 0., 0., 0., 0.
    X_0 = torch.Tensor([x_p_eq, x_v_eq, x_theta_eq, x_theta_dot_eq])

    ### Start training process ##
    trainer.train(X, X_0, epochs=200, verbose=True)