import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt

class PendulumModel(nn.Module):
    def __init__(self,frictionless = True,include_neural_net = True):
        super(PendulumModel, self).__init__()
        self.frictionless = frictionless
        self.include_neural_net = include_neural_net
        # self.omega = nn.Parameter(torch.rand(1))
        self.omega = nn.Parameter(torch.tensor(0.55))


        if not self.frictionless:
            self.alpha = nn.Parameter(torch.rand(1))

        if self.include_neural_net:
            self.neural_net = MLP()

    def forward(self,t,x):
        pos,vel = x[:,0],x[:,1]
        dpos = vel

        if self.frictionless:
             dvel = -self.omega**2 * torch.sin(pos)
        else:
             dvel = -self.omega**2 * torch.sin(pos) - self.alpha * vel

        y = torch.stack((dpos,dvel),dim = 1)

        if self.include_neural_net:
            y += self.neural_net(t,x)

        return y

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # self.neural_net = nn.Sequential(nn.Linear(2,200),
        # nn.ReLU(),
        # nn.Linear(200,200),
        # nn.ReLU(),
        # nn.Linear(200,2)
        # )
        self.neural_net = nn.Linear(2,2,bias = False)

    def forward(self,t,x):
        return self.neural_net(x)
