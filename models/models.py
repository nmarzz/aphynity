import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt


class ODEfunc(nn.Module):
    '''
    Describes the ODE equation of a frictionless pendulum
    '''
    def __init__(self,frictionless):
        super(ODEfunc, self).__init__()

        self.omega = nn.Parameter(torch.rand(1))

        if frictionless:
            self.alpha = 0
        else:
            self.alpha = nn.Parameter(torch.rand([1]))

    def forward(self,t,x):
        pos , vel = x[0][0],x[0][1]
        dpos = vel
        dvel = (-self.omega ** 2) * torch.sin(pos) - self.alpha * vel


        return torch.tensor([[dpos,dvel]])


class PendulumModel(nn.Module):
    def __init__(self,frictionless = True,include_data = True):
        super(PendulumModel, self).__init__()

        self.include_data = include_data

        self.data_driven = nn.Sequential(nn.Linear(2,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,2)
        )

        self.physical = ODEfunc(frictionless)

    def forward(self,t,x):
        if self.include_data:            
            return self.physical(t,x) + self.data_driven(x)
        else:
            return self.physical(t,x)
