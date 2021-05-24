import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt

class PendulumModel(nn.Module):
    def __init__(self,frictionless = True,include_data = True):
        super(PendulumModel, self).__init__()
        self.frictionless = frictionless
        self.include_data = include_data
        self.omega = nn.Parameter(torch.rand([1]))


        if not self.frictionless:
            self.alpha = nn.Parameter(torch.rand([1]))

        if self.include_data:
            self.data_driven = nn.Sequential(nn.Linear(2,200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,2)
            )

    def forward(self,t,x):
        pos,vel = x
        dpos = vel

        if self.frictionless:
             dvel = -self.omega**2 * torch.sin(pos)
        else:
             dvel = -self.omega**2 * torch.sin(pos) - self.alpha * vel



        return dpos,dvel
