import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt



class PendulumModel(nn.Module):
    def __init__(self,frictionless = True,include_data = True):
        super(PendulumModel, self).__init__()

        self.include_data = include_data
        self.frictionless = frictionless
        if frictionless:
            self.alpha = 0
        else:
            self.alpha = nn.Parameter(torch.rand([1]))
        self.omega = nn.Parameter(torch.rand([1]))

        self.updatemat = torch.tensor([[0.,1.],[0.,- self.alpha]]).t()

        if self.include_data:
            self.data_driven = nn.Sequential(nn.Linear(2,200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,2)
            )

    def forward(self,t,x):

        physical = torch.matmul(x,self.updatemat)
        physical[:,1] += - (self.omega ** 2) * torch.sin(x[:,0])

        if self.include_data:
            return physical + self.data_driven(x)
        else:
            return physical






class PendulumModel2(nn.Module):
    def __init__(self):
        super(PendulumModel2, self,frictionless = True).__init__()
        self.frictionless = frictionless
        self.omega = nn.Parameter(torch.rand([1]))

        if not self.frictionless:
            self.alpha = nn.Parameter(torch.rand([1]))

    def forward(self,t,x):
        pos,vel = x
        dpos = vel

        if self.frictionless:
             dvel = -self.omega**2 * torch.sin(pos)
        else:
             dvel = -self.omega**2 * torch.sin(pos) + self.alpha * vel



        return dpos,dvel
