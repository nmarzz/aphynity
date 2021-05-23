import torch
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt


from models.models import PendulumModel


y0 = torch.tensor([[1., 0.]])
t = torch.linspace(0., 40., 100)
model = PendulumModel(frictionless = True,include_data = True)

true_y = odeint_adjoint(model, y0, t, method='dopri5')

pos = true_y[:,:,0].detach().numpy()
vel = true_y[:,:,1].detach().numpy()

plt.plot(t,pos)
plt.show()
