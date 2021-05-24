import torch
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import data_utils
from models.models import PendulumModel

# Get environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define integration parameters
t0 = 0.
te = 20.
t = torch.linspace(t0, te, 40).to(device)
model = PendulumModel(frictionless = False,include_data = True).to(device)
train,val,test = data_utils.get_pendulum_datasets(n=25)
train,val,test = train.to(device),val.to(device),test.to(device)
init_state = train[:,0,:]


# Define accessories
optimizer = optim.Adam(model.parameters(), lr=0.03)
loss_function = torch.nn.MSELoss(reduction = 'sum')


# Training loop
# TODO: Add testing loop
for i in range(50):
    optimizer.zero_grad()
    sol = odeint_adjoint(model,init_state , t, atol=1e-8, rtol=1e-8,method='dopri5')
    pos = sol[:,:,0].transpose(0,1)
    vel = sol[:,:,1].transpose(0,1)
    loss = loss_function(pos,train[:,:,0]) + loss_function(vel,train[:,:,1])
    loss.backward()
    optimizer.step()

    # Track physical model parameters
    print('*' * 20)
    print(loss)
    for name, param in model.named_parameters():
        if param.requires_grad and name in ['omega','alpha']:
            print(name, param.data)

# Plot the end results
pos,vel = odeint_adjoint(model,init_state , t, atol=1e-8, rtol=1e-8,method='dopri5')
plt.plot(t,pos[:,0].detach())
plt.plot(t,train[0,:,0])
plt.show()
