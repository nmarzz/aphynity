import torch
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import data_utils
from models.models import PendulumModel

# Get environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')
# Define integration parameters
t0 = 0.
te = 20.
t = torch.linspace(t0, te, 40).to(device)
model = PendulumModel(frictionless = False,include_data = True).to(device)
train,val,test = data_utils.get_pendulum_datasets(n=25)
train,val,test = train.to(device),val.to(device),test.to(device)
init_state = train[:,0,:]


# Define accessories
optimizer = optim.Adam(model.parameters(), lr=0.002)
loss_function = torch.nn.MSELoss(reduction = 'sum')

sol = odeint_adjoint(model,init_state , t, atol=1e-4, rtol=1e-4,method='dopri5')
pos = sol[:,:,0].transpose(0,1)
vel = sol[:,:,1].transpose(0,1)
# plt.plot(pos[0,:].detach())
# plt.plot(train[0,:,0])
# plt.show()


# Training loop
# TODO: Add testing loop
for i in range(50):
    optimizer.zero_grad()
    sol = odeint_adjoint(model,init_state , t, atol=1e-2, rtol=1e-2,method='dopri5')
    pos = sol[:,:,0].transpose(0,1)
    vel = sol[:,:,1].transpose(0,1)
    loss = torch.sum(torch.linalg.norm(model.data_driven(train),dim=2)) + (loss_function(pos,train[:,:,0]) + loss_function(vel,train[:,:,1]))
    print('Backpropping')
    loss.backward()
    optimizer.step()

    # Track physical model parameters
    print('*' * 20)
    print(f'Data contribution = {torch.sum(torch.linalg.norm(model.data_driven(train),dim=2))}')
    print(f'loss contribution =  {(loss_function(pos,train[:,:,0]) + loss_function(vel,train[:,:,1]))}')
    print(loss)
    for name, param in model.named_parameters():
        if param.requires_grad and name in ['omega','alpha']:
            print(name, param.data)

sol = odeint_adjoint(model,init_state , t, atol=1e-4, rtol=1e-4,method='dopri5')
pos = sol[:,:,0].transpose(0,1)
vel = sol[:,:,1].transpose(0,1)
plt.plot(pos[0,:].detach())
plt.plot(train[0,:,0])
plt.show()
