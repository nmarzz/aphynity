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
model = PendulumModel(frictionless = True,include_data = True).to(device)
batch_size = 40
train,val,test = data_utils.get_pendulum_datasets(n=batch_size)
train,val,test = train.to(device),val.to(device),test.to(device)
init_state = train[:,0,:]


# Define accessories
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss(reduction = 'mean')


# Training loop
# TODO: Add testing loop
lam = 1
for i in range(2000):
    optimizer.zero_grad()
    sol = odeint_adjoint(model,init_state , t, atol=1e-2, rtol=1e-2,method='dopri5').transpose(0,1)

    data_loss = torch.sum(torch.linalg.norm(model.data_driven(t,train),dim=2)**2)
    l2_loss = lam * torch.sum(torch.linalg.norm(sol - train,dim = 2)) / batch_size
    loss = data_loss + l2_loss
    loss.backward()
    optimizer.step()

    if i % 5 ==0:
        lam += 2



    if i % 1 == 0:
        # Track physical model parameters
        print('*' * 20)
        print(f'iteration {i}')
        print(f'Data contribution = {data_loss}')
        print(f'loss contribution =  {l2_loss}')
        print(loss)
        for name, param in model.named_parameters():
            if param.requires_grad and name in ['omega','alpha']:
                print(name, param.data)





torch.save(model,'model.pt')


# sol = odeint_adjoint(model,init_state , t, atol=1e-4, rtol=1e-4,method='dopri5')
# pos = sol[:,:,0].transpose(0,1)
# vel = sol[:,:,1].transpose(0,1)
# plt.plot(t,pos[0,:].detach())
# plt.plot(t,train[0,:,0])
# plt.show()
