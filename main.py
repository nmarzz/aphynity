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
n = 40
dt = te  / n

t = torch.linspace(t0, te, n).to(device)
model = PendulumModel(frictionless = True,include_data = True).to(device)
batch_size = 20
train,val,test = data_utils.get_pendulum_datasets(n=batch_size)
train,val,test = train.to(device),val.to(device),test.to(device)
init_state = train[:,0,:]


# Define accessories
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss(reduction = 'mean')



# Training loop
# TODO: Add testing loop
lam = 1
for i in range(50):

    optimizer.zero_grad()
    sol = odeint_adjoint(model,init_state , t, atol=1e-2, rtol=1e-2,method='dopri5').transpose(0,1)

    fa = model.data_driven(train)
    L2fa = torch.sum(torch.linalg.norm(fa,dim=1)**2) # Ok, so finding the L2 norm is fairly easy
    diff_fa = (fa[:,2:,:] - fa[:,0:-2,:]) / (2*dt)
    L2fa_grad = torch.sum(torch.linalg.norm(diff_fa,dim=1)**2)


    data_loss = torch.sqrt((L2fa + L2fa_grad)/batch_size)
    l2_loss = lam * torch.sum(torch.linalg.norm(sol - train,dim = 2)) / batch_size
    loss = torch.sqrt(data_loss) + l2_loss
    loss.backward()
    optimizer.step()

    if i % 10 ==0:
        train,val,test = data_utils.get_pendulum_datasets(n=batch_size)
        train,val,test = train.to(device),val.to(device),test.to(device)
        init_state = train[:,0,:]
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
#
# sol = odeint_adjoint(model,init_state , t, atol=1e-4, rtol=1e-4,method='dopri5')
# pos = sol[:,:,0].transpose(0,1)
# vel = sol[:,:,1].transpose(0,1)
# plt.plot(t,pos[0,:].detach())
# plt.plot(t,train[0,:,0])
# plt.show()
