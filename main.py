import torch
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import data_utils
from models.models import PendulumModel



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PendulumModel(frictionless = True,include_data = False).to(device)
train,val,test = data_utils.get_pendulum_datasets(n=25)



t0 = 0.
te = 20.
t = torch.linspace(t0, te, 40)

def loss_fn(model,train):
    train = train.to(device)
    output = odeint_adjoint(model, (0,1), t, atol=1e-8, rtol=1e-8, method='dopri5').transpose(0,1)
    l2norm = torch.linalg.norm(output - train,dim = 2)
    l2loss = torch.sum(l2norm)

    return l2loss

train =train.to(device)
t = t.to(device)

init_state = (train[:,0,0] ,train[:,0,1])
print(init_state)
optimizer = optim.Adam(model.parameters(), lr=0.03)
loss_function = torch.nn.MSELoss(reduction = 'sum')



pos,vel = odeint_adjoint(model,init_state , t, atol=1e-8, rtol=1e-8,method='dopri5')
plt.plot(t,pos[:,0].detach())
plt.plot(t,train[0,:,0])
plt.show()


for i in range(50):
    optimizer.zero_grad()
    pos,vel = odeint_adjoint(model,init_state , t, atol=1e-8, rtol=1e-8,method='dopri5')
    pos = pos.transpose(0,1)
    vel = vel.transpose(0,1)
    loss = loss_function(pos,train[:,:,0]) + loss_function(vel,train[:,:,1])
    loss.backward()
    optimizer.step()

    print('*' * 20)
    print(loss)
    for name, param in model.named_parameters():
        if param.requires_grad and name in ['omega','alpha']:
            print(name, param.data)




pos,vel = odeint_adjoint(model,init_state , t, atol=1e-8, rtol=1e-8,method='dopri5')
plt.plot(t,pos[:,0].detach())
plt.plot(t,train[0,:,0])
plt.show()
