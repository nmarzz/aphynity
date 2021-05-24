import torch
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import data_utils
from models.models import PendulumModel2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PendulumModel2().to(device)
train,val,test = data_utils.get_pendulum_datasets(n=1)



t0 = 0.
te = 20.
t = torch.linspace(t0, te, 40)

def loss_fn(model,train):
    train = train.to(device)
    output = odeint_adjoint(model, (0,1), t, atol=1e-8, rtol=1e-8, method='dopri5').transpose(0,1)
    l2norm = torch.linalg.norm(output - train,dim = 2)
    l2loss = torch.sum(l2norm)

    return l2loss

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

#
# print('Training...')
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum = 0.9)
# for i in range(2):
#     optimizer.zero_grad()
#     loss = loss_fn(model,train)
#     print('Computing gradient')
#     loss.backward()
#     optimizer.step()




train =train.to(device)
t = t.to(device)

init_state = (train[0,0,0] ,train[0,0,1])
print(init_state)
optimizer = optim.Adam(model.parameters(), lr=0.01)



for i in range(5):
    optimizer.zero_grad()
    pos,vel = odeint_adjoint(model,init_state , t, atol=1e-8, rtol=1e-8,method='dopri5')
    pos_loss = pos - train[0,:,0]
    vel_loss = vel - train[0,:,1]
    loss = torch.sum(torch.sqrt(pos_loss ** 2 + vel_loss**2))
    loss.backward()
    optimizer.step()

    print('*' * 20)
    print(loss)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)





# plt.plot(t,pos.detach())
# plt.plot(t,train[0,:,0])
# plt.show()
