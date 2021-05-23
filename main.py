import torch
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import data_utils
from models.models import PendulumModel


def loss_fn(model,train):
    train = train.to(device)

    output = odeint_adjoint(model, train[:,0,:], t, method='dopri5').transpose(0,1)
    l2norm = torch.linalg.norm(output - train,dim = 2)
    l2loss = torch.sum(l2norm)

    Fa = model.data_driven(train)
    fanorm = torch.linalg.norm(Fa,dim = 2)
    faloss = torch.sum(fanorm)

    return faloss + l2loss

t0 = 0.
te = 20.
t = torch.linspace(t0, te, 40)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = PendulumModel(frictionless = True,include_data = True).to(device)
train,val,test = data_utils.get_pendulum_datasets()
loss = loss_fn(model,train)





optimizer = optim.SGD(model.parameters(), lr=0.1, momentum = 0.9)

print('Training...')
for i in range(5):
    print(f'Epoch {i}')
    optimizer.zero_grad()
    loss = loss_fn(model,train)
    loss.backward()
    optimizer.step()


# output = odeint_adjoint(model, y0, t, method='dopri5').transpose(0,1)
#
#
# plt.plot(t,output[0,:,0].detach())
# plt.plot(t,test[0,:,0])
# plt.show()
