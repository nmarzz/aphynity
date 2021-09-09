from numpy.core.numeric import Inf
import torch
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import data_utils
from models import PendulumModel
import copy

# Get environment
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device}')

# Define integration parameters
t0 = 0.
te = 20.
t = torch.linspace(t0, te, 40).to(device)
include_neural_net = True
model = PendulumModel(frictionless = True,include_neural_net = include_neural_net).to(device)

# dynamical system parameters
T0 = 12
omega = 2 * 3.1415 / T0
alpha = 0.35
print(f'Expecting an omega of {omega}')

# Get data
batch_size = 25
train,val,test = data_utils.get_pendulum_datasets(n=batch_size,T0 = T0,alpha = alpha)
train,val,test = train.to(device),val.to(device),test.to(device)




# Define accessories
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model():
    model.train()
    optimizer.zero_grad()

    # Calculate loss
    init_state = train[:,0,:]
    sol = odeint_adjoint(model,init_state , t, atol=1e-2, rtol=1e-2,method='dopri5').transpose(0,1)
    if include_neural_net:
        nn_loss = torch.sum(torch.linalg.norm(model.neural_net(t,train),dim=2)**2) / batch_size
    else:
        nn_loss = 0
    l2_loss = lam * torch.sum(torch.linalg.norm(sol - train,dim = 2)) / batch_size
    
    loss = nn_loss + l2_loss
    
    # Apply backprop
    loss.backward()    
    optimizer.step()


def test_model():
    model.eval()        
    with torch.no_grad():
        # Get data        
        init_state = test[:,0,:]
        sol = odeint_adjoint(model,init_state , t, atol=1e-2, rtol=1e-2,method='dopri5').transpose(0,1)        
        
        # Calculate loss
        if include_neural_net:            
            nn_loss =  torch.sum(torch.linalg.norm(model.neural_net(t,test),dim=2)**2) / batch_size
        else:
            nn_loss = 0
        l2_loss = lam * torch.sum(torch.linalg.norm(sol - test,dim = 2)) / batch_size
        loss = nn_loss + l2_loss
        
    return loss, nn_loss,l2_loss


lam = 1
patience = 40
epochs_since_last_improvement = 0
best_loss = Inf
for i in range(2000):
    train_model()
    loss, nn_loss,l2_loss = test_model()    

    # Print training progress
    if i % 1 == 0:
        # Track physical model parameters
        print('*' * 20)
        print(f'iteration {i}')
        print(f'Data contribution = {nn_loss}')
        print(f'L2 contribution =  {l2_loss}')
        print(f'Total loss = {loss}')
        for name, param in model.named_parameters():
            if param.requires_grad and name in ['omega','alpha']:
                print(f'{name} = {param.data.item()} ')

    # Implement early stopping
    if loss < best_loss:
        best_loss = loss
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_since_last_improvement = 0
    else:
        epochs_since_last_improvement += 1
        if epochs_since_last_improvement >= patience:
            break

    # Update lambda parameter
    # if epochs_since_last_improvement >= (patience //2):
    #     lam += 2


model.load_state_dict(best_model_weights)
model = model.to('cpu')

print('Final named parameter values are: ')
for name, param in model.named_parameters():
            if param.requires_grad and name in ['omega','alpha']:
                print(f'{name} = {param.data.item()} ')


train,val,test = data_utils.get_pendulum_datasets(n=batch_size,T0 = T0,alpha = alpha)
train,val,test = train.to('cpu'),val.to('cpu'),test.to('cpu')
init_state = train[:,0,:]

t = t.to('cpu')
sol = odeint_adjoint(model,init_state , t, atol=1e-4, rtol=1e-4,method='dopri5')
pos = sol[:,:,0].transpose(0,1)
vel = sol[:,:,1].transpose(0,1)
plt.plot(t,pos[0,:].detach())
plt.plot(t,train[0,:,0])
plt.legend(['Learnt','True'])
plt.savefig('final_result.png')



torch.save(model,'model.pt')