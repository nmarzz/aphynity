from  scipy.integrate import solve_ivp
import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64

def get_pendulum_datasets(n = 25,t0 = 0,te = 20,dt = 0.5,T0 = 12,alpha = 0.2):
    '''
From paper:
    For each train / validation / test split, we simulate a dataset with 25 trajectories of 40
    timesteps (time interval [0, 20], timestep δt = 0.5) with fixed ODE coefficients (T0 = 12, α = 0.2)
    and varying initial conditions. The simulation integrator is Dormand-Prince Runge-Kutta method
    of order (4)5 (DOPRI5, Dormand & Prince, 1980). We also add a small amount of white gaussian
    noise (σ = 0.01) to the state. Note that our pendulum dataset is much more challenging than the
    ideal frictionless pendulum considered in Greydanus et al. (2019).
    '''
    omega_0 = 2 * 3.1415 / T0
    alpha = 0.2
    time = np.arange(start = t0,stop = te,step =dt)
    ode = functools.partial(pendulum_ode,omega_0,alpha)

    train,val,test = _generate_samples(n,ode,(t0,te),time),_generate_samples(n,ode,(t0,te),time),_generate_samples(n,ode,(t0,te),time)

    return train,val,test

def _generate_samples(n,ode,t_span,time):
    rng = np.random.default_rng(1331)    
    samples = []
    for i in range(n):
        ics = rng.uniform(low = -1,high = 1,size = (2))
        sol = solve_ivp(fun = ode,t_span =t_span,y0 = ics,method = 'RK45',t_eval = time,vectorized = True)
        samples.append(sol.y.transpose())

    samples = torch.tensor(samples,dtype = torch.float)
    samples += torch.randn_like(samples) * 0.01
    return samples

def pendulum_ode(omega_0, alpha,t,x):
    pos,vel = x[0],x[1]
    dpos = vel
    dvel = (-omega_0 ** 2) * np.sin(pos) - alpha * vel

    return np.array([dpos,dvel])



if __name__ == '__main__':
    train,val,test = get_pendulum_datasets(n = 47)
    print(train.shape)
    print(train)

    plt.plot(train[0,0,:])
    plt.show()


    train += torch.randn_like(train)* 0.01
