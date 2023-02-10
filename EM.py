import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


sigma = 0.2
r     = 0.06
T     = 1
M     = 365 ## or 52 (weekly)
dt    = T/M
K     = 100
t = np.linspace(0,T,M)

S_0   = 99
sims = {}
n_sims = 10000


#### Euler - Maruyama Option pricing
def EM(sigma,r,S_0,T,M):
    dt = T/M
    t = np.linspace(0,T,M)
    x = np.zeros(M)
    x[0] = S_0
    for i in range(M-1):
        x[i+1] = x[i] + dt*(r)*x[i]+ sigma*np.sqrt(dt)*x[i]*np.random.normal()
    return x

#### Pricing method using exact GBM solution
def Exact_solution(sigma,r,S_0,T,M):
    dt = T/M
    t = np.linspace(0,T,M)
    dB = np.sqrt(dt)*np.random.randn(M)
    B  = np.cumsum(dB)
    S = S_0*np.exp(sigma*B + (r - 1/2*sigma**2)*t)
    return S


#### Using exact solution

for i in range(n_sims):
    sims[i] = Exact_solution(sigma,r,S_0,T,M)

data = pd.DataFrame(sims)
price_maturity = data.values[-1:]
call_option_values = np.zeros(n_sims)
for n in range(n_sims):
    call_option_values[n] = np.e**(-r*T)*np.maximum(price_maturity[0][n]-K,0)
plt.hist(call_option_values)
plt.show()

print(np.mean(call_option_values))

#### Using EM method

for i in range(n_sims):
    sims[i] = EM(sigma,r,S_0,T,M)

for i in range(10):
    plt.plot(t, sims[i], lw=2)
plt.show()

data = pd.DataFrame(sims)
price_maturity = data.values[-1:]
call_option_values = np.zeros(n_sims)
for n in range(n_sims):
    call_option_values[n] = np.e**(-r*T)*np.maximum(price_maturity[0][n]-K,0)
plt.hist(call_option_values)
plt.show()

print(np.mean(call_option_values))

vol = np.arange(0.1,0.9,0.1)

call_option_vol = []
for vol_ in vol:
    call_option_values = np.zeros(n_sims)
    for i in range(n_sims):
        sims[i] = EM(vol_,r,S_0,T,M)
    data = pd.DataFrame(sims)
    price_maturity = data.values[-1:]
    for i in range(n_sims):
        call_option_values[i] = np.e**(-r*T)*np.maximum(price_maturity[0][i] - K,0)
    call_option_vol.append(np.mean(call_option_values))

plt.plot(vol,call_option_vol)
plt.show()
