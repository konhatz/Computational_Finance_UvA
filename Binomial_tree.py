import numpy as np
import scipy
import matplotlib.pyplot as plt

def buildTree(S,vol,T,N):
    dt = T/N

    matrix = np.zeros((N+1,N+1))

    u = np.e**(vol*np.sqrt(dt))
    d = np.e**(-vol*np.sqrt(dt))

    matrix[0,0] = S
    for j in range(1,N+1):
        for i in range(0,j):
            matrix[i,j] = matrix[i,j-1]*u
        matrix[i+1,j]     = matrix[i,j-1]*d
    return matrix

# #
# def buildTree_2(S,vol,T,N):
#     dt = T/N
#
#     matrix = np.zeros((N+1,N+1))
#
#     u = np.e**(vol*np.sqrt(dt))
#
#     d = np.e**(-vol*np.sqrt(dt))
#
#     for j in range(0,N+1):
#         for i in range(0,j+1):
#             matrix[i,j] = S*d**j*u**(i-j)
#     print(np.around(matrix,2))
#     return matrix

S_0 = 100
vol = 0.2
T   = 1
# N   = 1000
# print(buildTree_2(S_0,vol,T,N))


def valueOptionMatrix(stockprice,T,r,K,vol,N):
    dt = T/N

    u = np.e**(vol*np.sqrt(dt))
    d = np.e**(-vol*np.sqrt(dt))
    q =(np.e**(r*dt) - d)/(u-d)
    call_option = np.zeros((N+1,N+1))
    call_option[:,-1] = np.where(stockprice[:,-1]-K<0,0,stockprice[:,-1]-K)
    for j in range(N-1,-1,-1):
        for i in range(0,j+1):
            call_option[i,j] = (q*call_option[i,j+1]+(1-q)*call_option[i+1,j+1])*np.e**(-r*dt)
    delta_t_0 = (call_option[0,1] - call_option[1,1])/(stockprice[0,0]*u - stockprice[0,0]*d)
    return call_option[0,0],delta_t_0


r = 0.06
K = 99


def Black_Scholes_value(S,r,T,K,vol):
    d_1 = (np.log(S/K)+ (r+1/2*vol**2))*T/(vol*np.sqrt(T))
    d_2 = d_1 - vol*np.sqrt(T)
    return S*scipy.stats.norm(0, 1).cdf(d_1) - np.e**(-r*T)*K*scipy.stats.norm(0, 1).cdf(d_2)

#### Change of volatility - Question 2

vol_values = np.arange(0.1,0.9,0.1)
N = 50
values_dict = {}
Black_scholes_values = []
CRR_values = []
for vol_ in vol_values:
    black_scholes_value = Black_Scholes_value(S_0,r,T,K,vol_)
    stock_price_values  = buildTree(S_0,vol_,T,N)
    call_option_CRR     = valueOptionMatrix(stock_price_values,T,r,K,vol_,N)[0]
    CRR_values.append(call_option_CRR)
    Black_scholes_values.append(black_scholes_value)
    values_dict[vol_] = [black_scholes_value,call_option_CRR]

keys = [key for key in values_dict.keys()]
values = [value for value in values_dict.values()]
fig, ax = plt.subplots()
ax.bar(np.arange(len(keys)) - 0.2, [value[0] for value in values],
       width=0.2, color='b', align='center')
ax.bar(np.arange(len(keys)) + 0.2,
       [value[1] if len(value) == 2 else 0 for value in values],
       width=0.2, color='g', align='center')
ax.set_xticklabels(keys)
ax.set_xticks(np.arange(len(keys)))
plt.show()
plt.plot()


plt.plot(vol_values,Black_scholes_values)
plt.plot(vol_values,CRR_values)
plt.show()

#
# #### Convergence - Question 3
N = np.arange(10,1000,10)
call_values_list = []
error_list = []
for n in N:
    stockprice = buildTree(S_0,vol,T,n)
    call_value = valueOptionMatrix(stockprice,T,r,K,vol,n)[0]
    call_values_list.append(call_value)
    black_scholes_value = Black_Scholes_value(S_0,r,T,K,vol)
    error_list.append(call_value - black_scholes_value)




## Convergence error versus N
fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(N, error_list,label = '$error-Binomial Tree$',s = 20)
# Set logarithmic scale on the y variable
ax.set_yscale("log");
ax.set_ylabel(r"$\epsilon$")
ax.set_title("error convergence")
ax.set_xlabel('N')
ax.legend()
plt.show()

#### Delta parameters - Question 4

vol_values = np.arange(0.1,0.9,0.1)
N = 50
values_dict = {}
Black_scholes_delta = []
CRR_Delta = []
for vol_ in vol_values:
    Deltas_BS = scipy.stats.norm(0, 1).cdf((np.log(S_0/K)+ (r+1/2*vol_**2))*T/(vol_*np.sqrt(T)))
    stock_price_values  = buildTree(S_0,vol_,T,N)
    Deltas_CRR     = valueOptionMatrix(stock_price_values,T,r,K,vol_,N)[1]
    values_dict[vol_] = [Deltas_BS,Deltas_CRR]
    Black_scholes_delta.append(Deltas_BS)
    CRR_Delta.append(Deltas_CRR)





keys = [key for key in values_dict.keys()]
values = [value for value in values_dict.values()]
fig, ax = plt.subplots()
ax.bar(np.arange(len(keys)) - 0.2, [value[0] for value in values],
       width=0.2, color='b', align='center')
ax.bar(np.arange(len(keys)) + 0.2,
       [value[1] if len(value) == 2 else 0 for value in values],
       width=0.2, color='g', align='center')
ax.set_xticklabels(keys)
ax.set_xticks(np.arange(len(keys)))
plt.show()

plt.plot(vol_values,Black_scholes_delta,label = 'Black_scholes')
plt.plot(vol_values,CRR_Delta,label = 'CRR_model')
plt.show()

#### Question 5 - American call/put option

# - American Call same as European Call

##### American put option


def American_Put_option(stockprice,T,r,K,vol,N):
    dt = T/N

    u = np.e**(vol*np.sqrt(dt))
    d = np.e**(-vol*np.sqrt(dt))
    q =(np.e**(r*dt) - d)/(u-d)
    call_option = np.zeros((N+1,N+1))
    call_option[:,-1] = np.where(K-stockprice[:,-1]<0,0,K-stockprice[:,-1])
    for j in range(N-1,-1,-1):
        for i in range(0,j+1):
            call_option[i,j] = np.maximum(stockprice[i,j] - K,(q*call_option[i,j+1]+(1-q)*call_option[i+1,j+1])*np.e**(-r*dt))
    delta = (call_option[0,1] - call_option[1,1])/(stockprice[0,0]*u - stockprice[0,0]*d)
    gammma = (call_option[0,2] - call_option[2,2])/(stockprice[0,0]*u - stockprice[0,0]*d)
    return call_option[0,0],delta




vol_values = np.arange(0.1,0.9,0.1)
N = 50
values_dict = {}
for vol_ in vol_values:
    stock_price_values  = buildTree(S_0,vol_,T,N)
    put_option_CRR_American     = American_Put_option(stock_price_values,T,r,K,vol_,N)[0]
    values_dict[vol_] = [put_option_CRR_American]

keys = [key for key in values_dict.keys()]
values = [value for value in values_dict.values()]
fig, ax = plt.subplots()
ax.bar(np.arange(len(keys)) - 0.2, [value[0] for value in values],
       width=0.2, color='b', align='center')
ax.bar(np.arange(len(keys)) + 0.2,
       [value[1] if len(value) == 2 else 0 for value in values],
       width=0.2, color='g', align='center')
ax.set_xticklabels(keys)
ax.set_xticks(np.arange(len(keys)))
plt.show()
