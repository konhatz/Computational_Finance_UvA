#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm


# In[2]:


#matrix representation
def buildTree(S ,vol , T, N):
    dt = T / N
    matrix = np.zeros((N + 1 , N + 1))
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
    #Iterate over the lower triangle
    for i in np.arange(N+1): #iterate over rows
        for j in np.arange(i+1): #iterate over columns
            matrix[i, j] = S*d**(i-j)*u**j
    return matrix


# In[3]:


#calculating the option value
def valueOptionMatrix(tree, T, r, K, vol, N):
    dt = T / N
    
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
    
    p = (np.exp(r*dt)-d)/(u - d)
    
    columns = tree.shape[1]
    rows = tree.shape[0]
    
    # Walk backward , we start in last row of the matrix
    # Add the payoff function in the last row
    
    for c in np.arange(columns):
        S = tree[rows - 1 , c] # value in the matrix
        tree[rows - 1 , c] = np.max([S - K, 0])

    # For all other rows, we need to combine from previous rows
    # We walk backwards , from the last row to the first row
    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1 , j]
            up = tree[i + 1 , j + 1]
            tree[i , j] = np.exp(-r*dt)*(p*up + (1-p)*down)
    return tree


# In[4]:


tree = buildTree(100, 0.2, 1, 50)
valueoptionmatrix = valueOptionMatrix(tree, 1, 0.06, 99, 0.2, 50)


# In[5]:


#getting the value of the option at time 0
valueOption = valueoptionmatrix[0][0]
valueOption


# In[6]:


#Black-Scholes 
def BlackScholes(S, vol, T, r, K):
    d1 = (np.log(S/K) + (r + 0.5*(vol**2))*T)/((vol)*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return S*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)


# In[7]:


#getting the analytical value of the option at time 0
Optionvalue = BlackScholes(100, 0.2, 1, 0.06, 99)
Optionvalue


# In[11]:


analytical_values = [] #list of analytical values
approx_values = [] #list of approximate values
vols = np.linspace(0.01, 10, 1000) #list of volatilies we are going to test on

for vol in vols:
    tree = buildTree(100, vol, 1, 50)
    valueoptionmatrix = valueOptionMatrix(tree, 1, 0.06, 99, vol, 50)
    
    approx_values.append(valueoptionmatrix[0][0])
    analytical_values.append(BlackScholes(100, vol, 1, 0.06, 99))


# In[12]:


#plot of approximate and analytical values for different values of volatility
plt.plot(vols, analytical_values, label = "Theoretical formula", alpha = 0.5)
plt.plot(vols, approx_values, label = "Approximate value", alpha = 0.5)
plt.legend()


# In[16]:


#error values for different values of volatility
plt.plot(vols, np.array(approx_values) - np.array(analytical_values), label = "Error")


# In[19]:


error_values = [] #error values
analytical_value = BlackScholes(100, 0.2, 1, 0.06, 99) #analytical value

for N in range(1, 101): #trying out different values of N
    tree = buildTree(100, 0.2, 1, N)
    valueoptionmatrix = valueOptionMatrix(tree, 1, 0.06, 99, 0.2, N)
    error_values.append(valueoptionmatrix[0][0] - analytical_value)


# In[21]:


#error plot
plt.figure(figsize=(8,4))
plt.plot(error_values)


# In[22]:


#absolute error plot
plt.figure(figsize=(8,4))
plt.plot(np.abs(error_values))


# In[23]:


S = 100
K = 99
r = 0.06
T = 1
N = 50
vol = 0.2

dt = T/N

u = np.exp(vol*np.sqrt(dt))
d = np.exp(-vol*np.sqrt(dt))

tree = buildTree(100, 0.2, 1, 50)
valueoptionmatrix = valueOptionMatrix(tree, 1, 0.06, 99, 0.2, 50)

#approximate delta value
delta_value = (valueoptionmatrix[1][1] - valueoptionmatrix[1][0])/(S*u - S*d)

#theoretical delta value
d1 = (np.log(S/K) + (r + 0.5*(vol**2))*T)/((vol)*np.sqrt(T))
theoretical_delta_value = norm.cdf(d1)


# In[25]:


print(delta_value, theoretical_delta_value)


# In[26]:


vols = np.linspace(0.01, 10, 50)
delta_values = [] #list of approximate delta values
theoretical_delta_values = [] #list of theoretical delta values

for vol in vols: #experimenting for different values of volatility
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))

    tree = buildTree(100, vol, 1, 50)
    valueoptionmatrix = valueOptionMatrix(tree, 1, 0.06, 99, vol, 50)

    delta_values.append((valueoptionmatrix[1][1] - valueoptionmatrix[1][0])/(S*u - S*d))
    d1 = (np.log(S/K) + (r + 0.5*(vol**2))*T)/((vol)*np.sqrt(T))
    theoretical_delta_values.append(norm.cdf(d1))


# In[27]:


#plot of approximate and analytical values of delta for different volatility values
plt.plot(vols, delta_values, label = "Binomial model delta")
plt.plot(vols, theoretical_delta_values, label = "Black Scholes model delta")
plt.legend()


# In[53]:


#calculating the american option value
def valueAmericanOptionMatrix(tree, T, r, K, vol, N, type_): #new input variable type - either 'call' or 'put'
    dt = T / N
    
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
    
    p = (np.exp(r*dt)-d)/(u - d)
    
    columns = tree.shape[1]
    rows = tree.shape[0]
    
    # Walk backward , we start in last row of the matrix
    # Add the payoff function in the last row
    
    for c in np.arange(columns):
        S = tree[rows - 1 , c] # value in the matrix
        if type_ == 'call':
            tree[rows - 1 , c] = np.max([S - K, 0])
        if type_ == 'put':
            tree[rows - 1 , c] = np.max([K - S, 0])

    # For all other rows, we need to combine from previous rows
    # We walk backwards , from the last row to the first row
    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1 , j]
            up = tree[i + 1 , j + 1]
            if type_ == 'call':
                tree[i , j] = np.max([np.max([tree[i , j] - K, 0]),np.exp(-r*dt)*(p*up + (1-p)*down)]) #exercise now if the expected payoff in the future is lower
            if type_ == 'put':
                tree[i , j] = np.max([np.max([K - tree[i , j], 0]),np.exp(-r*dt)*(p*up + (1-p)*down)])   
    return tree


# In[54]:


tree = buildTree(100, 0.2, 1, 50)
valueAmericanOptionMatrix(tree, 1, 0.06, 99, 0.2, 50, "call")[0][0] #call value


# In[57]:


tree = buildTree(100, 0.2, 1, 50)
valueAmericanOptionMatrix(tree, 1, 0.06, 99, 0.2, 50, "put")[0][0] #put value


# In[68]:


vols = np.linspace(0.1, 10, 50)
american_call_values = [] #list of american call values
american_put_values = [] #list of american put values

for vol in vols: #experimenting for different values of volatility
    tree = buildTree(100, vol, 1, 50)
    american_call_values.append(valueAmericanOptionMatrix(tree, 1, 0.06, 99, vol, 50, "call")[0][0])
    tree = buildTree(100, vol, 1, 50)
    american_put_values.append(valueAmericanOptionMatrix(tree, 1, 0.06, 99, vol, 50, "put")[0][0])


# In[69]:


#plot american call values
plt.plot(vols, american_call_values, label = 'American call value')
plt.plot(vols, american_put_values, label = 'American put value')
plt.legend()

