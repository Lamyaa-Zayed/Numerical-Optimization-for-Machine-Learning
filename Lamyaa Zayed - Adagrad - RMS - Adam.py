#!/usr/bin/env python
# coding: utf-8

# ## importing libraries

# In[1]:


import numpy as np  
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import r2_score   
from math import sqrt


# ## preparing data

# In[2]:


a = -1
b = 2
x = np.linspace(0,20)
y = a * x + b


# In[3]:


print("x=",x)
print("y=",y)


# In[4]:


plt.scatter(x,y)


# ## Adagrad implementation

# In[5]:


def Adagrad (x, y, alpha, v0, v1, epoch):
    theta_0 = 0
    theta_1 = 0
    epsilon = 1e-8

    cost_function = []
    theta0_values = []
    theta1_values = []
    new_hypothesis = []
    v0_list = []
    v1_list = []
    hypothesis = np.zeros(len(x))
    
    for i in range (epoch):
        
        hypothesis = theta_0 + theta_1 * x
        new_hypothesis.append(hypothesis)
        
        cost = (1/2*len(x))* np.sum(((hypothesis - y) ** 2))
        cost_function.append(cost)
        
        gradiant_theta0 = (1/len(x)) * np.sum(hypothesis - y)
        gradiant_theta1 = (1/len(x)) * np.sum((hypothesis - y) * x)
        
        v0 = v0 + (gradiant_theta0 ** 2)
        v0_list.append(v0)
        theta_0 = theta_0 - ((alpha / (epsilon + sqrt(v0))) * gradiant_theta0)
        theta0_values.append(theta_0)
        
        v1 = v1 + (gradiant_theta1 ** 2)
        v1_list.append(v1)
        theta_1 = theta_1 - ((alpha / (epsilon + sqrt(v1))) * gradiant_theta1)
        theta1_values.append(theta_1)
                             
        if (i > 0) & (abs(cost_function[i-1] - cost_function[i]) < 0.01):
          print("Stoped after {} iteration\n".format(i+1))
          break
        elif(i == epoch-1):
          print("Stoped after {} iteration\n".format(i+1))
    
        
    print("last theta0: ", theta0_values[-1], "\nlast theta1: ", theta1_values[-1], "\nlast losses: ", cost_function[-1], "\nlast hypothesis: ", new_hypothesis[-1])
    
    return theta0_values, theta1_values, cost_function, new_hypothesis


# In[6]:


alpha = 0.01
v0 = 0
v1 = 0
epoch = 100000

theta0_adagrad, theta1_adagrad, losses_adagrad, hypo_adagrad = Adagrad (x, y, alpha, v0, v1, epoch)


# In[7]:


r2_score(y,hypo_adagrad[-1])


# In[8]:


plt.plot(losses_adagrad)


# In[9]:


plt.plot(theta0_adagrad, losses_adagrad)


# In[10]:


plt.plot(theta1_adagrad, losses_adagrad)


# In[11]:


for  i in range(len(hypo_adagrad)):
    plt.plot(x,hypo_adagrad[i])
plt.show()


# In[12]:


plt.plot(x,hypo_adagrad[-1],'r')
plt.scatter(x,y)
plt.show


# In[13]:


best_hypo = hypo_adagrad[-1]
plt.plot(best_hypo)


# ## RMS Prop implementation

# In[14]:


def RMSprop (x, y, alpha, beta, v0, v1, epoch):
    theta_0 = 0
    theta_1 = 0
    epsilon = 1e-8

    cost_function = []
    theta0_values = []
    theta1_values = []
    new_hypothesis = []
    v0_list = []
    v1_list = []
    hypothesis = np.zeros(len(x))
    
    for i in range (epoch):
        
        hypothesis = theta_0 + theta_1 * x
        new_hypothesis.append(hypothesis)
        
        cost = (1/2*len(x))* np.sum(((hypothesis - y) ** 2))
        cost_function.append(cost)
        
        gradiant_theta0 = (1/len(x)) * np.sum(hypothesis - y)
        gradiant_theta1 = (1/len(x)) * np.sum((hypothesis - y) * x)
        
        v0 = beta * v0 + (1 - beta) * (gradiant_theta0 ** 2)
        v0_list.append(v0)
        
        theta_0 = theta_0 - ((alpha / (epsilon + sqrt(v0))) * gradiant_theta0)
        theta0_values.append(theta_0)
        
        v1 = beta * v1 + (1 - beta) * (gradiant_theta1 ** 2)
        v1_list.append(v1)
        
        theta_1 = theta_1 - ((alpha / (epsilon + sqrt(v1))) * gradiant_theta1)
        theta1_values.append(theta_1)
                             
        if (i > 0) & (abs(cost_function[i-1] - cost_function[i]) < 0.01):
          print("Stoped after {} iteration\n".format(i+1))
          break
        elif(i == epoch-1):
          print("Stoped after {} iteration\n".format(i+1))
    
        
    print("last theta0: ", theta0_values[-1], "\nlast theta1: ", theta1_values[-1], "\nlast losses: ", cost_function[-1], "\nlast hypothesis: ", new_hypothesis[-1])
    
    return theta0_values, theta1_values, cost_function, new_hypothesis


# In[15]:


alpha = 0.01
beta = 0.9
v0 = 0
v1 = 0
epoch = 100000

theta0_rms, theta1_rms, losses_rms, hypo_rms = RMSprop (x, y, alpha, beta, v0, v1, epoch)


# In[16]:


r2_score(y,hypo_rms[-1])


# In[17]:


plt.plot(losses_rms)


# In[18]:


plt.plot(theta0_rms, losses_rms)


# In[19]:


plt.plot(theta1_rms, losses_rms)


# In[20]:


for  i in range(len(hypo_rms)):
    plt.plot(x,hypo_rms[i])
plt.show()


# In[21]:


plt.plot(x,hypo_rms[-1],'r')
plt.scatter(x,y)
plt.show


# In[22]:


best_hypo = hypo_rms[-1]
plt.plot(best_hypo)


# ## Adam implementation

# In[52]:


def Adam (x, y, alpha, beta1, beta2, m0, m1, v0, v1, epsilon, epoch):
    theta_0 = 0
    theta_1 = 0

    cost_function = []
    theta0_values = []
    theta1_values = []
    new_hypothesis = []
    hypothesis = np.zeros(len(x))
    
    for i in range (1, epoch):
        
        hypothesis = theta_0 + theta_1 * x
        new_hypothesis.append(hypothesis)
        
        cost = (1/2*len(x))* np.sum(((hypothesis - y) ** 2))
        cost_function.append(cost)
        
        gradiant_theta0 = (1/len(x)) * np.sum(hypothesis - y)
        gradiant_theta1 = (1/len(x)) * np.sum((hypothesis - y) * x)
        
        m0 = (beta1 * m0) + ((1 - beta1) * gradiant_theta0)
        m0_new = m0 / (1 - (beta1 ** i))
        
        v0 = (beta2 * v0) + ((1 - beta2) * (gradiant_theta0 ** 2))
        v0_new = v0 / (1 - (beta2 ** i))
        
        theta_0 = theta_0 - ((alpha / (epsilon + sqrt(v0_new))) * m0_new)
        theta0_values.append(theta_0)
        
        m1 = (beta1 * m1) + ((1 - beta1) * gradiant_theta1)
        m1_new = m1 / (1 - (beta1 ** i))
        
        v1 = (beta2 * v1) + ((1 - beta2) * (gradiant_theta1 ** 2))
        v1_new = v1 / (1 - (beta2 ** i))
        
        theta_1 = theta_1 - ((alpha / (epsilon + sqrt(v1_new))) * m1_new)
        theta1_values.append(theta_1)
                             
        if (cost < 0.05):
            final_epoch = i
            print("final epoch= ",final_epoch)
            break
    
        
    print("last theta0: ", theta0_values[-1], "\nlast theta1: ", theta1_values[-1], "\nlast losses: ", cost_function[-1], "\nlast hypothesis: ", new_hypothesis[-1])
    
    return theta0_values, theta1_values, cost_function, new_hypothesis


# In[56]:


alpha = 0.05
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
v0 = 0
v1 = 0
m0 = 0
m1 = 0
epoch = 1000

theta0_adam, theta1_adam, losses_adam, hypo_adam = Adam (x, y, alpha, beta1, beta2, m0, m1, v0, v1, epsilon, epoch)


# In[57]:


r2_score(y,hypo_adam[-1])


# In[58]:


plt.plot(losses_adam)


# In[59]:


plt.plot(theta0_adam, losses_adam)


# In[60]:


plt.plot(theta1_adam, losses_adam)


# In[61]:


for  i in range(len(hypo_adam)):
    plt.plot(x,hypo_adam[i])
plt.show()


# In[62]:


plt.plot(x,hypo_adam[-1],'r')
plt.scatter(x,y)
plt.show


# In[63]:


best_hypo = hypo_adam[-1]
plt.plot(best_hypo)


# In[ ]:





# ##  thank you :)
