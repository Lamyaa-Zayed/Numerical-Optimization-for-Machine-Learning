#!/usr/bin/env python
# coding: utf-8

# ## Practical Work 4

# For this practical work, the student will have to develop a Python program that is able to implement the accelerated gradient descent methods with adaptive learning rate <b>(Adagrad, RMSProp, and Adam)</b> in order to achieve the linear regression of a set of datapoints.

# #### Import numpy, matplotlib.pyplot and make it inline

# In[1]:


import numpy as np  
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import r2_score   
from math import sqrt


# To have a dataset or set of data points, the student must generate a pair of arrays <b>X</b> and <b>y</b> with the values in <b>X</b> equally distributed between <b>0</b> and <b>20</b> and the values in <b>y</b> such that: 
# <b>yi = a*xi + b (and a = -1, b = 2)</b>
# 

# In[2]:


a = -1
b = 2
x = np.linspace(0,20)
y = a * x + b


# In[3]:


print("x=",x)
print("y=",y)


# #### Plot your data points. 

# In[4]:


plt.scatter(x,y)


# ## Adagrad

# ### For a single variable linear regression ML model, build a function to find the optimum Theta_0 and Theta_1 parameters using Adagrad optimization algorithm.
# #### The funtion should have the following input parameters:
# ##### 1. Input data as a matrix (or vector based on your data).
# ##### 2. Target label as a vector.
# ##### 3. Learning rate.
# ##### 4. Epsilon.
# ##### 5. Maximum number of iterations (Epochs).
# #### The funtion should return the following outputs:
# ##### 1. All predicted Theta_0 in all iterations.
# ##### 2. All predicted Theta_1 in all iterations.
# ##### 3. Corresponding loss for each Theta_0 and Theta_1 predictions.
# ##### 4.All hypothesis outputs (prdicted labels) for each Theta_0 and Theta_1 predictions.
# ##### 5.Final Optimum values of Theta_0 and Theta_1.
# #### Choose the suitable number of iterations, learning rate, Epsilon, and stop criteria.
# #### Calculate r2 score. Shouldn't below 0.9
# #### Plot the required curves (loss-epochs, loss-theta0, loss-theta1, all fitted lines per epoch (single graph), best fit line)
# #### Try different values of the huperparameters and see the differnce in your results.

# ![image.png](attachment:image.png)

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
    
        
    print("optimum theta0: ", theta0_values[-1], "\noptimum theta1: ", theta1_values[-1], "\nlast losses: ", cost_function[-1], "\nlast hypothesis: ", new_hypothesis[-1])
    
    return theta0_values, theta1_values, cost_function, new_hypothesis


# In[6]:


alpha = 0.05
v0 = 0
v1 = 0
epoch = 10000

theta0_adagrad, theta1_adagrad, losses_adagrad, hypo_adagrad = Adagrad (x, y, alpha, v0, v1, epoch)


# In[7]:


print("R2Score: ", r2_score(y,hypo_adagrad[-1]))


# In[13]:


plt.plot(losses_adagrad)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("losses over epochs for Adagrad")


# In[14]:


plt.plot(theta0_adagrad, losses_adagrad,'r')
plt.scatter(theta0_adagrad, losses_adagrad)
plt.xlabel('theta0')
plt.ylabel('Losses')
plt.title("theta0 and losses for Adagrad")


# In[15]:


plt.plot(theta1_adagrad, losses_adagrad,'r')
plt.scatter(theta1_adagrad, losses_adagrad)
plt.xlabel('theta1')
plt.ylabel('Losses')
plt.title("theta1 and losses for Adagrad")


# In[37]:


for  i in range(len(hypo_adagrad)):
    plt.plot(x,hypo_adagrad[i])
plt.scatter(x,y)
plt.title("all fitted lines per epoch for Adagrad")


# In[16]:


best_hypo = hypo_adagrad[-1]
plt.scatter(x,y)
plt.plot(x, best_hypo,'r')
plt.title("best fitted line for Adagrad")


# In[ ]:





# ## RMSProp

# ### Update the previos implementation to be RMSProp.
# #### Compare your results with Adagrad results.

# ![image.png](attachment:image.png)

# In[17]:


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
    
        
    print("optimum theta0: ", theta0_values[-1], "\noptimum theta1: ", theta1_values[-1], "\nlast losses: ", cost_function[-1], "\nlast hypothesis: ", new_hypothesis[-1])
    
    return theta0_values, theta1_values, cost_function, new_hypothesis


# In[18]:


alpha = 0.05
beta = 0.9
v0 = 0
v1 = 0
epoch = 10000

theta0_rms, theta1_rms, losses_rms, hypo_rms = RMSprop (x, y, alpha, beta, v0, v1, epoch)


# In[19]:


print("R2Score: ", r2_score(y,hypo_rms[-1]))


# In[21]:


plt.plot(losses_rms)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("losses over epochs for Rms")


# In[22]:


plt.plot(theta0_rms, losses_rms, 'r')
plt.scatter(theta0_rms,losses_rms)
plt.xlabel('theta0')
plt.ylabel('Losses')
plt.title("theta0 and losses for Rms")


# In[23]:


plt.plot(theta1_rms, losses_rms, 'r')
plt.scatter(theta1_rms,losses_rms)
plt.xlabel('theta1')
plt.ylabel('Losses')
plt.title("theta1 and losses for Rms")


# In[36]:


for  i in range(len(hypo_rms)):
    plt.plot(x,hypo_rms[i])
plt.scatter(x,y)
plt.title("all fitted lines per epoch for Rms")


# In[25]:


best_hypo = hypo_rms[-1]
plt.plot(x,best_hypo,'r')
plt.scatter(x,y)
plt.title("best fitted line for Rms")


# In[ ]:





# ## Adam

# ### Update the previos implementation to be Adam.
# #### Compare your results with Adagrad and RMSProp results.

# ![image-4.png](attachment:image-4.png)

# In[26]:


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
    
        
    print("optimum theta0: ", theta0_values[-1], "\noptimum theta1: ", theta1_values[-1], "\nlast losses: ", cost_function[-1], "\nlast hypothesis: ", new_hypothesis[-1])
    
    return theta0_values, theta1_values, cost_function, new_hypothesis


# In[27]:


alpha = 0.05
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
v0 = 0
v1 = 0
m0 = 0
m1 = 0
epoch = 10000

theta0_adam, theta1_adam, losses_adam, hypo_adam = Adam (x, y, alpha, beta1, beta2, m0, m1, v0, v1, epsilon, epoch)


# In[28]:


print("R2Score: ", r2_score(y,hypo_adam[-1]))


# In[30]:


plt.plot(losses_adam)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("losses over epochs for Adam")


# In[31]:


plt.plot(theta0_adam, losses_adam, 'r')
plt.scatter(theta0_adam,losses_adam)
plt.xlabel('theta0')
plt.ylabel('Losses')
plt.title("theta0 and losses for Adam")


# In[32]:


plt.plot(theta1_adam, losses_adam, 'r')
plt.scatter(theta1_adam,losses_adam)
plt.xlabel('theta1')
plt.ylabel('Losses')
plt.title("theta1 and losses for Adam")


# In[35]:


for  i in range(len(hypo_adam)):
    plt.plot(x,hypo_adam[i])
plt.scatter(x,y)
plt.title("all fitted lines per epoch for Adam")


# In[33]:


best_hypo = hypo_adam[-1]
plt.plot(x, best_hypo,'r')
plt.scatter(x,y)
plt.title("best fitted line for Adam")


# In[ ]:





# ## all above optimizations are implemented for the same parameters ...

# ## using different parameters for the three types of optimizations:

# In[43]:


##### Adagrad #####

alpha = 0.1
v0 = 0
v1 = 1
epoch = 5000

theta0_adagrad1, theta1_adagrad1, losses_adagrad1, hypo_adagrad1 = Adagrad (x, y, alpha, v0, v1, epoch)


# In[44]:


print("R2Score: ", r2_score(y,hypo_adagrad1[-1]))


# In[45]:


plt.plot(losses_adagrad1)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("losses over epochs for Adagrad")


# In[46]:


plt.plot(theta0_adagrad1, losses_adagrad1,'r')
plt.scatter(theta0_adagrad1, losses_adagrad1)
plt.xlabel('theta0')
plt.ylabel('Losses')
plt.title("theta0 and losses for Adagrad")


# In[47]:


plt.plot(theta1_adagrad1, losses_adagrad1,'r')
plt.scatter(theta1_adagrad1, losses_adagrad1)
plt.xlabel('theta1')
plt.ylabel('Losses')
plt.title("theta1 and losses for Adagrad")


# In[48]:


for  i in range(len(hypo_adagrad1)):
    plt.plot(x,hypo_adagrad1[i])
plt.scatter(x,y)
plt.title("all fitted lines per epoch for Adagrad")


# In[49]:


best_hypo = hypo_adagrad1[-1]
plt.scatter(x,y)
plt.plot(x, best_hypo,'r')
plt.title("best fitted line for Adagrad")


# In[52]:


#### RMS PROP #####

alpha = 0.1
beta = 0.5
v0 = 0
v1 = 1
epoch = 500

theta0_rms1, theta1_rms1, losses_rms1, hypo_rms1 = RMSprop (x, y, alpha, beta, v0, v1, epoch)


# In[53]:


print("R2Score: ", r2_score(y,hypo_rms1[-1]))


# In[54]:


plt.plot(losses_rms1)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("losses over epochs for Rms")


# In[55]:


plt.plot(theta0_rms1, losses_rms1, 'r')
plt.scatter(theta0_rms1,losses_rms1)
plt.xlabel('theta0')
plt.ylabel('Losses')
plt.title("theta0 and losses for Rms")


# In[56]:


plt.plot(theta1_rms1, losses_rms1, 'r')
plt.scatter(theta1_rms1,losses_rms1)
plt.xlabel('theta1')
plt.ylabel('Losses')
plt.title("theta1 and losses for Rms")


# In[57]:


for  i in range(len(hypo_rms1)):
    plt.plot(x,hypo_rms1[i])
plt.scatter(x,y)
plt.title("all fitted lines per epoch for Rms")


# In[59]:


best_hypo = hypo_rms1[-1]
plt.plot(x,best_hypo,'r')
plt.scatter(x,y)
plt.title("best fitted line for Rms")


# In[61]:


##### Adam ####

alpha = 0.1
beta1 = 0.5
beta2 = 0.5
epsilon = 1e-8
v0 = 0
v1 = 1
m0 = 0
m1 = 1
epoch = 500

theta0_adam1, theta1_adam1, losses_adam1, hypo_adam1 = Adam (x, y, alpha, beta1, beta2, m0, m1, v0, v1, epsilon, epoch)


# In[62]:


print("R2Score: ", r2_score(y,hypo_adam1[-1]))


# In[63]:


plt.plot(losses_adam1)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("losses over epochs for Adam")


# In[64]:


plt.plot(theta0_adam1, losses_adam1, 'r')
plt.scatter(theta0_adam1,losses_adam1)
plt.xlabel('theta0')
plt.ylabel('Losses')
plt.title("theta0 and losses for Adam")


# In[69]:


plt.plot(theta1_adam1, losses_adam1, 'r')
plt.scatter(theta1_adam1,losses_adam1)
plt.xlabel('theta1')
plt.ylabel('Losses')
plt.title("theta1 and losses for Adam")


# In[66]:


for  i in range(len(hypo_adam1)):
    plt.plot(x,hypo_adam1[i])
plt.scatter(x,y)
plt.title("all fitted lines per epoch for Adam")


# In[67]:


best_hypo = hypo_adam1[-1]
plt.plot(x, best_hypo,'r')
plt.scatter(x,y)
plt.title("best fitted line for Adam")


# ## Congratulations 
# ![image.png](attachment:image.png)

# In[ ]:




