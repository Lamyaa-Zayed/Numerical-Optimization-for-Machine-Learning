# -*- coding: utf-8 -*-
"""LamyaaZayed_Newton's function.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T1y6nOeU0CFqrBpVexadQK7ePWZk5SI8

Use python code to build a function that calculate the approximate solution 
of a function using Newton’s method.
a.  Think about function arguments.
b.  Choose the suitable stop criteria. 
c.  Use your function to solve the above equations.
"""

import numpy as np

def newtonFunction (functionEquation , initialValue):
  print("Finding the solution for F(X) =\n",functionEquation,"\n")
  derivatedFunction = np.polyder(functionEquation)
  i=0
  x=[initialValue]
  while ((functionEquation(x[i]) != 0) and ( (i<2) or (float("{:.6f}".format(x[i])) != float("{:.6f}".format(x[i-1]))))):
    print('X[',i,'] = ',x[i],' F(x)= ',functionEquation(x[i]),' F\'(x)=',derivatedFunction(x[i]),'\n')
    temp = x[i]-(functionEquation(x[i]) / derivatedFunction(x[i]))
    x.append(temp)
    i+=1
  print("The solution is at X" , i , " and its value is " , x[i])
  print("=======================================================================")

functionEquation=np.poly1d([1,-1,-1])  #x2-x-1 ,x0=1
x0=1
newtonFunction(functionEquation,x0)

newtonFunction(np.poly1d([1,-7,8,-3]),5)  #x3-7x2+8x-3  ,x0=5