# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:58:06 2014

@author: nicolas
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:47:07 2014

@author: nicolas

y'(t) = f(y(t),t))
y(a)=y0

"""
import math
import time
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt

def f(y,t): # cas de l'equation y'=y, qui a pour solution, x->Cexp(x)
	return y


def euler1(f,a,b,y0,h):
    """ Resolution de y'(t)=f(y(t),t) avec y(a)=y0.
    Entree : f une fonction, a,b deux reels tels que a<b, h un reel >0 : le pas, y0 un reel.
    Sortie : une valeur approchee de y(b) un reel ."""
    n=int(math.floor((b-a)/h)) # on calcule le nombre de subdivisions
    h=(b-a)/n			  # on ajuste au besoin le pas.
    t=a
    y=y0
    for k in range(1,n+1):
        y=y+h*f(y,t)
        t=t+h
    return y
    
def rk4(f,a,b,y0,h):
    """ methode de Runge Kutta. Retourne une liste pour t et une pour y."""
    n=int((b-a)/h)
    h=(b-a)/n
    t=a
    y=y0
    k=0
    while k<n:
        k1=h*f(y,t)
        k2=h*f(y+k1/2.0,t+h/2.0)
        k3=h*f(y+k2/2.0,t+h/2.0)
        k4=h*f(y+k3,t+h)
        y=y+(k1+2*k2+2*k3+k4)/6.0
        t=t+h
        k=k+1
    return y

        
def erreur(u):
    return abs((u-np.exp(1))/np.exp(1))
    
les_pas=[10**-k for k in range(7)]

exp1_euler=[]
err_euler=[]
temps_euler=[]

exp1_rk4=[]
err_rk4=[]
temps_rk4=[]

exp1_sc=[]
err_sc=[]
temps_sc=[]

exp1_scplus=[]
err_scplus=[]
temps_scplus=[]


for h in les_pas:
    t1=time.time()
    exp1_euler.append(euler1(f,0,1,1,h))
    err_euler.append(erreur(exp1_euler[-1]))
    t2=time.time()
    temps_euler.append(t2-t1)

for h in les_pas:
    t1=time.time()
    exp1_rk4.append(rk4(f,0,1,1,h))
    err_rk4.append(erreur(exp1_rk4[-1]))
    t2=time.time()
    temps_rk4.append(t2-t1)

for h in les_pas:
    t1=time.time()
    les_t=np.linspace(0,1,int(math.floor(1/h)))
    les_y_si=si.odeint(f,1.0,les_t)
    exp1_sc.append(les_y_si[-1])
    err_sc.append(erreur(exp1_sc[-1]))
    t2=time.time()
    temps_sc.append(t2-t1)

for h in les_pas:
    t1=time.time()
    les_t=np.linspace(0,1,int(math.floor(1/h)))
    les_y_siplus=si.odeint(f,1.0,les_t,rtol=1e-13,atol=1e-13)
    #les_y_si=si.odeint(f,1.0,les_t)
    exp1_scplus.append(les_y_siplus[-1])
    err_scplus.append(erreur(exp1_scplus[-1]))
    t2=time.time()
    temps_scplus.append(t2-t1)
    
print( temps_euler,err_euler)
print( temps_rk4,err_rk4)
print( temps_sc,err_sc)
print( temps_scplus,err_scplus)

plt.figure(3)
plt.clf()
plt.loglog()
plt.plot(les_pas,err_euler,'og')
plt.plot(les_pas,err_euler,'--g')
plt.plot(les_pas,err_rk4,'ob')
plt.plot(les_pas,err_rk4,'--b')
plt.plot(les_pas,err_sc,'oc')
plt.plot(les_pas,err_sc,'--c')
plt.plot(les_pas,err_scplus,'or')
plt.plot(les_pas,err_scplus,'--r')
plt.title('erreur en fonction du pas')
plt.xlabel('pas')
plt.ylabel('erreur')
plt.legend(['Euler','','RK4','','odeint','','odeint+',''],loc='best')
plt.grid()
plt.show()