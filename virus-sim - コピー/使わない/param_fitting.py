
#include package
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#define differencial equation of seir model
def virus_eq(v,t,d,lamda,beta,delta,p,c):
    
    dT_dt=lamda-d*v[0]-beta*v[0]*v[2]
    dI_dt=beta*v[0]*v[2]-delta*v[1]
    dV_dt=p*v[1]-c*v[2]
    return [dT_dt,dI_dt,dV_dt]

#solve seir model
T0,V0,I0=400000000,0.961,1
ini_state=[T0,I0,V0]
beta,d,lamda,delta,p,c=0.00001157, 0,0, 3.412,0.02009,3.381 #2.493913  , 0.95107715, 1.55007883
t_max=7
dt=0.01
t=np.arange(0,t_max,dt)
plt.plot(t,odeint(virus_eq,ini_state,t,args=(d,lamda,beta,delta,p,c))) #0.0001,1,3

plt.pause(5)
plt.close()

#show observed i
#obs_i=np.loadtxt('fitting.csv')
data_influ=[10**2.0,10**5.5,10**4.0,10**5.5,10**3.0,10**0.4,10**0.1]
data_day = [1,2,3,4,5,6,7]
obs_i = data_influ

plt.plot(obs_i,"o", color="red",label = "data")
plt.legend()
plt.pause(1)
plt.close()

#function which estimate i from seir model func 
def estimate_i(ini_state,d,lamda,beta,delta,p,c):
    v=odeint(virus_eq,ini_state,t,args=(d,lamda,beta,delta,p,c))
    est=v[0:int(t_max/dt):int(1/dt)]
    return est[:,2]
    
#define logscale likelihood function
def y(params):
    est_i=estimate_i(ini_state,params[0],params[1],params[2],params[3],params[4],params[5])
    return np.sum(est_i-obs_i*np.log(np.abs(est_i)))
    
#optimize logscale likelihood function
mnmz=minimize(y,[d,lamda,beta,delta,p,c],method="nelder-mead")
print(mnmz)
#R0
#N_total = S_0+I_0+R_0
#R0 = N_total*beta_const *(1/gamma_const)
#beta_const,lp,gamma_const = mnmz.x[0],mnmz.x[1],mnmz.x[2] #感染率、感染待時間、除去率（回復率）
#print(beta_const,lp,gamma_const)
#R0 = beta_const*(1/gamma_const)
#print(R0)

#plot reult with observed data
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(1.6180 * 4, 4*2))
lns1=ax1.plot(obs_i,"o", color="red",label = "data")
lns2=ax1.plot(estimate_i(ini_state,mnmz.x[0],mnmz.x[1],mnmz.x[2],mnmz.x[3],mnmz.x[4],mnmz.x[5]), label = "estimation")
lns_ax1 = lns1+lns2
labs_ax1 = [l.get_label() for l in lns_ax1]
ax1.legend(lns_ax1, labs_ax1, loc=0)

lns3=ax2.plot(obs_i,"o", color="red",label = "data")
lns4=ax2.plot(t,odeint(virus_eq,ini_state,t,args=(mnmz.x[0],mnmz.x[1],mnmz.x[2],mnmz.x[3],mnmz.x[4],mnmz.x[5])))
ax2.legend(['data','Susceptible','Exposed','Infected','Recovered'], loc=0)
#ax2.set_title('SEIR_b{:.2f}_ip{:.2f}_gamma{:.2f}_N{:d}_E0{:d}_I0{:d}_R0{:.2f}'.format(beta_const,lp,gamma_const,N,E0,I0,R0))
#plt.savefig('./fig/SEIR_b{:.2f}_ip{:.2f}_gamma{:.2f}_N{:d}_E0{:d}_I0{:d}_R0{:.2f}_.png'.format(beta_const,lp,gamma_const,N,E0,I0,R0)) 
plt.show()
plt.close()