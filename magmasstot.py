#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:01:49 2023
@author: Luca Guglielmi
"""

import numpy as np
import numpy.polynomial.polynomial as pol
import statistics as st
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
import pandas as pd

m_sun = 1.989e33   #solar mass in g

#PARAMETERS OF THE GAUSSIAN SAMPLE
mu = 1.33         #*m_sun, average NS mass in galactic distrib, ozel 2016
#mu = 1.32        # value from kilzilian2013
sigma = 0.09      #*m_sun, std for NS mass in galactic distrib, ozel2016
#sigma = 0.11     # value from kilzilian2013
n_sample = 100000 #number of outcomes that i want to generate
frac_min = 0.14117#min fraction of GRBs included in the magnetar subsample
frac_max = 0.2449 #max fraction of GRBs included in the magnetar subsample
m_ej = 0.         #mass of ejected material


#gaussian z score corresponding to the frac% left tail
z_min= norm.ppf(frac_min)  
z_max= norm.ppf(frac_max)


#gravitational masses, what we measure from data, 
#it's smaller then baryonic mass because it takes into  
#account also the binding energy, negative contribute
m1_g = np.random.normal(mu, sigma, n_sample) #* m_sun
m2_g = np.random.normal(mu,sigma, n_sample) #* m_sun


#baryonic masses, real mass of the star given by the number of baryons*mass of baryons
#they are larger then the mass we measure
m1_b = m1_g  +  0.0602 * m1_g**2  +  0.0180 * m1_g**3
m2_b = m2_g  +  0.0602 * m2_g**2  +  0.0180 * m2_g**3

mu_b=  st.mean(m1_b)  #mean of the values of mtot_g
sigma_b  =  st.stdev(m1_b) #std of the values of mtot_g







#print("The maximum mass we get, considering a fraction of "+str("{0:.2f}".format(frac_a*100))+"%")
import time
start=time.time()
for i in range(22):
    
   
    #baryonic mass of the remnant, 
    mtot_b = m1_b + m2_b - m_ej
    
    mtot_b_mean =  st.mean(mtot_b)  #mean of the values of mtot_b
    mtot_b_std  =  st.stdev(mtot_b) #std of the values of mtot_b
    
    
    
    #HERE I CALCULATE THE GRAVITATIONAL MASS OF THE REMNANT SOLVING A CUBIC EQUATION
    #i do a for loop because i want to act on all the outcomes i generated, mtot_g will be a list
    mtot_g = []
    for i in range(n_sample):
        coeff = [-mtot_b[i], 1, 0.0602, 0.0180] #list of the coefficients of the cubic equation
        poli = pol.Polynomial(coeff)          #the object poli is my cubic equation
        roots = poli.roots()                  #the method roots gives me a list of the three roots of the cubic equation, written as complex numer a + bj 
        mtot_g.append(roots[2].real)          # i take  only the third root from the array because it is in the form a+0j (real solution) and i take only the real part and add it to the list 
        
        
        
    mtot_g = np.array(mtot_g)       #I convert mtot_g in an nparray, to do statistics on it
    mtot_g_mean =  st.mean(mtot_g)  #mean of the values of mtot_g
    mtot_g_std  =  st.stdev(mtot_g) #std of the values of mtot_g
    
    
    #Here i COMPUTE TH X VALUE CORRESPONDING TO THE z VALUE I IMPLEMENTED IN THE BEGINNIG z = x- mean/std
    m_gmax_1 = z_min * mtot_g_std + mtot_g_mean
    m_bmax_1 = z_min * mtot_b_std + mtot_b_mean
    
    m_gmax_2 = z_max * mtot_g_std + mtot_g_mean
    m_bmax_2 = z_max * mtot_b_std + mtot_b_mean
    
    
    #HERE I PLOT THE HISTOGRAM OF VALUES I OBTAINED AND I USE THE METHOD norm TO PLOT A CONTINUOUS GAUSSIAN DISTRIB WITH SAME MEAN AND STD
    
    fig = plt.figure()
    plt.hist(m1_g,bins=100,density=True, alpha=0.8, color='#FF7D40',label = r'Initial distrib, $\mu = $'+str("{0:.2f}".format(mu))+r', $\sigma =$'+str("{0:.2f}".format(sigma)))
    h=plt.hist(mtot_g, bins=100, density=True, alpha=0.6, color='b', label = r'Final distrib, $\mu = $'+str("{0:.2f}".format(mtot_g_mean))+r', $\sigma =$'+str("{0:.2f}".format(mtot_g_std)))
    plt.plot(h[1], norm.pdf(h[1],mtot_g_mean, mtot_g_std),color = 'r', label = r'Gauss, $\mu = $'+str("{0:.2f}".format(mtot_g_mean))+r', $\sigma =$'+str("{0:.2f}".format(mtot_g_std)),linewidth = 2)
    plt.vlines(m_gmax_1,0,3.2, linestyles='dashed', color = '#32CD32',label=r"$M_{g,max}$ = "+str("{0:.2f}".format(m_gmax_1))+r" $M_{\odot}\,(f_{plateau}=$"+str("{0:.2f}".format(frac_min*100))+"%)")
    plt.title(r'Initial and final $M_g$ distribution ($M_{ej}=$'+str("{0:.3f}".format(m_ej))+" $M_{\odot}$)")
    plt.xlabel(r'Initial and final $M_g$ [$M_{\odot}$]')
    plt.ylabel(r'Number (units of $10^3$)')
    plt.ylim([0,4.5])
    plt.legend(loc= 'upper center', fontsize=7)
    
    plt.savefig('./'+'14_'+str("{0:.3f}".format(m_ej))+'ej_magmassg.pdf', dpi = 600)
    print("For threshold at"+str("{0:.2f}".format(frac_min*100))+"% and mejecta ="+str("{0:.3f}".format(m_ej))+" Msun the maximum mass is ", m_gmax_1,"Msun")
    #plt.show()
    
    fig = plt.figure()
    plt.hist(m1_g,bins=100,density=True, alpha=0.8, color='#FF7D40',label = r'Initial distrib, $\mu = $'+str("{0:.2f}".format(mu))+r', $\sigma =$'+str("{0:.2f}".format(sigma)))
    h=plt.hist(mtot_g, bins=100, density=True, alpha=0.6, color='b', label = r'Final distrib, $\mu = $'+str("{0:.2f}".format(mtot_g_mean))+r', $\sigma =$'+str("{0:.2f}".format(mtot_g_std)))
    plt.plot(h[1], norm.pdf(h[1],mtot_g_mean, mtot_g_std),color = 'r', label = r'Gauss, $\mu = $'+str("{0:.2f}".format(mtot_g_mean))+r', $\sigma =$'+str("{0:.2f}".format(mtot_g_std)),linewidth = 2)
    plt.vlines(m_gmax_2,0,3.2, linestyles='dashed', color = '#32CD32',label=r"$M_{g,max}$ = "+str("{0:.2f}".format(m_gmax_2))+r" $M_{\odot}\,(f_{plateau}=$"+str("{0:.2f}".format(frac_max*100))+"%)")
    plt.title(r'Initial and final $M_g$ distribution ($M_{ej}=$'+str("{0:.3f}".format(m_ej))+" $M_{\odot}$)")
    plt.xlabel(r'Initial and final $M_g$ [$M_{\odot}$]')
    plt.ylabel(r'Number (units of $10^3$)')
    plt.ylim([0,4.5])
    plt.legend(loc= 'upper center', fontsize=7)
    
    plt.savefig('./'+'24_'+str("{0:.3f}".format(m_ej))+'ej_magmassg.pdf', dpi = 600)
    print("For threshold at"+str("{0:.2f}".format(frac_max*100))+"% and mejecta ="+str("{0:.3f}".format(m_ej))+" Msun the maximum mass is ", m_gmax_2,"Msun")
    

    m_ej+=0.005
    
end=time.time()
print("time=", end-start)


