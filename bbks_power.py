import numpy as np 

from scipy.integrate import quad

import math

### Define parameters for calculations

H_0 = 67.74
Omega_m = 0.3075
Omega_b = 0.0486
Omega_l = 1.0-Omega_m-Omega_b
ns = 0.9667

n = 1000

Omega_o = Omega_m+Omega_b

h = H_0/100.0

delta_h = 1.91*(10**(-5))

keq = 0.01

c = 299792.0

parameters = [H_0,Omega_m,Omega_l,Omega_b,ns,n,Omega_o,h,delta_h,c,keq]

np.save("cosmo_parameters.npy",parameters)

### Define and save wavenumbers

k_values = (np.logspace(-5,np.log10(1),n))

np.save("k_values.npy",k_values)

### Calculate BBKS transfer function

def bbks_transfer(x_values):

	bbks = ((np.log(1.0+2.34*x_values))/(2.34*x_values))*np.power(1.0+3.89*x_values+(16.1*x_values)**2+(5.46*x_values)**3+(6.71*x_values)**4,-0.25)

	return bbks

### Calculate the power spectrum

def bbks_power(k,delta_h,c,H_0,ns,keq,h,Omega_o,Omega_b):

	L = (Omega_o*h)*np.exp(-Omega_b-np.sqrt(2.0*h)*(Omega_b/Omega_o))

	P = (2.0*np.pi**2)*(delta_h**2)*(np.power(c/H_0,3.0+ns)*np.power(k,ns))*(bbks_transfer(k/L)**2)

	return P