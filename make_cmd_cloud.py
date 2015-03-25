#!/usr/bin/python

'''
Created: 2015-01-08
Purpose: Simulate a CMD and generate semi-empirical LF cloud
Inputs : Binary fraction
'''

import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mywrangle import *
from myanalysis import *
from my_em import *
from scipy import interpolate
from copy import deepcopy

class DartmouthIsochrone(object):

    def __init__(self,feh,afe,age,system):
        try:
            f = open(os.getenv('ASTRO_DIR')+'/dwarfs_imf/iso/'+'dartmouth_'+system+'.obj','rb')
        except: 
            raise ValueError("Isochrone library for "+system+" not found!")
        data = pickle.load(f)
        f.close()
        #Check for existence of input parameters in isochrone library
        if age not in np.unique(data['age']): 
            raise ValueError("Age not in isochrone library.")
        if feh not in np.unique(data['feh']): 
            raise ValueError("[Fe/H] not in isochrone library.")
        if afe not in np.unique(data['afe']): 
            raise ValueError("[a/Fe] not in isochrone library.")
        #Select particular isochrone for specified [Fe/H], [a/Fe], and age. 
        self.iso = data[(data['age'] == age) & (data['feh'] == feh) &
                      (data['afe'] == afe)] 
        self.data = np.zeros(0) #empty placeholder array for interpolated array
        #Assign descriptive variables to object
        self.age = age
        self.afe = afe
        self.feh = feh
        self.system = system
        self.mass_min = self.iso['mass'].min()
        self.mass_max = self.iso['mass'].max()
        self.interp_flag = 0

    def print_params(self): 
        print "==Dartmouth Isochrone=="
        print "Age    = {0:.1f} Gyr".format(self.age)
        print "[Fe/H] = {0:.2f}".format(self.feh)
        print "[a/Fe] = {0:.2f}".format(self.afe)
        print "M_min  = {0:.3f} Msun".format(self.mass_min)
        print "M_max  = {0:.3f} Msun".format(self.mass_max)

    def change_min_mass(self,mass_min):
        #If mass_min < min mass available in isochrone, set to min mass available
        if mass_min < self.iso['mass'].min(): 
            mass_min = self.iso['mass'].min()
        self.mass_min = mass_min

    def change_max_mass(self,mass_max):
        #If mass_max > max mass available in isochrone, set to max mass available
        if mass_max > self.iso['mass'].max(): 
            mass_max = self.iso['mass'].max()
        self.mass_max = mass_max
    
    #Future: interpolate isochrone if it does not satisfy the condition
    #of being finely graded in mass or magnitude - need dM intervals to be small
    #relative to the range of mass considered otherwise dN_*(dM) is not accurate.

    def interpolate(self,dm=0.001,diagnose=False):
        #create a np struct array of same type as original.
        #First, sort rows by mass, and interpolate
        isonew = np.copy(self.iso)  # dont use isonew=self.iso!
        isonew = isonew[0]
        isort = np.argsort(self.iso['mass']) 

        npts = long((self.mass_max - self.mass_min) / dm)

        #assign size of interpolated array given mass bounds and dm
        massarr = np.linspace(self.mass_min,self.mass_max,npts)
        
        #check that interpolation would result in more data points that
        #original array, else interpolate.splrep fails. 
        if len(massarr) <= len(self.iso['mass']): 
            print "No interpolation done; returning..."
            return None
        else: 
            print "Proceed to interpolate based on mass..."

        isonew = np.repeat(isonew,len(massarr)) #(aside: np.repeat or np.tile work equally well here)
        isonew['mass'] = massarr
        isonew['idx']  = np.arange(len(massarr))
        isonew['feh']  = self.iso['feh'][0]
        isonew['afe']  = self.iso['afe'][0]
        isonew['age']  = self.iso['age'][0]

        colnames = self.iso.dtype.names 

        colnames2 = colnames[colnames != 'idx' and colnames != 'feh' and 
                colnames != 'afe' and colnames != 'age' and colnames != 'mass']

        print colnames2  #tuple is immutable!

        for icol,colname in enumerate(colnames):

            if (colname != 'idx' and colname != 'feh' and colname != 'afe' and colname != 'age' and
                colname != 'mass'):
            
                #For each magnitude - mass relation, interpolate
                xx = self.iso['mass'][isort]
                yy = self.iso[colname][isort]
                f = interpolate.splrep(xx,yy)
                magarr = interpolate.splev(massarr,f)

                #plt.plot(massarr,magarr,lw=4,color='blue')
                #plt.plot(self.iso['mass'],self.iso[colname],lw=1,color='red')
                #plt.show()

                isonew[colname] = magarr
            else:
                pass

        #Reassign self.iso using new interpolated array
        self.data = isonew
        self.interp_flag = 1

        if diagnose == True:
            plt.plot(self.iso['F110W'],self.iso['F160W'],'r-',lw=3)
            plt.plot(self.data['F110W'],self.data['F160W'],'b--',lw=1)
            plt.show()

    def has_interp(self):
        if self.interp_flag == 0:
            print "No interpolation done on file"
        else:
            print "Interpolated data located as self.data"
            print "Non-interpolated data located as self.iso"

#Generate an isochrone object and interpolate to a uniformly-spaced mass array.

myiso = DartmouthIsochrone(-2.0,0.4,14.0,'wfc3',)
myiso.interpolate(dm=0.001,diagnose=False)
myiso.has_interp()

strmag1 = 'F110W'
strmag2 = 'F160W'

plt.plot(myiso.data[strmag1]-myiso.data[strmag2],myiso.data[strmag2],lw=2)
plt.axis([-1,0,15,0])
plt.xlabel(strmag1 + '-' + strmag2)
plt.ylabel(strmag2)
plt.show()

#Proceed to draw stars from distribution. 
#Will use rejection sampling as it not clear how to use inverse transformation
#when binaries are included, except when 

#binary approach 1: 
    # dN/dM: both primary and secondary drawn from full IMF distribution, then matched randomly
    # binary fraction, q binary systems: ntot = (1-q)*nsys + 2q*nsys, fraction of stars in binaries = 
    # 2q*nsys / ntot = 2q*nsys / (1+q)*nsys , n single stars = ntot * (1-q)*nsys/(1+q)*nsys = (1-q)/(1+q)
    # 1 alt: change boundaries on Mmin, and Mmax.

mass = myiso.data['mass']

print min(mass),max(mass)

alpha = 2.35
fs = f_salpeter(mass,alpha)
fk = f_kroupa(mass,1.35,1.7,alpha_3=2.30)
Phi_s = np.cumsum(fs)
Phi_s = Phi_s / max(Phi_s)
Phi_k = np.cumsum(fk)
Phi_k = Phi_k / max(Phi_k)

#do inverse transform sampling

plt.plot(Phi_s,mass,color='red',label='PL')
plt.plot(Phi_k,mass,color='blue',label='BkPL')
plt.axis([0,1,0,0.9])
plt.legend()
plt.show()




