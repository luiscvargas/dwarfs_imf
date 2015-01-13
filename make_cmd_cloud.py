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

class DartmouthIsochrone(object):

    def __init__(self,feh,afe,age,system,band1,band2):
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
        #Assign descriptive variables to object
        self.age = age
        self.afe = afe
        self.feh = feh
        self.system = system
        self.mass_min = self.iso['mass'].min()
        self.mass_max = self.iso['mass'].max()
        self.interp_flag = 0

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
    
    def print_params(self): 
        print "==Dartmouth Isochrone=="
        print "Age    = {0:.1f} Gyr".format(self.age)
        print "[Fe/H] = {0:.2f}".format(self.feh)
        print "[a/Fe] = {0:.2f}".format(self.afe)
        print "M_min  = {0:.3f} Msun".format(self.mass_min)
        print "M_max  = {0:.3f} Msun".format(self.mass_max)
		

    #Future: interpolate isochrone if it does not satisfy the condition
    #of being finely graded in mass or magnitude - need dM intervals to be small
    #relative to the range of mass considered otherwise dN_*(dM) is not accurate.

    def interpolate(self,dm):
        #create a np struct array of same type as original, sort by mass, and interpolate
        isonew = self.iso[0]
        colnames = self.iso.dtype.names  
        isort = np.argsort(self.iso['mass']) 
		
        #assign size of interpolated array given mass bounds and dm
        massarr = np.arange(self.mass_min,self.mass_max,dm)
        isonew = np.repeat(isonew,len(massarr)) #(aside: np.repeat or np.tile work equally well here)
        isonew[:]['mass'] = massarr
        isonew[:]['idx']  = np.arange(len(massarr))
        isonew[:]['feh']  = self.iso[0]['feh']
        isonew[:]['afe']  = self.iso[0]['afe']
        isonew[:]['age']  = self.iso[0]['afe']

        for icol,colname in enumerate(colnames):

            if (colname != 'idx' and colname != 'feh' and colname != 'afe' and colname != 'age' and
                colname != 'mass'):
            
                #For each magnitude - mass relation, interpolate
                f = interpolate.splrep(self.iso['mass'][isort],self.iso[colname][isort])
                magarr = interpolate.splev(massarr,f)

                #plt.plot(xarr,yarr,lw=4,color='blue')
                #plt.plot(isodata['mass'],isodata[colname],lw=1,color='red')
                #plt.show()

                isonew[:][colname] = yarr

        #Reassign self.iso using new interpolated array
		self.iso = isonew
        self.interp_flag = 1

    def is_interp(self):
        if self.interp_flag == 0:
            print "No interpolation done on file"
        else:
            print "Already interpolated"

myiso = DartmouthIsochrone(-2.0,0.4,14.0,'wfc3','F110W','F160W')
#myiso.change_min_mass(.30)
#myiso.change_max_mass(.75)
myiso.

