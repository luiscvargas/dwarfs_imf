import sys
import os 
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import interpolate
import pandas as pd
from mywrangle import *
from myanalysis import *

#now specify a Salpeter LF, alpha is exponent in linear eqn, alpha = Gamma + 1

#Set env variables for latex-style plotting
if len(sys.argv) != 1: sys.exit()
rc('text', usetex=True)
rc('font', family='serif')

#The data to be fit is described by one photometric system, and two photometric bands. 
#By convention, mag2 = mag, and color = mag1 - mag2
system = 'acs'
sysmag1   = 'F606W'
sysmag2   = 'F814W'

isoage = 14.0
isofeh = -2.5
isoafe = 0.4
dmod0 = 20.63  #dmod to Hercules
nstars = 10000
mass_min = 0.20
mass_max = 0.80

#Define a dummy magnitude, magnitude error array
#Later: Will import actual observed data and create a magnitude-magnitude error relation instead.

magarr1 = np.arange(22.,30.,.01)  ; magarr2 = magarr1.copy()

if 0:
    magerrarr = magarr.copy()
    magerrarr[magarr < 22] = 0.005
    magerrarr[(magarr >= 22) & (magarr < 24)] = 0.01
    magerrarr[(magarr >= 24) & (magarr < 26)] = 0.02
    magerrarr[(magarr >= 26) & (magarr < 28)] = 0.04
    magerrarr[magarr >= 28] = 0.06
    magerrarr1 = magerrarr.copy()
    magerrarr2 = magerrarr.copy()
    plt.plot(magarr,magerrarr,ms=3,color='red')
    plt.show()
else:
    phot = read_phot('Herc',system,sysmag1,sysmag2)
    #raise SystemExit
    magsort1 = np.argsort(phot['F606W'])
    magsort2 = np.argsort(phot['F814W'])
    p1 = np.polyfit(phot['F606W'][magsort1],phot['F606Werr'][magsort1],4,cov=False)
    p2 = np.polyfit(phot['F814W'][magsort2],phot['F814Werr'][magsort2],4,cov=False)
    magerrarr1 = np.polyval(p1,magarr1)
    magerrarr1[magarr1 <= phot['F606W'].min()] = phot['F606Werr'].min()
    magerrarr2 = np.polyval(p2,magarr2)
    magerrarr2[magarr2 <= phot['F814W'].min()] = phot['F814Werr'].min()
    plt.scatter(phot['F606W'],phot['F606Werr'],s=2,color='blue',marker='^')
    plt.scatter(phot['F814W'],phot['F814Werr'],s=2,color='red',marker='^')
    plt.plot(magarr1,magerrarr1,ms=3,color='green',lw=2.5,ls='-.')
    plt.plot(magarr2,magerrarr2,ms=3,color='magenta',lw=2.5,ls='--')
    plt.xlabel(r"mag")
    plt.ylabel(r"$\sigma$(mag)")
    plt.show()

data = simulate_cmd(nstars,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,system,sysmag1,sysmag2,imftype='salpeter',
   alpha=2.35,mass_max=mass_max)


#data = simulate_cmd(nstars,age,feh,afe,dmod,magarr,magerrarr,system,imftype='chabrier',mc=0.4,sigmac=0.2,mass_min=0.05,mass_max=0.80)

#Determine representative errors for bins in magnitude

iso = read_iso_darth(isoage,isofeh,isoafe,system,mass_min=mass_min,mass_max=mass_max)

isomass = iso['mass'] 
isocol = iso[sysmag1] - iso[sysmag2] 
isomag = iso[sysmag2] + dmod0

plt.plot(isocol,isomag,ls='-',color='red',lw=2)
plt.xlabel(r"$F606W-F814W$")
plt.ylabel(r"$F814W$")
plt.scatter(data['color'],data[sysmag2],marker='o',s=3,color='b')
plt.axis([isocol.min()-.25,isocol.max()+.25,dmod0+12,dmod0-2])
plt.show()

if 0:

   plt.plot(isocol0,isomag0,lw=1,ls='-')
   plt.plot(isocol,isomag,lw=3,ls='--')
   plt.ylabel(mag_name)
   plt.xlabel(col_name)
   plt.scatter(phot_raw['col'],phot_raw['mag'],color='k',marker='.',s=1)
   plt.scatter(phot['col'],phot['mag'],color='r',marker='o',s=2)
   plt.show()

