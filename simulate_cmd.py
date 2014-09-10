import sys
import os 
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import interpolate
import pandas as pd
from mywrangle import *

#now specify a Salpeter LF, alpha is exponent in linear eqn, alpha = Gamma + 1

def f_salpeter(mass_arr,mass_min,mass_max,alpha):
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)  #to end sets last element to 0 otherwise
       #one element too few.
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] #update last element
    dmass_arr = abs(dmass_arr)
    dN_arr = (mass_arr**(-1.*alpha)) * dmass_arr
    dN_arr[(mass_arr < mass_min) & (mass_arr > mass_max)] = 0.0
    return dN_arr

#specify a Chabrier LF, but given in dN/dM. The Chabrier IMF is given typically as dN/d(logM)
#dN/dM = (1/ln10)*(1/M)*dN/dlogM, and this is calculated within the function. Finally, return
#dM, as for f_salpeter .
#Careful: denominator in first term has ln10 = np.log(10), but exponential is log10 M, so np.log10(m)
def f_chabrier(mass_arr,mass_min,mass_max,mass_crit,sigma_mass_crit):
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)  
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] 
    dmass_arr = abs(dmass_arr)
    dN_arr = ((1./(np.log(10.)*mass_arr)) * (1./(np.sqrt(2.*np.pi)*sigma_mass_crit)) * 
        np.exp(-1. * (np.log10(mass_arr)-np.log10(mass_crit))**2 / (2. * sigma_mass_crit**2)) * 
        dmass_arr)
    dN_arr[(mass_arr < mass_min) & (mass_arr > mass_max)] = 0.0
    return dN_arr
    
def likelihood_matrix(cmd_point,iso_point,error_cov):
    """Perform calculations as ndarrays and not as matrices;  have
    checked that the behavior and cpu usage is the same"""
    diff = cmd_point - iso_point
    arg = -0.5*(np.dot(diff,np.dot(np.linalg.inv(error_cov),diff)))
    #print diff,arg
    return arg

def likelihood(sigma_r,sigma_gr,cov_gr_r,delta_gr_arr,delta_r_arr):
    #arrays must be ndarr, not python lists
    det_sigma_matrix = sigma_r*sigma_r*sigma_gr*sigma_gr - cov_gr_r*cov_gr_r
    det_sigma_matrix_inv = 1.0 / det_sigma_matrix
    P = 1.0/(2.0*np.pi*np.sqrt(det_sigma_matrix))
    exp_arg = np.exp(-0.5*(det_sigma_matrix_inv)*
          (sigma_r**2*delta_gr_arr**2 - 
           2.0*cov_gr_r*delta_gr_arr*delta_r_arr + 
           sigma_gr**2*delta_r_arr**2))
    #print P*exp_arg
    return P*exp_arg

def simulate_cmd(nstars,isoage,isofeh,isoafe,dist_mod,inmagarr,inmagerrarr,system,sysmag1,sysmag2,**kwargs):

   testing = 0

   if 'imftype' not in kwargs.keys(): raise SystemExit

   print "Warning! This program will generate synthetic CMDs"
   print "only for the MS region where the MF = IMF."""

   #raw_input("Press any key to continue>>>")

   if 'mass_min' in kwargs.keys():
       mass_min = kwargs['mass_min']
   else: mass_min = 0.05

   if 'mass_max' in kwargs.keys():
       mass_max = kwargs['mass_max']
   else: mass_max = 0.80

   if mass_max <= mass_min: raise SystemExit

   #Now import isochrone of given age, [Fe/H], [a/Fe], making desired mass cuts for fitting
   iso = read_iso_darth(isoage,isofeh,isoafe,system,mass_min=mass_min,mass_max=mass_max)

   isomass = iso['mass'] 
   isocol = iso[sysmag1] - iso[sysmag2] 
   isomag = iso[sysmag2] + dist_mod

   if system == 'wfpc2':
       col_name = r'$m_{606w} - m_{814w}$' ; mag_name = r'$m_{814w}$'
   elif system == 'sdss':
       col_name = r'$(g - r)_0$' ; mag_name = r'$r_0$'
   elif system == 'acs':
       col_name = r'$m_{606w} - m_{814w}$' ; mag_name = r'$m_{814w}$'
   else:
       pass

   if kwargs['imftype'] == 'salpeter':
       if 'alpha' not in kwargs.keys():
           print "Error: alpha not specified for Salpeter function" 
           raise SystemExit
       alpha_ = kwargs['alpha']

   elif kwargs['imftype'] == 'chabrier':
       if 'mc' not in kwargs.keys():
           print "Error: M_c (kwarg=mc) not specified for Chabrier function" 
           raise SystemExit
       elif 'sigmac' not in kwargs.keys():
           print "Error: sigma_c (kwarg=sigmac) not specified for Chabrier function" 
           raise SystemExit
       pass
       mc_ = kwargs['mc']
       sigmac_ = kwargs['sigmac']

   else:
       print "Need to specify either Salpeter or Chabrier and their respective params!"
       raise SystemExit
       

   #Find min and max in dN/dM in order to set limits in y-axis for random number
   #generator

   xdum = np.arange(mass_min,mass_max,0.0001)
   if kwargs['imftype'] == 'salpeter':
       ydum = f_salpeter(xdum,mass_min,mass_max,alpha_)
   elif kwargs['imftype'] == 'chabrier':
       ydum = f_chabrier(xdum,mass_min,mass_max,mc_,sigmac_)

   if testing == 1:
       plt.subplot(1,2,1)
       plt.plot(xdum,ydum,color='b',ls='-')
       plt.axis([mass_min,mass_max,ydum.min(),ydum.max()])
       plt.xlabel(r"$M$") ; plt.ylabel(r"$dN/dM$")
       plt.subplot(1,2,2)
       plt.loglog(xdum,ydum,color='b',ls='-',basex=10,basey=10)
       plt.axis([mass_min,mass_max,ydum.min(),ydum.max()])
       plt.xlabel(r"log\,$M$") ; plt.ylabel(r"log\,$dN/dM$")
       #plt.savefig(os.getenv('HOME')+'/Desktop/mass_dndm_function.png',bbox_inches='tight')
       plt.show()

   #Generate random data points in 2D and stop when 100 points are within 
   #desired region

   #First generate very large array in x and y, so that hopefully there will be at least nstars
   #that fall under dN/dM - M relation.
   np.random.seed(seed=12345)

   #Define limits of masses as lowest isochrone mass and highest isochrone mass *within* the mass cuts
   #specified as input args. If I use instead those input args directly, some points will be slightly
   #outside of range of isochrone masses, as mass_min < M_iso < mass_max is set in read_darth_iso, 
   #causing the spline interpolator to crash later on.
   xrantmparr = np.random.random_sample(nstars*200) * (iso['mass'].max() - iso['mass'].min()) + iso['mass'].min()
   yrantmparr = np.random.random_sample(nstars*200) * 1.05*(ydum.max() - ydum.min()) + ydum.min()

   xranarr = np.arange(nstars)*0.0
   yranarr = np.arange(nstars)*0.0

   #Now find the pairs (x,y) of simulated data that fall under envelope of dN/dM - M relation.

   count = 0
   for i,xrantmp in enumerate(xrantmparr):
       if count == nstars: break
       idx = np.abs(xdum - xrantmp).argmin()
       if (yrantmparr[i] <= ydum[idx]) & (xdum[idx] > iso['mass'].min()) & (xdum[idx] < iso['mass'].max()):
           xranarr[count] = xrantmparr[i]
           yranarr[count] = yrantmparr[i]
           count += 1
       else:
           pass

   if len(yranarr[yranarr > 0.0]) < nstars:
       print "Need to generate more samples!"
       raise SystemExit

   if testing == 1:
       plt.scatter(xrantmparr,yrantmparr,s=1,c='k',marker='.')
       plt.scatter(xranarr,yranarr,s=6,c='r',marker='o')
       plt.plot(xdum,ydum,color='b',ls='-')
       plt.axis([mass_min,mass_max,ydum.min(),ydum.max()])
       plt.xlabel(r"$M$") ; plt.ylabel(r"$dN/dM$")
       plt.show()

   #Interpolate isochrone magnitude-mass relation
   isort = np.argsort(iso['mass'])  #! argsort = returns indices for sorted array, sort=returns sorted array
   if testing == 1:
       plt.plot(iso['mass'][isort],iso[sysmag2][isort]+dist_mod,'b.',ls='--')
       plt.show()
   f1 = interpolate.splrep(iso['mass'][isort],iso[sysmag1][isort]+dist_mod)
   f2 = interpolate.splrep(iso['mass'][isort],iso[sysmag2][isort]+dist_mod)

   #Assign magnitudes to each star based on their mass and the mass-magnitude relation calculated above.
   mag1ranarr_0 = interpolate.splev(xranarr,f1)
   mag2ranarr_0 = interpolate.splev(xranarr,f2)  #band 2 = for system=wfpc2
   colorranarr_0  = mag1ranarr_0 - mag2ranarr_0

   #Initialize data magnitude arrays which will include photometric uncertainties.
   mag1ranarr = np.arange(len(mag1ranarr_0))*0.0
   mag2ranarr = np.arange(len(mag1ranarr_0))*0.0

   #Based on mag-errmag relation from input args, assign random Gaussian deviates to each "star".
   for i,mag in enumerate(mag1ranarr_0):
       idx = np.abs(mag - inmagarr).argmin()
       mag1ranarr[i] = mag + inmagerrarr[idx]*np.random.normal()
   for i,mag in enumerate(mag2ranarr_0):
       idx = np.abs(mag - inmagarr).argmin()
       mag2ranarr[i] = mag + inmagerrarr[idx]*np.random.normal()

   colorranarr = mag1ranarr - mag2ranarr
    
   if testing < 10:   
       plt.plot(isocol,isomag,ls='-',color='red',lw=2)
       plt.xlabel(r"$F606W-F814W$")
       plt.ylabel(r"$F814W$")
       plt.scatter(colorranarr,mag2ranarr,marker='o',s=3,color='b')
       plt.axis([isocol.min()-.25,isocol.max()+.25,dist_mod+12,dist_mod-2])
       plt.show()

   raise SystemExit

   #For each data point, find corresponding magnitudes for the mass. 

   #Given mags and assumed magnitude error characteristics (e.g., error = f(mag)),
   #add noise to each magnitude value. 
  
   #Output data as pd dataframe analogous with that output by read_phot() - ie
   #so that it can be passed to main.py just as real data. 


#Set env variables for latex-style plotting
if len(sys.argv) != 1: sys.exit()
rc('text', usetex=True)
rc('font', family='serif')

#The data to be fit is described by one photometric system, and two photometric bands. 
#By convention, mag2 = mag, and color = mag1 - mag2
system = 'acs'
sysmag1   = 'F606W'
sysmag2   = 'F814W'

age = 14.0
feh = -2.5
afe = 0.4
dmod = 20.63  #dmod to Hercules
nstars = 10000

#Define a dummy magnitude, magnitude error array
#Later: Will import actual observed data and create a magnitude-magnitude error relation instead.

magarr = np.arange(22.,30.,.01)  
if 0:
    magerrarr = magarr.copy()
    magerrarr[magarr < 22] = 0.005
    magerrarr[(magarr >= 22) & (magarr < 24)] = 0.01
    magerrarr[(magarr >= 24) & (magarr < 26)] = 0.02
    magerrarr[(magarr >= 26) & (magarr < 28)] = 0.04
    magerrarr[magarr >= 28] = 0.06
    plt.plot(magarr,magerrarr,ms=3,color='red')
    plt.show()
else:
    phot = read_phot('Herc',dataset=system,cuts=True)
    magsort = np.argsort(phot['m606'])
    p = np.polyfit(phot['m606'][magsort],phot['m606err'][magsort],4,cov=False)
    magerrarr = np.polyval(p,magarr)
    magerrarr[magarr <= phot['m606'].min()] = phot['m606err'].min()
    plt.scatter(phot['m606'],phot['m606err'],s=2,color='blue',marker='^')
    plt.plot(magarr,magerrarr,ms=3,color='red',lw=2.5)
    plt.xlabel(r"mag")
    plt.ylabel(r"$\sigma$(mag)")
    plt.show()

data = simulate_cmd(nstars,age,feh,afe,dmod,magarr,magerrarr,system,sysmag1,sysmag2,imftype='salpeter',alpha=2.35,mass_min=0.20,mass_max=0.80)

#data = simulate_cmd(nstars,age,feh,afe,dmod,magarr,magerrarr,system,imftype='chabrier',mc=0.4,sigmac=0.2,mass_min=0.05,mass_max=0.80)

#Determine representative errors for bins in magnitude

if 0:
   plt.plot(isocol0,isomag0,lw=1,ls='-')
   plt.plot(isocol,isomag,lw=3,ls='--')
   plt.ylabel(mag_name)
   plt.xlabel(col_name)
   if system == 'wfpc2': plt.axis([-1.25,0.75,10.+dmod0,0+dmod0])
   if system == 'sdss': plt.axis([-1.25,0.75,6.+dmod0,-2+dmod0])
   if system == 'wfpc2': plt.errorbar(-0.9+0.0*magerrmean,magbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=2.0)
   if system == 'sdss': plt.errorbar(-0.2+0.0*magerrmean,magbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=2.0)
   plt.scatter(phot_raw['col'],phot_raw['mag'],color='k',marker='.',s=1)
   plt.scatter(phot['col'],phot['mag'],color='r',marker='o',s=2)
   #plt.savefig(os.getenv('HOME')+'/Desktop/fitting_data.png',bbox_inches='tight')
   plt.show()

