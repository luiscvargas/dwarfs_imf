import sys
import os 
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
import pandas as pd
import timeit
from mywrangle import *
from myanalysis import *

#Set env variables for latex-style plotting
rc('text', usetex=True)
rc('font', family='serif')

sim = 1

if sim == 1:

    titlestring = 'Simulated'

    system = 'acs'
    sysmag1   = 'F606W'
    sysmag2   = 'F814W'

    isoage = 14.0
    isofeh = -2.5
    isoafe =  0.4
    dmod0  = 20.63  #dmod to Hercules
    nstars = 4500
    mass_min = 0.20
    mass_max = 0.80

    #Use Herc data to model error as a function of magnitude
    phot = read_phot('Herc',system,sysmag1,sysmag2,cuts=True)
    magsort1 = np.argsort(phot['F606W'])
    magsort2 = np.argsort(phot['F814W'])
    p1 = np.polyfit(phot['F606W'][magsort1],phot['F606Werr'][magsort1],4,cov=False)
    p2 = np.polyfit(phot['F814W'][magsort2],phot['F814Werr'][magsort2],4,cov=False)
    magarr1 = np.arange(22.,30.,.01)  
    magarr2 = np.copy(magarr1)
    magerrarr1 = np.polyval(p1,magarr1)
    magerrarr1[magarr1 <= phot['F606W'].min()] = phot['F606Werr'].min()
    magerrarr2 = np.polyval(p2,magarr2)
    magerrarr2[magarr2 <= phot['F814W'].min()] = phot['F814Werr'].min()
    phot_true = np.copy(phot)
    del phot

    #Create simulated data. simulate_cmd module called from myanalysis.py
    #For test version, see test_simulate_cmd.py
    phot = simulate_cmd(nstars,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
    system,sysmag1,sysmag2,imftype='salpeter',alpha=2.35,mass_min=mass_min,mass_max=mass_max)

    phot_raw = np.copy(phot)

elif sim == 0:

    if len(sys.argv) != 2: sys.exit()

    system = 'acs'
    sysmag1   = 'F606W'
    sysmag2   = 'F814W'

    isoage = 14.0
    isofeh = -2.5
    isoafe =  0.4
    dmod0  = 20.63  #dmod to Hercules
    #nstars = 10000  #only for simulated data
    mass_min = 0.05
    mass_max = 0.75  #mass max must be below MSTO - make plot to check for this?

    dsph_select = str(sys.argv[1])
    titlestring = dsph_select

    #Read-in MW dwarf spheroidal data, e.g., Mv, distance modulus, velocity dispersion
    #The data comes from a data table I try to maintain updated with high quality data 
    #for each quantity.
    dsphs = read_dsph_data()

    dmod0  = dsphs.loc[dsph_select,'dmod0']  
    ra_dwarf  = dsphs.loc[dsph_select,'ra'] 
    dec_dwarf  = dsphs.loc[dsph_select,'dec']  
    rhalf_dwarf  = dsphs.loc[dsph_select,'r_h']  
    print '=============================='
    print 'The distance modulus is {0:4f}.'.format(dmod0)
    print 'The central ra is {0:4f} deg.'.format(ra_dwarf)
    print 'The central decl is {0:4f} deg.'.format(dec_dwarf)
    print 'The half-light radius is {0:4f} arcmin.'.format(rhalf_dwarf)
    print '=============================='

    #Read in photometry database and extract relevant quantities
    phot = read_phot('Herc',system,sysmag1,sysmag2,cuts=True)
    phot_raw = read_phot('Herc',system,sysmag1,sysmag2,cuts=False)
    #phot = read_phot(dsph_select,dataset=system,cuts=True)
    #phot_raw = read_phot(dsph_select,dataset=system,cuts=False)

#Find representative errors for bins in magnitude
magmin = 18. ; dmag = 0.5
magbin = np.arange(18.,30.,.5) + .5  #excludes endpoint 25.
magerrmean = []

for mag in magbin:
    magerrmean.append(phot[(phot[sysmag2] > mag - 0.5) & (phot[sysmag2] < mag + 0.5)][sysmag2+'err'].mean())
 
magerrmean = np.array(magerrmean)
#Print a few elements of the matrix
for i in range(0,2):
    print  'MAG_ERR = {0:3f}'.format(phot[i][sysmag2+'err'])
    print  'COVAR = {0:3f}'.format(phot[i]['covar'] - phot[i][sysmag2+'err']**2)
    print  'COLOR_ERR = {0:3f}'.format(phot[i]['colorerr'])

#Now import isochrone of given age, [Fe/H], [a/Fe], making desired mass cuts for fitting
iso = read_iso_darth(isoage,isofeh,isoafe,system,mass_min=mass_min,mass_max=mass_max)
iso0 = read_iso_darth(isoage,isofeh,isoafe,system)

isomass = iso['mass'] 
isocol = iso[sysmag1] - iso[sysmag2] 
isomag = iso[sysmag2] + dmod0

isomass0 = iso0['mass'] 
isocol0 = iso0[sysmag1] - iso0[sysmag2] 
isomag0 = iso0[sysmag2] + dmod0

if system == 'wfpc2':
    col_name = r'F606W - F814W' ; mag_name = r'F814W'
elif system == 'sdss':
    col_name = r'$(g - r)_0$' ; mag_name = r'$r_0$'
if system == 'acs':
    col_name = r'F606W - F814W' ; mag_name = r'F814W'
else:
    pass

#isocol = isocol0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]
#isomag = isomag0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]
#isomass = isomass0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]

if 1:
   plt.plot(isocol0,isomag0,lw=1,ls='-')
   plt.plot(isocol,isomag,lw=3,ls='--')
   plt.ylabel(mag_name)
   plt.xlabel(col_name)
   if system == 'acs': plt.axis([-1.25,0.75,10.+dmod0,0+dmod0])
   if system == 'wfpc2': plt.axis([-1.25,0.75,10.+dmod0,0+dmod0])
   if system == 'sdss': plt.axis([-1.25,0.75,6.+dmod0,-2+dmod0])
   if system == 'acs': plt.errorbar(-0.9+0.0*magerrmean,magbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=2.0)
   if system == 'wfpc2': plt.errorbar(-0.9+0.0*magerrmean,magbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=2.0)
   if system == 'sdss': plt.errorbar(-0.2+0.0*magerrmean,magbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=2.0)
   plt.scatter(phot_raw['color'],phot_raw[sysmag2],color='k',marker='.',s=1)
   plt.scatter(phot['color'],phot[sysmag2],color='r',marker='o',s=2)
   #plt.savefig(os.getenv('HOME')+'/Desktop/fitting_data.png',bbox_inches='tight')
   plt.title(titlestring)  #loc=1, urh; loc=2: ul; 3: ll; 4: lr; 5: r
   plt.show()

#raise SystemExit

"""Here, I can play with interpolating isochrone to a regular grid in say rmag
isomag   = x
#f = interp1d(isomag0,isocol0,kind='cubic')
isocol = f(x)
"""

#Shift isochrone using E(B-V) and some A_X to E(B-V) relation
#For more flexibility shift this to a separate function later.
#EBV  = 0.017  ; A_g  =    ; A_r  = 

#Loop over data points and isochrone points 

alpha_arr = [1.95,2.15,2.35,2.55,2.75]  #"x" = -alpha
logL_arr  = np.empty(len(alpha_arr)) ; logL_arr.fill(0.)

tic = timeit.default_timer()

for ialpha,alpha in enumerate(alpha_arr):
    logL_i = 0.0
    #for i in range(1000):
    for i in range(len(phot['color'])):
        delta_color = phot['color'][i] - isocol
        delta_mag   = phot[sysmag2][i]  - isomag
        error_cov = np.array([[phot['colorerr'][i],0.0],[0.0,phot[sysmag2+'err'][i]]])
        a  = likelihood(phot['colorerr'][i],phot[sysmag2+'err'][i],phot['covar'][i],delta_color,delta_mag)
        dN = f_salpeter(isomass,mass_min,mass_max,alpha)
        L_tmp = np.sum(a*dN)
        if L_tmp < 1e-200: logL_tmp = -5000.
        if L_tmp >= 1e-200: logL_tmp = np.log(L_tmp)
        logL_i += logL_tmp
        print i,logL_i
        if 1:
            plt.subplot(2,2,1)
            plt.ylabel(r'$\rho$exp(...)')
            plt.plot(isomass,a*dN,'bo',ms=3,ls='-')
            plt.subplot(2,2,2)
            plt.ylabel(r'$\rho$')
            plt.plot(isomass,dN,'bo',ms=3,ls='-')
            plt.subplot(2,2,3)
            plt.ylabel(r'exp(...)')
            plt.plot(isomass,a,'bo',ms=3,ls='-')
            plt.subplot(2,2,4)
            plt.ylabel(r'$'+sysmag2+'$')
            plt.xlabel(r'$'+sysmag1+' - '+sysmag2+'$')
            plt.axis([-1.0,1.0,11.+dmod0,0+dmod0])
            plt.scatter(phot_raw['color'],phot_raw[sysmag2],marker='.',s=1)
            plt.scatter(phot['color'][i],phot[sysmag2][i],marker='o',color='red',s=8)
            plt.show()
    logL_arr[ialpha] = logL_i   

print alpha_arr
print logL_arr

plt.plot(alpha_arr,logL_arr,'bo',markersize=5)
plt.title(r'ln\,$L$ as Func of $\alpha$')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'ln\,$L$')
#plt.savefig(os.getenv('HOME')+'/Desktop/alpha_lnL.png',bbox_inches='tight')
plt.show()

"""
markers:  .  ,  o  v  ^  >  <  1  2  3  4  8  s  p  *  h  H  +  x  D  d  |   _  
colors:  (R,G,B)-tuple, OR #aaffhh <html>, OR b,g,r,c,m,y,k,w, OR html names, eg burlywood
"""



"""
plt.scatter(phot['r'],phot['g'],c=['k'],marker='.',s=1)
plt.xlim(16,22)
plt.ylim(16,22)
plt.show()
"""

plt.subplot(1,2,1)
plt.scatter(phot['ra'],phot['dec'],c='k',marker='.',s=1)
plt.xlabel(r'$\alpha$',fontdict={'size':12})
plt.ylabel(r'$\delta$',fontdict={'size':12})
plt.xlim(phot['ra'].min()-.01,phot['ra'].max()+.01)
plt.ylim(phot['dec'].min()-.01,phot['dec'].max()+.01)

#plot CMD on native pixels vs interpolated to fixed mag bin (if interpolated enabled)
plt.subplot(1,2,2)
plt.ylabel(r'$r_0$')
plt.xlabel(r'$(g-r)_0$')
plt.axis([-0.2,0.75,6.+dmod0,-2+dmod0])
plt.errorbar(0.0*magerrmean,rbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=3.0)
plt.scatter(phot['col'],phot['mag'],color='b',marker='.',s=1)
plt.plot(isocol0,isomag0+dmod0,'r.',linestyle='-',lw=1.0)
plt.show()

"""

plt.subplot(2,3,1)
plt.scatter(phot['ra'],phot['dec'],c='k',marker='.',s=1)
plt.xlabel(r'$\alpha$',fontdict={'size':12})
plt.ylabel(r'$\delta$',fontdict={'size':12})
plt.xlim(phot['ra'].min()-.01,phot['ra'].max()+.01)
plt.ylim(phot['dec'].min()-.01,phot['dec'].max()+.01)

plt.subplot(2,3,2)
plt.scatter(phot['g']-phot['r'],phot['r'],c='k',marker='.',s=1)
plt.xlabel(r'$g-r$',fontdict={'size':12})
plt.ylabel(r'$r$',fontdict={'size':12})
plt.xlim((phot['g']-phot['r']).min()-.05,(phot['g']-phot['r']).max()+.05)
plt.ylim(phot['r'].max()+.05,phot['r'].min()-.05)

plt.subplot(2,3,3)
plt.scatter(phot['r'],phot['rerr'],c='k',marker='.',s=1)
plt.xlabel(r'$r$',fontdict={'size':12})
plt.ylabel(r'$\sigma_r$',fontdict={'size':12})
plt.xlim(phot['r'].min()-.1,phot['r'].max()+.1)
plt.ylim(0.0,min(phot['rerr'].max(),0.8))

plt.subplot(2,3,4)
plt.scatter(phot['r'],phot['chi'],c='k',marker='.',s=1)
plt.xlabel(r'$r$',fontdict={'size':12})
plt.ylabel(r'$\chi_{\nu}^{2}$',fontdict={'size':12})
plt.xlim(phot['r'].min()-.1,phot['r'].max()+.1)
plt.ylim(0,5)

plt.subplot(2,3,5)
plt.scatter(phot['r'],phot['sharp'],c='k',marker='.',s=1)
plt.xlabel(r'$r$',fontdict={'size':12})
plt.ylabel(r'$sharp$',fontdict={'size':12})
plt.xlim(phot['r'].min()-.1,phot['r'].max()+.1)
plt.ylim(-4,4)

#plot CMD on native pixels vs interpolated to fixed mag bin (if interpolated enabled)
plt.subplot(2,3,6)
plt.ylabel(r'$r_0$')
plt.xlabel(r'$(g-r)_0$')
plt.axis([-0.2,0.75,6.+dmod,-2+dmod])
plt.errorbar(0.0*rerrmean,rbin,xerr=rerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=3.0)
plt.scatter(phot['gr0'],phot['r0'],color='b',marker='.',s=1)
plt.plot(isocol0,isomag0+dmod,'r.',linestyle='-',lw=1.0)
plt.show()

"""
