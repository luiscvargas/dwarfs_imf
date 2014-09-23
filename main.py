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
from my_em import *

#Set env variables for latex-style plotting
rc('text', usetex=True)
rc('font', family='serif')

sim = 1

if sim == 1:

    #nstars_arr = [1000,2000,5000]
    #seed_arr = 5*np.arange(10) + 1
    #f_err_arr = [0.2,1.0,5.0]
    #alpha_in_arr = [0.5,1.0,1.5,2.0,2.5,3.0]

    #mag_max_arr = [28.5,29.5,30.5]   <----done manually

    alpha_in = 2.35
    imftype = 'kroupa'
    alpha_in_1 = 1.00
    alpha_in_2 = 3.00
    start_seed = 12345

    system = 'acs'
    sysmag1   = 'F606W'
    sysmag2   = 'F814W'

    isoage = 14.0
    isofeh = -2.5
    isoafe =  0.4
    dmod0  = 20.63  #dmod to Hercules
    #nstars = 4500
    nstars = 2500

    y2max = 30.0

    titlestring = 'Simulated CMD, '+'{0}'.format(nstars)+' Members'

    #Given Nstars, calculate how many should be required so that AFTER cuts, one is left with ~ Nstars.
    #MG: do not make cuts in magnitude due to Malmquist biases and other systematic biases, only do cuts
    #on observations after making mock samples. 

    #Use Herc data to model error as a function of magnitude
    phot = read_phot('Herc',system,sysmag1,sysmag2)
    magsort1 = np.argsort(phot['F606W'])
    magsort2 = np.argsort(phot['F814W'])
    p1 = np.polyfit(phot['F606W'][magsort1],phot['F606Werr'][magsort1],4,cov=False)
    p2 = np.polyfit(phot['F814W'][magsort2],phot['F814Werr'][magsort2],4,cov=False)
    magarr1 = np.arange(22.,32.,.01)  
    magarr2 = np.copy(magarr1)
    magerrarr1 = np.polyval(p1,magarr1)
    magerrarr1[magarr1 <= phot['F606W'].min()] = phot['F606Werr'].min()
    magerrarr2 = np.polyval(p2,magarr2)
    magerrarr2[magarr2 <= phot['F814W'].min()] = phot['F814Werr'].min()
    phot_true = np.copy(phot)
    del phot

    #Create simulated data. simulate_cmd module called from myanalysis.py
    #For test version, see test_simulate_cmd.py

    #magerrarr1 = magerrarr1*0.1 #+ .01
    #magerrarr2 = magerrarr2*0.1 #+ .01

    #mass_min cut not included: want all stars in cmd to avoid malmquist bias
    #upper mass cut more complex: imf is zero-age imf, whereas LF changes with
    #age.

    #Define some mass cut way belwo observational cut but not as low as limit of isochrone in order to make
    #MC mock data code run faster (less samples are thrown out)
    if y2max >= 29.4: mass_min_global = 0.12
    if y2max < 29.4: mass_min_global = 0.25

    nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',20.63,24.3,y2max,  
        #y1=24.3, y2=28.5
       imftype='salpeter',alpha=alpha_in,mass_min_global=mass_min_global)

    if imftype == 'salpeter':
        phot = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
        system,sysmag1,sysmag2,imftype='salpeter',alpha=alpha_in,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional
    elif imftype == 'chabrier':
        phot = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
        system,sysmag1,sysmag2,imftype='chabrier',mc=mc_in,sigmac=sigmac_in,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional
    elif imftype == 'kroupa':
        phot = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
        system,sysmag1,sysmag2,imftype='kroupa',alpha1=alpha_in_1,alpha2=alpha_in_2,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional
     #cuts in color not in estimate_required_n

    phot_raw = np.copy(phot)

    phot     = filter_phot(phot,system,sysmag1,sysmag2,x1=-1.5,x2=2.5,y1=24.3,y2=y2max)

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
    mass_min_global = 0.05
    mass_max_global = 0.77  #mass max must be below MSTO - make plot to check for this?

    nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',20.63,24.3,28.0,  #28.5
       imftype='salpeter',alpha=alpha_in,mass_min_global=mass_min_global)

    dsph_select = str(sys.argv[1])
    titlestring = dsph_select+' CMD'

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
    phot     = read_phot('Herc',system,sysmag1,sysmag2)
    phot_raw = np.copy(phot)
    phot     = filter_phot(phot,system,sysmag1,sysmag2)
    #phot = read_phot(dsph_select,dataset=system,cuts=True)
    #phot_raw = read_phot(dsph_select,dataset=system,cuts=False)

#Find representative errors for bins in magnitude
magmin = 18. ; dmag = 0.5
magbin = np.arange(18.,31.,.5) + .5  #excludes endpoint 25.
magerrmean = []

for mag in magbin:
    magerrmean.append(phot[(phot[sysmag2] > mag - 0.5) & (phot[sysmag2] < mag + 0.5)][sysmag2+'err'].mean())
 
magerrmean = np.array(magerrmean)
#Print a few elements of the matrix: this was useful in debugging colorerr definition in mid sep 2014, do not delete
for i in range(0,2):
    print  'MAG_ERR = {0:3f}'.format(phot[i][sysmag2+'err'])
    print  'COVAR = {0:3f}'.format(phot[i]['covar'] - phot[i][sysmag2+'err']**2)
    print  'COLOR_ERR = {0:3f}'.format(phot[i]['colorerr'])

#Now import isochrone of given age, [Fe/H], [a/Fe], making desired mass cuts for fitting
iso = read_iso_darth(isoage,isofeh,isoafe,system,mass_min=mass_min_fit,mass_max=mass_max_fit)
iso0 = read_iso_darth(isoage,isofeh,isoafe,system)

isomass = iso['mass']                ; isomass0 = iso0['mass']
isocol = iso[sysmag1] - iso[sysmag2] ; isocol0 = iso0[sysmag1] - iso0[sysmag2]
isomag = iso[sysmag2] + dmod0        ; isomag0 = iso0[sysmag2] + dmod0

#Loop over data points and isochrone points 

#alpha_arr = [1.1.95,2.15,2.35,2.55,2.75]  #"x" = -alpha
alpha_arr = np.array([-0.4,0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2,3.6,4.0])

nlogL_arr,result,xtmparr,ytmparr = maximize_em_salpeter(alpha_arr,phot,iso0,sysmag1,sysmag2,dmod0,mass_min_fit,mass_max_fit)

alpha_fit = result[0] ; delta_minus = result[1] ; delta_plus = result[2]

if system == 'wfpc2':
    col_name = r'F606W - F814W (WFPC2)' ; mag_name = r'F814W (WFPC2)'
    xmin = -1.25; xmax = 0.75; ymax = 12+dmod0; ymin = dmod0
    plt.axis([-1.25,0.75,10.+dmod0,0+dmod0])
elif system == 'sdss':
    col_name = r'$(g - r)_0$' ; mag_name = r'$r_0$'
    xmin = -1.25; xmax = 0.75; ymax = 6.+dmod0; ymin = -2.+dmod0
if system == 'acs':
    col_name = r'F606W - F814W (ACS)' ; mag_name = r'F814W (ACS)'
    xmin = -1.25; xmax = 0.75; ymax = 12+dmod0; ymin = dmod0
    plt.axis([-1.25,0.75,10.+dmod0,0+dmod0])
else:
    pass

plt.subplot(2,2,1) 

plt.plot(isocol0,isomag0,lw=1,ls='-',color='green')
plt.plot(isocol,isomag,lw=5,ls='--',color='blue')
plt.ylabel(mag_name)
plt.xlabel(col_name)
plt.axis([xmin,xmax,ymax,ymin])
plt.errorbar(xmin+.35+0.0*magerrmean,magbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=2.0)
plt.scatter(phot_raw['color'],phot_raw[sysmag2],color='k',marker='.',s=1)
plt.scatter(phot['color'],phot[sysmag2],color='r',marker='o',s=2)
plt.title(titlestring)  #loc=1, urh; loc=2: ul; 3: ll; 4: lr; 5: r

dy = ymax - ymin
if alpha_fit > -99.:
    plt.text(xmax-.90,ymin+0.1*dy,r'$\alpha_{in}$='+'{0:5.2f}'.format(alpha_in)+'',fontsize=11)

plt.text(xmax-.90,ymin+0.18*dy,r'Iso Age    ='+'{0:5.2f}'.format(isoage)+' Gy ',fontsize=11)
plt.text(xmax-.90,ymin+0.26*dy,r'Iso [Fe/H] ='+'{0:5.2f}'.format(isofeh)+' ',fontsize=11)
plt.text(xmax-.90,ymin+0.34*dy,r'Iso [$\alpha$/Fe] ='+'{0:5.2f}'.format(isoafe)+' ',fontsize=11)

plt.subplot(2,2,2)
ymin = phot[sysmag2].min()-.3
ymax = phot[sysmag2].max()+.3
nbins = int((ymax-ymin)/0.2)
n_r , rhist = np.histogram(phot[sysmag2],bins=nbins)
plt.bar(rhist[:-1],n_r,rhist[1]-rhist[0],edgecolor='k')
plt.xlabel(r'$'+sysmag2+'$')
plt.ylabel(r'$dN$')
plt.axis([ymin,ymax,0,1.1*max(n_r)])

plt.subplot(2,2,3) 
xmin = alpha_arr.min()-.2; xmax = alpha_arr.max()+.2 
ymax = nlogL_arr.max()+.1*(nlogL_arr.max()-nlogL_arr.min()); ymin = nlogL_arr.min()-.1*(nlogL_arr.max()-nlogL_arr.min())
plt.plot(alpha_arr,nlogL_arr,'bo',markersize=7)
plt.plot(xtmparr,ytmparr,ls='--',lw=1.25,color='blue')
if alpha_fit > -99.: plt.plot([alpha_in,alpha_in],[ymin,ymax],ls='..',lw=1.5,color='red')
plt.axis([xmin,xmax,ymin,ymax])
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$-$ln\,$L$ + $k$')
#plt.savefig(os.getenv('HOME')+'/Desktop/alpha_lnL.png',bbox_inches='tight')
if alpha_fit > -99.: plt.text(xmin+.2*(xmax-xmin),ymax-.2*(ymax-ymin),r'$\alpha\,='+'{0:5.2f}'.format(alpha_fit)+'^{'+
    '{0:+5.2f}'.format(delta_plus)+'}_{'+'{0:5.2f}'.format(delta_minus)+'}$',fontsize=13.5)


plt.show()



