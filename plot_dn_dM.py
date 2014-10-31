import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mywrangle import *
from myanalysis import *

#Now import isochrone of given age, [Fe/H], [a/Fe], making desired mass cuts for fitting
isoage = 14.
isofeh = -2.5
isoafe = 0.4
system = 'acs'
sysmag1 = 'F606W'
sysmag2 = 'F814W'
start_seed = 1234
nstars = 50000
y2max = 29.5
imftype = 'chabrier'
mass_min = 0.30
mass_max = 0.74
dist_mod = 20.63

iso = read_iso_darth(isoage,isofeh,isoafe,system)
#iso = read_iso_darth(isoage,isofeh,isoafe,system,mass_min=mass_min,mass_max=mass_max)

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

if mass_max <= mass_min: raise SystemExit
if mass_min <= iso['mass'].min(): mass_min = iso['mass'].min()
if mass_max >= iso['mass'].max(): mass_max = iso['mass'].max()

if imftype == 'salpeter':
    alpha_ = 2.35

elif imftype == 'chabrier':
    mc_ = 0.30
    sigmac_ = 0.69

elif imftype == 'kroupa':
    alpha1_ = 2.30
    alpha2_ = 2.30

xdum = np.arange(mass_min,mass_max,0.0001)
if imftype == 'salpeter':
    ydum = f_salpeter(xdum,mass_min,mass_max,alpha_)
    ydum2 = f_salpeter(xdum,mass_min,mass_max,1.31)
    ydum3 = f_salpeter(xdum,mass_min,mass_max,0.5)
elif imftype == 'chabrier':
    ydum = f_chabrier(xdum,mass_min,mass_max,mc_,sigmac_)
    ydum2 = f_chabrier(xdum,mass_min,mass_max,mc_+.2,sigmac_)
    ydum3 = f_chabrier(xdum,mass_min,mass_max,mc_+.4,sigmac_)
elif imftype == 'kroupa':
    ydum = f_kroupa(xdum,mass_min,mass_max,alpha1_,alpha2_)
    ydum2 = f_kroupa(xdum,mass_min,mass_max,alpha1_-0.8,alpha2_)
    ydum3 = f_kroupa(xdum,mass_min,mass_max,alpha1_-1.6,alpha2_)

ydum_cumul = np.cumsum(ydum)

ydum_cumul = ydum_cumul / max(ydum_cumul)

#Invert cumulative function

fcumul = interpolate.splrep(xdum,ydum_cumul)
gcumul = interpolate.splrep(ydum_cumul,xdum)

xtmp_fwd = np.arange(mass_min,mass_max,0.01)
ytmp_fwd = interpolate.splev(xtmp_fwd,fcumul)
xtmp_inv = np.arange(0.,1.,.01)
ytmp_inv = interpolate.splev(xtmp_inv,gcumul)


if 1:
    plt.subplot(1,2,1)
    plt.plot(xdum,ydum,color='r',ls='-')
    plt.plot(xdum,ydum2,color='g',ls='-')
    plt.plot(xdum,ydum3,color='b',ls='-')
    plt.axis([mass_min,mass_max,0.0,ydum.max()])
    plt.xlabel(r"$M$") ; plt.ylabel(r"$dN/dM$")
    plt.subplot(1,2,2)
    plt.loglog(xdum,ydum,color='r',ls='-',basex=10,basey=10)
    plt.loglog(xdum,ydum2,color='g',ls='-',basex=10,basey=10)
    plt.loglog(xdum,ydum3,color='b',ls='-',basex=10,basey=10)
    plt.axis([mass_min,mass_max,ydum.min(),ydum.max()])
    plt.xlabel(r"log\,$M$") ; plt.ylabel(r"log\,$dN/dM$")
    plt.show()

#############################################

#First generate very large array in x and y, so that hopefully there will be at least nstars
#that fall under dN/dM - M relation.
np.random.seed(seed=12345)

#Do with rejection sampling

xrantmparr = np.random.random_sample(nstars*15) * (mass_max - mass_min) + mass_min   
yrantmparr = np.random.random_sample(nstars*15) * 1.02*ydum.max() 

xranarr_rej = np.arange(nstars)*0.0
yranarr_rej = np.arange(nstars)*0.0

#Now find the pairs (x,y) of simulated data that fall under envelope of dN/dM - M relation.
count = 0
for i,xrantmp in enumerate(xrantmparr):
  #print count
   if count == nstars: break
   idx = np.abs(xdum - xrantmp).argmin()
   if (yrantmparr[i] <= ydum[idx]) & (xdum[idx] >= mass_min) & (xdum[idx] < mass_max):
       xranarr_rej[count] = xrantmparr[i]
       yranarr_rej[count] = yrantmparr[i]
       count += 1
   else:
       pass

   if len(yranarr_rej[yranarr_rej > 0.0]) < nstars:
       print "Need to generate more samples!"
       raise SystemExit

mc_mass_arr_pri = xranarr_rej

mc_mass_arr_tot = mc_mass_arr_pri

#Interpolate isochrone magnitude-mass relation
isort = np.argsort(iso['mass'])  #! argsort = returns indices for sorted array, sort=returns sorted array
#if testing == 1:
#    plt.plot(iso['mass'][isort],iso[sysmag2][isort]+dist_mod,'b.',ls='--')
#    plt.show()
f1 = interpolate.splrep(iso['mass'][isort],iso[sysmag1][isort]+dist_mod)
f2 = interpolate.splrep(iso['mass'][isort],iso[sysmag2][isort]+dist_mod)

#Assign magnitudes to each star based on their mass and the mass-magnitude relation calculated above.
mag1ranarr_0 = interpolate.splev(mc_mass_arr_tot,f1)
mag2ranarr_0 = interpolate.splev(mc_mass_arr_tot,f2)  #band 2 = for system=wfpc2
der_mag2ranarr_0 = interpolate.splev(xranarr,f2,der=1)  #band 2 = for system=wfpc2
colorranarr_0  = mag1ranarr_0 - mag2ranarr_0

#############################################

np.random.seed(seed=12345)
xrantmparr2 = np.random.random_sample(nstars*15) * (mass_max - mass_min) + mass_min   
yrantmparr2 = np.random.random_sample(nstars*15) * 1.02*ydum2.max() 
xranarr2 = np.arange(nstars)*0.0
yranarr2 = np.arange(nstars)*0.0

count = 0
for i,xrantmp2 in enumerate(xrantmparr2):
    #print count
    if count == nstars: break
    idx = np.abs(xdum - xrantmp2).argmin()
    #if (yrantmparr[i] <= ydum[idx]) & (xdum[idx] > iso['mass'].min()) & (xdum[idx] < iso['mass'].max()):
    if (yrantmparr2[i] <= ydum2[idx]) & (xdum[idx] >= mass_min) & (xdum[idx] < mass_max):
        xranarr2[count] = xrantmparr2[i]
        yranarr2[count] = yrantmparr2[i]
        #print count,xrantmparr[i],yrantmparr[i],xranarr[count],yranarr[count]
        count += 1
    else:
        pass

#Assign magnitudes to each star based on their mass and the mass-magnitude relation calculated above.
mag1ranarr_0_2 = interpolate.splev(xranarr2,f1)
mag2ranarr_0_2 = interpolate.splev(xranarr2,f2)  #band 2 = for system=wfpc2
der_mag2ranarr_0_2 = interpolate.splev(xranarr2,f2,der=1)  #band 2 = for system=wfpc2
colorranarr_0_2  = mag1ranarr_0_2 - mag2ranarr_0_2


#############################################

np.random.seed(seed=12345)
xrantmparr3 = np.random.random_sample(nstars*15) * (mass_max - mass_min) + mass_min   
yrantmparr3 = np.random.random_sample(nstars*15) * 1.02*ydum3.max() 
xranarr3 = np.arange(nstars)*0.0
yranarr3 = np.arange(nstars)*0.0

count = 0
for i,xrantmp3 in enumerate(xrantmparr3):
    #print count
    if count == nstars: break
    idx = np.abs(xdum - xrantmp3).argmin()
    #if (yrantmparr[i] <= ydum[idx]) & (xdum[idx] > iso['mass'].min()) & (xdum[idx] < iso['mass'].max()):
    if (yrantmparr3[i] <= ydum3[idx]) & (xdum[idx] >= mass_min) & (xdum[idx] < mass_max):
        xranarr3[count] = xrantmparr3[i]
        yranarr3[count] = yrantmparr3[i]
        #print count,xrantmparr[i],yrantmparr[i],xranarr[count],yranarr[count]
        count += 1
    else:
        pass

#Assign magnitudes to each star based on their mass and the mass-magnitude relation calculated above.
mag1ranarr_0_3 = interpolate.splev(xranarr3,f1)
mag2ranarr_0_3 = interpolate.splev(xranarr3,f2)  #band 2 = for system=wfpc2
der_mag2ranarr_0_3 = interpolate.splev(xranarr3,f2,der=1)  #band 2 = for system=wfpc2
colorranarr_0_3  = mag1ranarr_0_3 - mag2ranarr_0_3

plt.hist(xranarr3,bins=30,range=(mass_min,mass_max),normed=1,color='b',alpha=.3)
plt.hist(xranarr2,bins=30,range=(mass_min,mass_max),normed=1,color='g',alpha=.3)
plt.hist(xranarr,bins=30,range=(mass_min,mass_max),normed=1,color='r',alpha=.3)
plt.show()
plt.hist(mag2ranarr_0_3,bins=30,range=(24.5,29.5),normed=1,color='b',alpha=.3)
plt.hist(mag2ranarr_0_2,bins=30,range=(24.5,29.5),normed=1,color='g',alpha=.3)
plt.hist(mag2ranarr_0,bins=30,range=(24.5,29.5),normed=1,color='r',alpha=.3)
plt.show()

#############################################


ax = plt.subplot(2,1,1) 
nbins = int((mag2ranarr_0.max()-mag2ranarr_0.min())/0.25)
n_r , rhist = np.histogram(mag2ranarr_0,bins=nbins)
rhist_err = np.sqrt(n_r)
n_r2 , rhist2 = np.histogram(mag2ranarr_0_2,bins=nbins)
rhist_err2 = np.sqrt(n_r2)
n_r3 , rhist3 = np.histogram(mag2ranarr_0_3,bins=nbins)
rhist_err3 = np.sqrt(n_r3)
ax.set_yscale("log", nonposy='clip')
g = np.argmin(abs(rhist-25.))
n_r2 = (float(n_r[g])/n_r2[g]) * n_r2
n_r3 = (float(n_r[g])/n_r3[g]) * n_r3
#ax.set_xscale("log", nonposx='clip')
plt.errorbar(rhist[:-1],n_r,yerr=rhist_err,color='r',marker='o',markersize=1)
plt.errorbar(rhist2[:-1],n_r2,yerr=rhist_err2,color='g',marker='o',markersize=1)
plt.errorbar(rhist3[:-1],n_r3,yerr=rhist_err3,color='b',marker='o',markersize=1)
plt.xlabel(r'$'+sysmag2+'$(zero scatter)')
plt.ylabel(r'$dN$')
plt.axis([24.5,mag2ranarr_0.max(),.05*max(n_r),1.5*max(n_r)])

ax = plt.subplot(2,1,2) 
nbins = int((xranarr.max()-xranarr.min())/0.025)
n_r , rhist = np.histogram(xranarr,bins=nbins)
rhist_err = np.sqrt(n_r)
n_r2 , rhist2 = np.histogram(xranarr2,bins=nbins)
rhist_err2 = np.sqrt(n_r2)
n_r3 , rhist3 = np.histogram(xranarr3,bins=nbins)
rhist_err3 = np.sqrt(n_r3)
g = np.argmin(abs(rhist-0.58))
n_r2 = (float(n_r[g])/n_r2[g]) * n_r2
n_r3 = (float(n_r[g])/n_r3[g]) * n_r3

ax.set_yscale("log", nonposy='clip')
ax.set_xscale("log", nonposx='clip')
plt.errorbar(rhist[:-1],n_r,yerr=rhist_err,color='r',marker='o',markersize=1)
plt.errorbar(rhist2[:-1],n_r2,yerr=rhist_err2,color='g',marker='o',markersize=1)
plt.errorbar(rhist3[:-1],n_r3,yerr=rhist_err3,color='b',marker='o',markersize=1)
plt.xlabel(r'log M')
plt.ylabel(r'$dN$')
plt.axis([xranarr.min(),xranarr.max(),.2*max(n_r),1.5*max(n_r)])
plt.plot([0.5,0.5],[.08*max(n_r),1.5*max(n_r)],color='g',ls='-.',lw=2)
plt.plot([0.6,0.6],[.08*max(n_r),1.5*max(n_r)],color='magenta',ls='-.',lw=2)
plt.plot([0.4,0.4],[.08*max(n_r),1.5*max(n_r)],color='orange',ls='-.',lw=2)
plt.show()
