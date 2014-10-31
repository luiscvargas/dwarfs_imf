import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mywrangle import *
from myanalysis import *

#Now import isochrone of given age, [Fe/H], [a/Fe], making desired mass cuts for fitting
isoage = 10.
isofeh = -2.5
isoafe = 0.4
system = 'acs'
sysmag1 = 'F606W'
sysmag2 = 'F814W'
start_seed = 1234
nstars = 5000
y2max = 29.5
imftype = 'chabrier'
mass_min = 0.30
mass_max = 0.74
dist_mod = 20.63

mass_min = 0.30
mass_max = 0.70

iso = read_iso_darth(isoage,isofeh,isoafe,system)

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

alpha_ = 2.35
alpha_sec = 2.35
mass_min = iso['mass'].min()

xdum = np.arange(mass_min,mass_max,0.0001)
ydum = f_salpeter(xdum,mass_min,mass_max,alpha_)

ydum_cumul = np.cumsum(ydum)
ydum_cumul = ydum_cumul / max(ydum_cumul)

#Invert cumulative function

fcumul = interpolate.splrep(xdum,ydum_cumul)
gcumul = interpolate.splrep(ydum_cumul,xdum)

xtmp_fwd = np.arange(mass_min,mass_max,0.01)
ytmp_fwd = interpolate.splev(xtmp_fwd,fcumul)
xtmp_inv = np.arange(0.,1.,.01)
ytmp_inv = interpolate.splev(xtmp_inv,gcumul)

nstars = 2000
fb = 0.90


#########################################################################

approach = 1

#Approach 1:
#first draw primary from salpeter
#then draw secondaries from truncated salpeter

if approach == 1:

    #First generate very large array in x and y, so that hopefully there will be at least nstars
    #that fall under dN/dM - M relation.
    np.random.seed(seed=12345)

    #Do with rejection sampling

    xranarr_rej = np.arange(nstars)*0.0
    yranarr_rej = np.arange(nstars)*0.0

    #Now find the pairs (x,y) of simulated data that fall under envelope of dN/dM - M relation.
    count = 0
    while 1:
    #print count
        if count == nstars: break
        xtmp = np.random.random_sample(1) * (mass_max - mass_min) + mass_min   
        ytmp = np.random.random_sample(1) * 1.02*ydum.max() 
        idx = np.abs(xdum - xtmp).argmin()
        if (ytmp <= ydum[idx]) & (xdum[idx] >= mass_min) & (xdum[idx] < mass_max):
            xranarr_rej[count] = xtmp
            yranarr_rej[count] = ytmp
            count += 1
        else:
            pass

    mc_mass_arr_pri = xranarr_rej ; mc_mass_arr_sec = np.zeros(len(mc_mass_arr_pri))  
    mc_mass_arr_tot = np.zeros(len(mc_mass_arr_pri))
    fb_arr = np.random.random(len(mc_mass_arr_pri))

    mass_min_sec = iso['mass'].min()  #mass_min_sec must be > mass_min_iso in order to find a matching magnitude

    for imass,mass_pri in enumerate(mc_mass_arr_pri):
 
        if mass_pri < mass_min_sec: 
            print "mass_pri = ",mass_pri
            print "mass of primary too close to lower isochrone mass limit"
            print "no secondary assigned", "fb random value = ",fb_arr[imass]
            raise SystemExit

        if fb_arr[imass] <= fb:

            if mass_pri <= mass_min_sec+0.0002:
                mc_mass_arr_sec[imass] = mass_pri 
                mc_mass_arr_tot[imass] = mass_pri + mass_pri
 
            else:
                xdum = np.arange(mass_min_sec,mass_pri,0.0001)
                ydum = xdum ** (-1.0*alpha_sec)
                ydum = ydum / max(ydum)

                while 1:
               
                #Calculate random deviates within range of masses with non-zero dN/dM2, 
                #max mass is mass of EACH respective primary, NOT maximum mass of all stars 
                #in sample.
                    xtmp = np.random.random_sample(1) * (mass_pri - mass_min_sec) + mass_min_sec
                    ytmp = np.random.random_sample(1) * 1.02 * max(ydum)
                    w = np.argmin(abs(xtmp - xdum))
                    if ytmp <= ydum[w]: break

                #Assign non-zero masses to secondary mass array, and add masses for total mass array
                mc_mass_arr_sec[imass] = xtmp
                mc_mass_arr_tot[imass] = xtmp + mass_pri

        else:
            mc_mass_arr_sec[imass] = -1.0
            mc_mass_arr_tot[imass] = mass_pri

#Approach 2:
#first draw primary from salpeter
#then draw secondaries from full salpeter

if approach == 2:

    #First generate very large array in x and y, so that hopefully there will be at least nstars
    #that fall under dN/dM - M relation.
    np.random.seed(seed=12345)

    #Do with rejection sampling

    xranarr_rej = np.arange(nstars)*0.0
    yranarr_rej = np.arange(nstars)*0.0

    #Now find the pairs (x,y) of simulated data that fall under envelope of dN/dM - M relation.
    count = 0
    while 1:
    #print count
        if count == nstars: break
        xtmp = np.random.random_sample(1) * (mass_max - mass_min) + mass_min   
        ytmp = np.random.random_sample(1) * 1.02*ydum.max() 
        idx = np.abs(xdum - xtmp).argmin()
        if (ytmp <= ydum[idx]) & (xdum[idx] >= mass_min) & (xdum[idx] < mass_max):
            xranarr_rej[count] = xtmp
            yranarr_rej[count] = ytmp
            count += 1
        else:
            pass

    mc_mass_arr_pri = xranarr_rej ; mc_mass_arr_sec = np.zeros(len(mc_mass_arr_pri))  
    mc_mass_arr_tot = np.zeros(len(mc_mass_arr_pri))
    fb_arr = np.random.random(len(mc_mass_arr_pri))

    mass_min_sec = iso['mass'].min()  #mass_min_sec must be > mass_min_iso in order to find a matching magnitude

    for imass,mass_pri in enumerate(mc_mass_arr_pri):
 
        if fb_arr[imass] <= fb:

            if mass_pri <= mass_min_sec+0.0002:
                mc_mass_arr_sec[imass] = mass_pri 
                mc_mass_arr_tot[imass] = mass_pri + mass_pri

            else:
                xdum = np.arange(mass_min_sec,mass_max,0.0001)
                ydum = xdum ** (-1.0*alpha_sec)
                ydum = ydum / max(ydum)

                while 1:
               
                #Calculate random deviates within range of masses with non-zero dN/dM2, 
                #max mass is mass of EACH respective primary, NOT maximum mass of all stars 
                #in sample.
                    xtmp = np.random.random_sample(1) * (mass_max - mass_min_sec) + mass_min_sec
                    ytmp = np.random.random_sample(1) * 1.02 * max(ydum)
                    w = np.argmin(abs(xtmp - xdum))
                    if ytmp <= ydum[w]: break

                #Assign non-zero masses to secondary mass array, and add masses for total mass array
                mc_mass_arr_sec[imass] = xtmp
                mc_mass_arr_tot[imass] = xtmp + mass_pri

        else:
            mc_mass_arr_sec[imass] = -1.0
            mc_mass_arr_tot[imass] = mass_pri

    #swap primary and secondary masses if M2 > M1

    for imass,mass_pri in enumerate(mc_mass_arr_pri):

        if ((mc_mass_arr_sec[imass] > mass_pri) and (mc_mass_arr_sec[imass]) > 0):
            dum = mc_mass_arr_sec[imass]
            mc_mass_arr_sec[imass] = mc_mass_arr_pri[imass]
            mc_mass_arr_pri[imass] = dum


##############################################


#Approach 3:
#first draw primary from salpeter
#then draw secondaries from flat distribution

raise SystemExit
       

bins = np.arange(0.05,0.80,0.0125)
n_mass , mass_bins = np.histogram(mc_mass_arr_pri,bins=bins)
mass_bins = (mass_bins[1:] + mass_bins[:-1]) / 2.
ax1 = plt.subplot(321)
ax1.bar(mass_bins,n_mass,mass_bins[1]-mass_bins[0],edgecolor='k',color='blue',alpha=0.5)
ax1.set_ylabel('N')
ax1.set_ylim(min(n_mass),max(n_mass))
ax1.set_xlim(min(mass_bins),max(mass_bins))
xtxt = min(mass_bins) + 0.5*(max(mass_bins)-min(mass_bins))
ytxt = max(n_mass) - 0.1*(max(n_mass))
ax1.text(xtxt,ytxt,'Mass Primary')

n_mass , mass_bins = np.histogram(mc_mass_arr_sec,bins=bins)
mass_bins = (mass_bins[1:] + mass_bins[:-1]) / 2.
ax2 = plt.subplot(322,sharex=ax1)
ax2.set_ylim(min(n_mass),max(n_mass))
ax2.set_xlim(min(mass_bins),max(mass_bins))
ax2.bar(mass_bins,n_mass,mass_bins[1]-mass_bins[0],edgecolor='k',color='red',alpha=0.5)
ax2.set_ylabel('N')
xtxt = min(mass_bins) + 0.5*(max(mass_bins)-min(mass_bins))
ytxt = max(n_mass) - 0.1*(max(n_mass))
ax2.text(xtxt,ytxt,'Mass Secondary')

mc_mass_arr_comb = np.concatenate((mc_mass_arr_pri,mc_mass_arr_sec[mc_mass_arr_sec > 0]))
n_mass , mass_bins = np.histogram(mc_mass_arr_comb,bins=bins)
mass_bins = (mass_bins[1:] + mass_bins[:-1]) / 2.
ax3 = plt.subplot(323,sharex=ax1)
ax3.bar(mass_bins,n_mass,mass_bins[1]-mass_bins[0],edgecolor='k',color='green',alpha=0.5)
ax3.set_ylabel('N')
ax3.set_xlim(min(mass_bins),max(mass_bins))
ax3.set_ylim(min(n_mass),max(n_mass))
xtxt = min(mass_bins) + 0.5*(max(mass_bins)-min(mass_bins))
ytxt = max(n_mass) - 0.1*(max(n_mass))
ax3.text(xtxt,ytxt,'Mass(Pri) U Mass(Sec)')

bins = np.arange(0.05,1.60,0.0125*2.)
n_mass , mass_bins = np.histogram(mc_mass_arr_tot,bins=bins)
mass_bins = (mass_bins[1:] + mass_bins[:-1]) / 2.
ax4 = plt.subplot(324)
ax4.bar(mass_bins,n_mass,mass_bins[1]-mass_bins[0],edgecolor='k',color='green',alpha=0.5)
ax4.set_ylabel('N')
ax4.set_xlim(min(mass_bins),max(mass_bins))
ax4.set_ylim(min(n_mass),max(n_mass))
xtxt = min(mass_bins) + 0.5*(max(mass_bins)-min(mass_bins))
ytxt = max(n_mass) - 0.1*(max(n_mass))
ax4.text(xtxt,ytxt,'Mass (Pri+Sec)')


qarr = mc_mass_arr_sec[mc_mass_arr_sec > 0.0] / mc_mass_arr_pri[mc_mass_arr_sec > 0.0]
qbins = np.arange(0.0,1.0,0.05)
n_q , q_bins = np.histogram(qarr,bins=qbins)
q_bins = (q_bins[1:] + q_bins[:-1]) / 2.
ax5 = plt.subplot(325)
ax5.bar(q_bins,n_q,q_bins[1]-q_bins[0],edgecolor='k',color='purple',alpha=0.5)
ax5.set_ylabel('N')
ax5.set_xlim(0.0,1.05)
ax5.set_ylim(0.0,max(n_q)*1.1)
xtxt = min(q_bins) + 0.1
ytxt = max(n_q) - 0.1*(max(n_q))
ax5.text(xtxt,ytxt,'q=M2/M1')

qarr = mc_mass_arr_sec[(mc_mass_arr_sec > 0.0) & (mc_mass_arr_pri > 0.3)] / mc_mass_arr_pri[(mc_mass_arr_sec > 0.0) & (mc_mass_arr_pri > 0.3)]
qbins = np.arange(0.0,1.0,0.05)
n_q , q_bins = np.histogram(qarr,bins=qbins)
q_bins = (q_bins[1:] + q_bins[:-1]) / 2.
ax6 = plt.subplot(326)
ax6.bar(q_bins,n_q,q_bins[1]-q_bins[0],edgecolor='k',color='orange',alpha=0.5)
ax6.set_ylabel('N')
ax6.set_xlim(0.0,1.05)
ax6.set_ylim(0.0,max(n_q)*1.1)
xtxt = min(q_bins) + 0.1
ytxt = max(n_q) - 0.1*(max(n_q))
ax6.text(xtxt,ytxt,'q=M2/M1,M1>0.3')

plt.show()




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

