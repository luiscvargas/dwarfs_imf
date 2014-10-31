#!/usr/bin/env python
"""This program calculates number of stars 
observable with WFC3 for Coma UFD."""

import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from myanalysis import *

#Read-in Coma Data

f = open(os.getenv('DATA')+'/HST/'+'comber_clean.cat','r')
data = np.genfromtxt(f,comments='#',names=True)
f.close()

#Estimate number of * for 12 ACS pointings (202 arcsec^2 total) in fixed mag range.
datasub = data[(data['F814W'] >= 22.5) & (data['F814W'] <= 25.5)]
n_acs = len(datasub['F814W'])

#Plot Coma CMD
plt.scatter(data['F606W']-data['F814W'],data['F814W'],s=1,marker='o',color='b')
plt.scatter(datasub['F606W']-datasub['F814W'],datasub['F814W'],s=1.5,marker='o',color='r')
plt.axis([-1.5,1.0,30,20])
plt.show()

#Estimaate number of * for 4 WFC3 pointings in same mag range.
    #ACS FOV  = 202" x 202", pixel size = 0.05"
    #WFC3 FOV = 123" x 126", pixel size = 0.13" (for *IR channel*, 162"x162" for UVIS)
f_wfc3_to_acs = (4. * 123.* 126.) / (12. * 202. * 202.)

n_wfc3 = round(n_acs * f_wfc3_to_acs)

#n_wfc3 = 100.

print 'N(WFC3/N(ACS): ',f_wfc3_to_acs
print 'N(ACS): ',n_acs
print 'N(WFC3): ',n_wfc3

#Map ACS magnitude range --> WFC3 magnitud0 range

system = 'wfc3'; isoage = 14.0; isofeh = -2.0; isoafe = 0.0; sysmag1 = 'F110W'; sysmag2 = 'F160W'
dmod_coma = 18.22 #\pm{0.20}

mag_min_acs_vega = 22.5
mag_max_acs_vega = 25.5
mag_min_acs_stmag = mag_min_acs_vega + 26.786 - 25.523  #For ACS F814W
mag_max_acs_stmag = mag_max_acs_vega + 26.786 - 25.523  #For ACS F814W

mass_min_global = 0.2  #dummy value
ndum, mass_min_acs, mass_max_acs = estimate_required_n(1000,isoage,isofeh,isoafe,'acs','F814W',dmod_coma,mag_min_acs_stmag,mag_max_acs_stmag,  
                            imftype='salpeter',alpha=2.35,mass_min_global=mass_min_global)

print 'ACS Mass Range: ',mass_min_acs,' < M < ',mass_max_acs

sigmac = 0.69

#mass_min_acs = 0.17

iso0 = read_iso_darth(isoage,isofeh,isoafe,system)
w1 = np.argmin(abs(iso0['mass']-mass_max_acs))
w2 = np.argmin(abs(iso0['mass']-mass_min_acs))
mag_min_wfc3 = iso0[sysmag2][w1] + dmod_coma  #in stmag system
mag_max_wfc3 = iso0[sysmag2][w2] + dmod_coma  #in stmag system

if 0:
    plt.plot(iso0[sysmag1]-iso0[sysmag2],iso0[sysmag2]+dmod_coma,ls='--')
    plt.axis([-1,0,32,20])
    plt.show()
    plt.plot(iso0['mass'],iso0[sysmag2],ls='--')
    plt.show()

print 'M_max = ',mass_max_acs,' ==> ',22.5,' (ACS) --> ',mag_min_wfc3,' (WFC3)'
print 'M_min = ',mass_min_acs,' ==> ',25.5,' (ACS) --> ',mag_max_wfc3,' (WFC3)'

#Extrapolate number of stars for three different IMFs in WFC3

#IMF 1: power law, alpha = 1.3
#IMF 2: lognormal, Mc = 0.3
#IMF 3: lognormal, Mc = 0.4

mass_min_global = 0.11 #set to M << range (M (data))

mag_max_prop = 26.3 + ( 28.1875 - 24.6949  )   #in F160W, 26.9 in F110W, convert to stmag 

nstars = n_wfc3  #This is the number of stars we want in the limited range observed by ACS.

#nall is the estimated number required down to the lower mass limit of the isochrone (Mglobal =~ 0.11 Msun)

nall1, mass_min_fit, mass_max_fit = estimate_required_n(nstars,isoage,isofeh,isoafe,'wfc3','F160W',dmod_coma,mag_min_wfc3,mag_max_wfc3,  
                            imftype='salpeter',alpha=1.3,mass_min_global=mass_min_global)
nall2, mass_min_fit, mass_max_fit = estimate_required_n(nstars,isoage,isofeh,isoafe,'wfc3','F160W',dmod_coma,mag_min_wfc3,mag_max_wfc3,  
                            imftype='chabrier',mc=0.3,sigmac=sigmac,mass_min_global=mass_min_global)
nall3, mass_min_fit, mass_max_fit = estimate_required_n(nstars,isoage,isofeh,isoafe,'wfc3','F160W',dmod_coma,mag_min_wfc3,mag_max_wfc3,  
                            imftype='chabrier',mc=0.4,sigmac=sigmac,mass_min_global=mass_min_global)
seed_arr = 5*np.arange(50) + 101

#define a magnitude error array, guided on analysis from Herc data for ACS?, or on a constant value

error_unique = 0.02
magarr1 = np.arange(24.,31.0,0.01)
magerrarr1 = np.copy(magarr1) * 0.0 + error_unique
magarr2 = np.copy(magarr1)
magerrarr2 = np.copy(magerrarr1)

magbins_cen = np.arange(26.,31.,0.3)
magbins = magbins_cen - 0.15

maglabels = np.array([26.,27.,28.,29.,30.,31.])
masslabels = np.copy(maglabels)*0.0

#Find masses corresponding to each of these magbins

for imag,mag in enumerate(maglabels):

    wmin = np.argmin(abs(mag-dmod_coma-iso0[sysmag2]))
    masslabels[imag] = iso0['mass'][wmin]

#################################################
#create mock samples for the three IMFs

phi_1_arr = np.zeros( (len(seed_arr),len(magbins)-1) )
for iseed,seed in enumerate(seed_arr):

    phot1 = simulate_cmd(nall1,isoage,isofeh,isoafe,dmod_coma,magarr1,magerrarr1,magarr2,magerrarr2,
            system,sysmag1,sysmag2,imftype='salpeter',alpha=1.3,mass_min=mass_min_global,start_seed=seed)  #1.5 bc of additional
    phot1 = filter_phot(phot1,system,sysmag1,sysmag2,x1=-5.0,x2=5.0,y1=mag_min_wfc3,y2=mag_max_prop)
    #print iseed,len(phot1[(phot1[sysmag2] >= mag_min_wfc3) & (phot1[sysmag2] <= mag_max_wfc3)]),len(phot1[(phot1[sysmag2] >= mag_min_wfc3) & (phot1[sysmag2] <= mag_max_prop)])
    phi_1 , maghist1 = np.histogram(phot1[sysmag2],bins=magbins)
    phi_1_arr[iseed,:] = phi_1
    #phi_1_err = np.sqrt(float(phi_1))

phi_2_arr = np.zeros( (len(seed_arr),len(magbins)-1) )
for iseed,seed in enumerate(seed_arr):

    phot2 = simulate_cmd(nall2,isoage,isofeh,isoafe,dmod_coma,magarr1,magerrarr1,magarr2,magerrarr2,
            system,sysmag1,sysmag2,imftype='chabrier',mc=0.3,sigmac=sigmac,mass_min=mass_min_global,testing=False,start_seed=seed)  #1.5 bc of additional
    phot2 = filter_phot(phot2,system,sysmag1,sysmag2,x1=-5.0,x2=5.0,y1=mag_min_wfc3,y2=mag_max_prop)
    #print iseed,len(phot1[(phot1[sysmag2] >= mag_min_wfc3) & (phot1[sysmag2] <= mag_max_wfc3)]),len(phot1[(phot1[sysmag2] >= mag_min_wfc3) & (phot1[sysmag2] <= mag_max_prop)])
    phi_2 , maghist2 = np.histogram(phot2[sysmag2],bins=magbins)
    phi_2_arr[iseed,:] = phi_2
    #phi_1_err = np.sqrt(float(phi_1))

phi_3_arr = np.zeros( (len(seed_arr),len(magbins)-1) )
for iseed,seed in enumerate(seed_arr):

    phot3 = simulate_cmd(nall3,isoage,isofeh,isoafe,dmod_coma,magarr1,magerrarr1,magarr2,magerrarr2,
            system,sysmag1,sysmag2,imftype='chabrier',mc=0.4,sigmac=sigmac,mass_min=mass_min_global,start_seed=seed)  #1.5 bc of additional
    phot3 = filter_phot(phot3,system,sysmag1,sysmag2,x1=-5.0,x2=5.0,y1=mag_min_wfc3,y2=mag_max_prop)
    #print iseed,len(phot1[(phot1[sysmag2] >= mag_min_wfc3) & (phot1[sysmag2] <= mag_max_wfc3)]),len(phot1[(phot1[sysmag2] >= mag_min_wfc3) & (phot1[sysmag2] <= mag_max_prop)])
    phi_3 , maghist3 = np.histogram(phot3[sysmag2],bins=magbins)
    phi_3_arr[iseed,:] = phi_3
    #phi_1_err = np.sqrt(float(phi_1))

phi_1_arr_err = np.sqrt(phi_1_arr) 
phi_1_arr_mean = np.mean(phi_1_arr,axis=0) 
phi_1_arr_mean_err = np.std(phi_1_arr,axis=0) 

phi_2_arr_err = np.sqrt(phi_2_arr) 
phi_2_arr_mean = np.mean(phi_2_arr,axis=0) 
phi_2_arr_mean_err = np.std(phi_2_arr,axis=0) 

phi_3_arr_err = np.sqrt(phi_3_arr) 
phi_3_arr_mean = np.mean(phi_3_arr,axis=0) 
phi_3_arr_mean_err = np.std(phi_3_arr,axis=0) 

magbins = magbins - (28.1875 - 24.6949)  #convert from STmag to Vegamag
maglabels = maglabels - (28.1875 - 24.6949) 

magbins_cen = (magbins[:-1] + magbins[1:])/2.

qq = np.argmin(abs(magbins-25.))

lf1 = np.genfromtxt(open(getenv('ASTRO_DIR')+'/dwarfs_imf/'+'tmp_alpha13.dat','r'),comments='#',names=True,skip_header=9)
w = np.argmin(abs(lf1['F160W']-(25.-dmod_coma)))
dN_darth_alpha13 = 10.**lf1['logdN'] * ( phi_1_arr_mean[qq] / 10.**lf1['logdN'][w])

lf2 = np.genfromtxt(open(getenv('ASTRO_DIR')+'/dwarfs_imf/'+'tmp_mc03.dat','r'),comments='#',names=True,skip_header=9)
w = np.argmin(abs(lf2['F160W']-(25.-dmod_coma)))
dN_darth_mc03 = 10.**lf2['logdN'] * ( phi_2_arr_mean[qq] / 10.**lf2['logdN'][w])

lf3 = np.genfromtxt(open(getenv('ASTRO_DIR')+'/dwarfs_imf/'+'tmp_mc04.dat','r'),comments='#',names=True,skip_header=9)
w = np.argmin(abs(lf3['F160W']-(25.-dmod_coma)))
dN_darth_mc04 = 10.**lf3['logdN'] * ( phi_3_arr_mean[qq] / 10.**lf3['logdN'][w])

#plt.plot(lf['F160W']+dmod_coma,dN_darth,color='magenta',ls='-.')
#plt.show()

fig = plt.figure(1)
ax1 = fig.add_subplot(2,1,1)
ax1.set_xlim(min(magbins_cen)-0.5,max(magbins_cen)+0.5)
ax1.set_ylim(5.,100.)
ax1.errorbar(magbins_cen,phi_1_arr[0,:],yerr=phi_1_arr_err[0,:],color='r',marker='o',markersize=3)
ax1.errorbar(magbins_cen,phi_2_arr[0,:],yerr=phi_2_arr_err[0,:],color='g',marker='o',markersize=3)
ax1.errorbar(magbins_cen,phi_3_arr[0,:],yerr=phi_3_arr_err[0,:],color='b',marker='o',markersize=3)
ax1.set_yscale("log", nonposy='clip')
ax1.set_xlabel(r'$'+sysmag2+'$ (VEGAMAG)')
ax1.set_ylabel(r'Number')
ax2 = ax1.twiny()
ax2.set_xlabel(r"$Mass\,(M_{\odot})$")
ax2.set_xlim(min(magbins_cen)-0.5,max(magbins_cen)+0.5)
ax2.set_xticks(maglabels)
ax2.set_xticklabels(np.array(masslabels,dtype='<S5'))
ax2.text(min(magbins_cen)+.1,50.,r'$N(WFC3) = {0:<3d}$'.format(int(n_wfc3)),fontsize=11)
ax2.text(min(magbins_cen)+.1,65.,r'$N(ACS) = {0:<3d}$'.format(int(n_acs)),fontsize=11)
ax2.text(min(magbins_cen)+.1,80.,r'$Single Mock$',fontsize=11)

w = np.argmin(abs(iso0['mass']-.17))
ax1.plot([iso0[sysmag2][w]+dmod_coma-(28.1875 - 24.6949),iso0[sysmag2][w]+dmod_coma-(28.1875 - 24.6949)],[5.,100],ls='--',color='k',lw=2)
#plt.axis([magbins.min()-.5,magbins.max()+.5,1.,300.])

ax1 = fig.add_subplot(2,1,2)
ax1.errorbar(magbins_cen,phi_1_arr_mean,yerr=phi_1_arr_mean_err,color='r',marker='o',markersize=3)
ax1.errorbar(magbins_cen,phi_2_arr_mean,yerr=phi_2_arr_mean_err,color='g',marker='o',markersize=3)
ax1.errorbar(magbins_cen,phi_3_arr_mean,yerr=phi_3_arr_mean_err,color='b',marker='o',markersize=3)
ax1.plot(lf1['F160W']+dmod_coma,dN_darth_alpha13,'o',color='r')
ax1.plot(lf2['F160W']+dmod_coma,dN_darth_mc03,'o',color='g')
ax1.plot(lf3['F160W']+dmod_coma,dN_darth_mc04,'o',color='b')
ax1.set_yscale("log", nonposy='clip')
ax1.set_xlabel(r'$'+sysmag2+'$ (VEGAMAG)')
ax1.set_ylabel(r'Number')
ax1.axis([magbins_cen.min()-.5,magbins_cen.max()+.5,5.,100.])
ax1.plot([iso0[sysmag2][w]+dmod_coma-(28.1875 - 24.6949),iso0[sysmag2][w]+dmod_coma-(28.1875 - 24.6949)],[5.,100],ls='--',color='k',lw=2)
ax1.text(min(magbins_cen)+.1,80.,r'$Average ({0:<3d} Mocks)$'.format(len(seed_arr)),fontsize=11)

plt.show()

#convert data to VEGAmag

phot1['F160W'] = phot1['F160W'] - (28.1875 - 24.6949)
phot2['F160W'] = phot2['F160W'] - (28.1875 - 24.6949)
phot3['F160W'] = phot3['F160W'] - (28.1875 - 24.6949)

phot1['F110W'] = phot1['F110W'] - (28.4401 - 26.0628)
phot2['F110W'] = phot2['F110W'] - (28.4401 - 26.0628)
phot3['F110W'] = phot3['F110W'] - (28.4401 - 26.0628)

plt.figure(2)
ax = plt.subplot(1,3,1)
ax.set_xlim([.2,.8])
ax.set_ylim([28,18])
ax.set_xlabel('F110W-F160W') ; ax.set_ylabel('F160W')
ax.scatter(phot1['F110W']-phot1['F160W'],phot1['F160W'],s=6,marker='o',c='r')
ax = plt.subplot(1,3,2)
ax.set_xlim([.2,.8])
ax.set_ylim([28,18])
ax.set_xlabel('F110W-F160W') ; ax.set_ylabel('F160W')
ax.scatter(phot2['F110W']-phot2['F160W'],phot2['F160W'],s=6,marker='o',c='g')
ax = plt.subplot(1,3,3)
ax.set_xlim([.2,.8])
ax.set_ylim([28,18])
ax.set_xlabel('F110W-F160W') ; ax.set_ylabel('F160W')
ax.scatter(phot3['F110W']-phot3['F160W'],phot3['F160W'],s=6,marker='o',c='b')

plt.show()


#nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',dmod_coma,mag_min_wfc3,mag_max_wfc3,  
#                            imftype='chabrier',mc=param_in,sigmac=sigmac,mass_min_global=mass_min_global)
#phot2 = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
#                            system,sysmag1,sysmag2,imftype='chabrier',mc=0.3,sigmac=0.69,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional

#nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',dmod_coma,mag_min_wfc3,mag_max_wfc3,  
#                            imftype='chabrier',alpha1=param_in,alpha2=alpha2,mass_min_global=mass_min_global)
#phot3 = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
#                            system,sysmag1,sysmag2,imftype='chabrier',mc=0.4,sigmac=0.69,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional

#Now filter the isochrones by the output magnitude limits of the proposal; note no constraints on sysmag1 at this point, but should be folded in

#phot2     = filter_phot(phot1,system,sysmag1,sysmag2,x1=-1.5,x2=2.5,y1=mag_min_wfc3,y2=mag_max_prop)  #make broad enough for mock data 
#phot3     = filter_phot(phot1,system,sysmag1,sysmag2,x1=-1.5,x2=2.5,y1=mag_min_wfc3,y2=mag_max_prop)  #make broad enough for mock data 

#Plot results

