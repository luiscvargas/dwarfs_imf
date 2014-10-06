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

#number of stars in mock sample: 
nstars_arr = [1000,2000]

#seed values - sets number of independentm mock samples
seed_arr = 5*np.arange(100) + 101

#depth of sample - maximum F814W magnitude
y2max_arr = [28.5,29.5,30.0]
#y2max_arr = [30.0]

#scaling for photometric uncertainties: 1.0 = error fnc of mag just as in real Herc data
ferr_arr = [0.5,1.0]

#####2x100x2x1 = 400 runs per param_in
#####400 x 3 params = 1200 runs per imftype
#####1200 * 60 sec / run / 3600 sec/ hour <= 20 hours !

imftype = 'chabrier'

#values for parameter to vary - same as parameter to recover
if imftype == 'salpeter': 
    imftype_in = 'salpeter'
    imftype_out = 'salpeter'
    param_in_arr = [1.1,1.7,2.3]  #alpha_in_arr
    param_in_name = 'alpha'
    param_out_arr = np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5]) # alpha_out_arr

if imftype == 'chabrier': 
    imftype_in = 'chabrier'
    imftype_out = 'chabrier'
    sigmac = 0.69  #fixed to be the same for both input and output, if not, need to specify sigmac_in, and sigmac_out
    param_in_arr = [0.08,0.30,0.60]  #mc_in_arr
    param_in_name = 'mc'
    #maximize likelihood over mc
    param_out_arr = np.array([0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]) #mc_out_arr
              ##note mc canNOT be zero, because log mc becomes undefined

if imftype == 'kroupa': 
    imftype_in = 'kroupa'
    imftype_out = 'kroupa'
    alpha2 = 2.30  #fixed to be the same for both input and output, if not, need to specify alpha2_in, and alpha2_out
    param_in_arr = [1.1,1.7] #alpha1_in_arr
    param_in_name = 'alpha1'
    #maximize likelihood over alpha1
    param_out_arr = np.array([0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8]) #alpha1_out_arr

system = 'acs'
sysmag1   = 'F606W'
sysmag2   = 'F814W'
isoage = 14.0
isofeh = -2.5
isoafe =  0.4
dmod0  = 20.63  #dmod to Hercules

#########################################################
#Use Herc data to model error as a function of magnitude
phot = read_phot('Herc',system,sysmag1,sysmag2)
magsort1 = np.argsort(phot['F606W'])
magsort2 = np.argsort(phot['F814W'])
p1 = np.polyfit(phot['F606W'][magsort1],phot['F606Werr'][magsort1],4,cov=False)
p2 = np.polyfit(phot['F814W'][magsort2],phot['F814Werr'][magsort2],4,cov=False)
magarr1 = np.arange(22.,32.,.01)  
magarr2 = np.copy(magarr1)
magerrarr1_ = np.polyval(p1,magarr1)
magerrarr1_[magarr1 <= phot['F606W'].min()] = phot['F606Werr'].min()
magerrarr2_ = np.polyval(p2,magarr2)
magerrarr2_[magarr2 <= phot['F814W'].min()] = phot['F814Werr'].min()
del phot
#########################################################

count_global = 0

for istars, nstars in enumerate(nstars_arr):

    for iseed, start_seed in enumerate(seed_arr):

        for iy2max, y2max in enumerate(y2max_arr):

            for iferr, ferr in enumerate(ferr_arr):

                for iparam_in, param_in in enumerate(param_in_arr):

                    magerrarr1 = ferr * magerrarr1_
                    magerrarr2 = ferr * magerrarr2_

                    #Define some mass cut way belwo observational cut but not as low as limit of isochrone in order to make
                    #MC mock data code run faster (less samples are thrown out)
                    if y2max >= 29.0: mass_min_global = 0.11
                    if y2max < 29.0: mass_min_global = 0.20

                    #y1=24.3, y2=28.5

                    if imftype_in == 'salpeter':
                        nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',20.63,24.3,y2max,  
                            imftype='salpeter',alpha=param_in,mass_min_global=mass_min_global)
                        phot = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
                            system,sysmag1,sysmag2,imftype='salpeter',alpha=param_in,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional
                    if imftype_in == 'chabrier':
                        nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',20.63,24.3,y2max,  
                            imftype='chabrier',mc=param_in,sigmac=sigmac,mass_min_global=mass_min_global)
                        phot = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
                            system,sysmag1,sysmag2,imftype='chabrier',mc=param_in,sigmac=sigmac,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional
                    if imftype_in == 'kroupa':
                        nall, mass_min_fit, mass_max_fit = estimate_required_n(nstars,14.0,-2.5,0.4,'acs','F814W',20.63,24.3,y2max,  
                            imftype='kroupa',alpha1=param_in,alpha2=alpha2,mass_min_global=mass_min_global)
                        phot = simulate_cmd(nall,isoage,isofeh,isoafe,dmod0,magarr1,magerrarr1,magarr2,magerrarr2,
                            system,sysmag1,sysmag2,imftype='kroupa',alpha1=param_in,alpha2=alpha2,mass_min=mass_min_global,start_seed=start_seed)  #1.5 bc of additional

                    phot_raw = np.copy(phot)
                    phot     = filter_phot(phot,system,sysmag1,sysmag2,x1=-1.5,x2=2.5,y1=24.3,y2=y2max)  #make broad enough for mock data 

                    #Print a few elements of the matrix: this was useful in debugging colorerr definition in mid sep 2014, do not delete
                    #for i in range(0,2):
                    #    print  'MAG_ERR = {0:3f}'.format(phot[i][sysmag2+'err'])
                    #    print  'COVAR = {0:3f}'.format(phot[i]['covar'] - phot[i][sysmag2+'err']**2)
                    #    print  'COLOR_ERR = {0:3f}'.format(phot[i]['colorerr'])

                    #Now import isochrone of given age, [Fe/H], [a/Fe], making desired mass cuts for fitting
                    iso = read_iso_darth(isoage,isofeh,isoafe,system,mass_min=mass_min_fit,mass_max=mass_max_fit)
                    iso0 = read_iso_darth(isoage,isofeh,isoafe,system)

                    isomass = iso['mass']                ; isomass0 = iso0['mass']
                    isocol = iso[sysmag1] - iso[sysmag2] ; isocol0 = iso0[sysmag1] - iso0[sysmag2]
                    isomag = iso[sysmag2] + dmod0        ; isomag0 = iso0[sysmag2] + dmod0

                    #Loop over data points and isochrone points 
                    #"x" = -alpha, so dN/dM prop to M^x, M^(-1*alpha)
                    if imftype_out == 'salpeter':
                        nlogL_arr,result,xtmparr,ytmparr = maximize_em_one(param_out_arr,phot,iso0,sysmag1,sysmag2,dmod0,mass_min_fit,mass_max_fit,'salpeter')
                        param_fit = result[0] ; delta_minus = result[1] ; delta_plus = result[2]
                    if imftype_out == 'chabrier':  #sigmac fixed - > input as **kwarg; thus param_out_arr = mc
                        nlogL_arr,result,xtmparr,ytmparr = maximize_em_one(param_out_arr,phot,iso0,sysmag1,sysmag2,dmod0,mass_min_fit,mass_max_fit,'chabrier',sigmac=sigmac)
                        param_fit = result[0] ; delta_minus = result[1] ; delta_plus = result[2]
                    if imftype_out == 'kroupa':    #alpha2 fixed - > input as **kwarg; thus param_out_arr = alpha1
                        nlogL_arr,result,xtmparr,ytmparr = maximize_em_one(param_out_arr,phot,iso0,sysmag1,sysmag2,dmod0,mass_min_fit,mass_max_fit,'kroupa',alpha2=alpha2)
                        param_fit = result[0] ; delta_minus = result[1] ; delta_plus = result[2]
                    if imftype_out == 'salpeter':
                        f = open('results/eda_salpeter.dat', 'a')
                        f.write('{0:<5d},{1:<6d},{2:<7d},{3:<5.2f},{4:<5.2f},{5:<5.2f},{6:<5.2f},{7:<6.3f},{8:<5.3f},{9:<5.3f}\n'.format(count_global,nstars,start_seed,y2max,ferr,param_in,-9.0,param_fit,delta_minus,delta_plus))
                    if imftype_out == 'chabrier':
                        f = open('results/eda_chabrier.dat', 'a')
                        f.write('{0:<5d},{1:<6d},{2:<7d},{3:<5.2f},{4:<5.2f},{5:<5.2f},{6:<5.2f},{7:<6.3f},{8:<5.3f},{9:<5.3f}\n'.format(count_global,nstars,start_seed,y2max,ferr,param_in,sigmac,param_fit,delta_minus,delta_plus))
                    if imftype_out == 'kroupa':
                        f = open('results/eda_kroupa.dat', 'a')
                        f.write('{0:<5d},{1:<6d},{2:<7d},{3:<5.2f},{4:<5.2f},{5:<5.2f},{6:<5.2f},{7:<6.3f},{8:<5.3f},{9:<5.3f}\n'.format(count_global,nstars,start_seed,y2max,ferr,param_in,alpha2,param_fit,delta_minus,delta_plus))
                    f.close()
                    count_global += 1

