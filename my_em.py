import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
from mywrangle import *
from myanalysis import *

def maximize_em_one(param_arr,phot,iso,sysmag1,sysmag2,dmod0,mass_min_fit,mass_max_fit,imftype,**kwargs):
    #iso is isochrone np struct array passed from main, it may or 
    #may not include cuts in mass, magnitude, etc (does not if iso0
    #is passed in

    testing = 0
    if 'testing' in kwargs.keys():
        if kwargs['testing'] == True: testing = 1
    else: 
        pass

    if 'phot_raw' in kwargs.keys():
        phot_raw = kwargs['phot_raw']

    isomass = iso['mass'] 
    isocol = iso[sysmag1] - iso[sysmag2] 
    isomag = iso[sysmag2] + dmod0

    logL_arr  = np.empty(len(param_arr)) ; logL_arr.fill(0.)

    for iparam,param in enumerate(param_arr):
        logL_i = 0.0
        #for i in range(1000):
        for i in range(len(phot['color'])):
            delta_color = phot['color'][i] - isocol  #note isocol0 goes with isomass0 below, not making cuts 
            delta_mag   = phot[sysmag2][i]  - isomag  #in isochrones, cuts are only done in data.
            error_cov = np.array([[phot['colorerr'][i],0.0],[0.0,phot[sysmag2+'err'][i]]])

            #a  = likelihood(phot[sysmag2+'err'][i],phot['colorerr'][i],phot['covar'][i],delta_color,delta_mag)
            a  = likelihood_nocovar(phot[sysmag2+'err'][i],phot['colorerr'][i],delta_color,delta_mag)
            if imftype == 'salpeter':
                dN = f_salpeter(isomass,mass_min_fit,mass_max_fit,param)
            elif imftype == 'chabrier':
                if 'sigmac' in kwargs.keys():
                    dN = f_chabrier(isomass,mass_min_fit,mass_max_fit,param,kwargs['sigmac']) 
                elif 'mc' in kwargs.keys():
                    dN = f_chabrier(isomass,mass_min_fit,mass_max_fit,kwargs['mc'],param)
                elif ('mc' in kwargs.keys() and 'sigmac' in kwargs.keys()):
                    print "Only mc or sigmac can be fixed"
                    raise SystemExit
                else:
                    print "Fix either mc or sigmac"
                    raise SystemExit
            elif imftype == 'kroupa':
                if 'alpha2' in kwargs.keys():
                    dN = f_kroupa(isomass,mass_min_fit,mass_max_fit,param,kwargs['alpha2'])
                elif 'alpha1' in kwargs.keys():
                    dN = f_kroupa(isomass,mass_min_fit,mass_max_fit,kwargs['alpha1'],param)
                elif ('alpha1' in kwargs.keys() and 'alpha2' in kwargs.keys()):
                    print "Only alpha1 or alpha2 can be fixed"
                    raise SystemExit
                else:
                    print "Fix either alpha1 or alpha2"
                    raise SystemExit

            #print len(a), len(dN), np.sum(a*dN)
            #dN = f_salpeter(isomass0,mass_min,mass_max,alpha)

            L_tmp = np.sum(a*dN)

            if L_tmp < 1e-200: logL_tmp = -5000.
            if L_tmp >= 1e-200: logL_tmp = np.log(L_tmp)
            logL_i += logL_tmp

            if i % 200 == 0: print 'Param = ',param,'i = ',i

            if testing == 1:
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
                if 'phot_raw' in kwargs.keys(): 
                    plt.scatter(phot_raw['color'],phot_raw[sysmag2],marker='.',s=1,color='gray')
                plt.scatter(phot['color'],phot[sysmag2],marker='.',s=1,color='k')
                plt.scatter(phot['color'][i],phot[sysmag2][i],marker='o',color='red',s=8)
                plt.show()
        logL_arr[iparam] = logL_i   

    nlogL_arr = -1.0*logL_arr + logL_arr.max() + 1

    #Post-analysis for lnL curve: find best fitting power-law alpha and its uncertainties 
    #from -lnL + 1/2. (chi^2+1)

    s = interpolate.UnivariateSpline(param_arr,nlogL_arr,k=3)
    xtmparr = np.arange(param_arr.min(),param_arr.max(),0.0025)
    ytmparr = s(xtmparr)

    #find value for minimum 
    hh = np.argmin(ytmparr)
    if hh > 2 and hh <= len(ytmparr)-3:
        param_fit = xtmparr[hh] ; Lfit = ytmparr[hh]
        hh = np.argmin(abs(ytmparr[xtmparr < param_fit]-(Lfit+0.5)))
        param_minus = xtmparr[xtmparr < param_fit][hh]
        hh = np.argmin(abs(ytmparr[xtmparr > param_fit]-(Lfit+0.5)))
        param_plus = xtmparr[xtmparr > param_fit][hh]
        delta_plus = param_plus - param_fit
        delta_minus = param_fit - param_minus 
    else:
        param_fit = -99.0
        delta_plus = 0.0
        delta_minus = 0.0
    hh = np.argmin(ytmparr)
 
    result_arr = np.array([param_fit,delta_minus,delta_plus])

    return nlogL_arr,result_arr,xtmparr,ytmparr

