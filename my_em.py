import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
from mywrangle import *
from myanalysis import *

def maximize_em_salpeter(alpha_arr,phot,iso,sysmag1,sysmag2,dmod0,mass_min_fit,mass_max_fit,**kwargs):
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

    logL_arr  = np.empty(len(alpha_arr)) ; logL_arr.fill(0.)

    for ialpha,alpha in enumerate(alpha_arr):
        logL_i = 0.0
        #for i in range(1000):
        for i in range(len(phot['color'])):
            delta_color = phot['color'][i] - isocol  #note isocol0 goes with isomass0 below, not making cuts 
            delta_mag   = phot[sysmag2][i]  - isomag  #in isochrones, cuts are only done in data.
            error_cov = np.array([[phot['colorerr'][i],0.0],[0.0,phot[sysmag2+'err'][i]]])

            #a  = likelihood(phot[sysmag2+'err'][i],phot['colorerr'][i],phot['covar'][i],delta_color,delta_mag)
            a  = likelihood_nocovar(phot[sysmag2+'err'][i],phot['colorerr'][i],delta_color,delta_mag)
            dN = f_salpeter(isomass,mass_min_fit,mass_max_fit,alpha)

            #print len(a), len(dN), np.sum(a*dN)
            #dN = f_salpeter(isomass0,mass_min,mass_max,alpha)

            L_tmp = np.sum(a*dN)

            if L_tmp < 1e-200: logL_tmp = -5000.
            if L_tmp >= 1e-200: logL_tmp = np.log(L_tmp)
            logL_i += logL_tmp

            if i % 200 == 0: print 'alpha = ',alpha,'i = ',i

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
        logL_arr[ialpha] = logL_i   

    nlogL_arr = -1.0*logL_arr + logL_arr.max() + 1

    #Post-analysis for lnL curve: find best fitting power-law alpha and its uncertainties 
    #from -lnL + 1/2. (chi^2+1)

    s = interpolate.UnivariateSpline(alpha_arr,nlogL_arr,k=3)
    xtmparr = np.arange(alpha_arr.min(),alpha_arr.max(),0.01)
    ytmparr = s(xtmparr)

    #find value for minimum 
    hh = np.argmin(ytmparr)
    if hh > 2 and hh <= len(ytmparr)-3:
        alpha_fit = xtmparr[hh] ; Lfit = ytmparr[hh]
        hh = np.argmin(abs(ytmparr[xtmparr < alpha_fit]-(Lfit+0.5)))
        alpha_minus = xtmparr[xtmparr < alpha_fit][hh]
        hh = np.argmin(abs(ytmparr[xtmparr > alpha_fit]-(Lfit+0.5)))
        alpha_plus = xtmparr[xtmparr > alpha_fit][hh]
        delta_plus = alpha_plus - alpha_fit
        delta_minus = alpha_fit - alpha_minus 
    else:
        alpha_fit = -99.0
        delta_plus = 0.0
        delta_minus = 0.0
    hh = np.argmin(ytmparr)
 
    result_arr = np.array([alpha_fit,delta_minus,delta_plus])

    return nlogL_arr,result_arr,xtmparr,ytmparr

