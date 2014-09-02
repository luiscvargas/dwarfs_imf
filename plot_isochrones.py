import sys
import os 
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
import pandas as pd
import timeit

def read_phot(photfile,**kwargs):
    if 'dataset' in kwargs.keys():
        tel = kwargs['dataset']
    else:
        tel = ''
    #Read-in photometric data, if dataset = 'hst', use HST data otherwise use cfht data
    if tel == 'wfpc2':
        f = open(os.getenv('DATA')+'/HST/'+photfile+'_hst.db','r')
        data_pd = pd.read_csv(f,comment="#",delim_whitespace=True,
            header=0)

	#Get E(B-V) at direction of dSph
        dsphs = read_dsph_data()
        ebv_sfd98  = dsphs.loc[dsph_select,'ebv'] 

	#Define ratios of extinction to E(B-V)_SFD98 from Schlafly & Finkbeiner 2011
        f_m606w = 2.415
        f_m814w = 1.549

	#De-redden and correct for extinction, and assign values to color, magnitude, and error series.
        data_pd['cov'] = 0.0  #assume cov(g,r) = 0.0 for now 
        data_pd['mag'] = data_pd['m814'] - f_m814w * ebv_sfd98
        data_pd['col'] = (data_pd['m606'] - f_m606w * ebv_sfd98) - (data_pd['m814'] - f_m814w * ebv_sfd98)
        data_pd['magerr'] = data_pd['m814err']
        data_pd['colerr'] = np.sqrt(data_pd['m606err']**2 + data_pd['m814err']**2 - 2.*data_pd['cov'])   

        x1=-0.7 ; x2= 0.2 ; y1=24.3 ; y2=28.5
        if 'cuts' in kwargs.keys(): 
            if kwargs['cuts'] == True: data_pd = data_pd[(data_pd['col'] >= x1) & (data_pd['col'] <= x2) & 
                  (data_pd['m814'] <= y2) & (data_pd['m814'] >= y1) & (data_pd['clip'] == 0)].reset_index()

    elif tel == 'sdss':
        f = open(os.getenv('DATA')+'/CFHT/'+photfile+'_cfht.db','r')
        #ids = ['id','ra','dec','g','gerr','r','rerr','chi','sharp',
        #    'rd','gl','gb','e_gr','a_g','a_r']
        data_pd = pd.read_csv(f,comment="#",delim_whitespace=True,
            header=0,skiprows=5) 
        f.close()

        dsphs = read_dsph_data()
        dmod0  = dsphs.loc[dsph_select,'dmod0'] 
        ra_dwarf  = dsphs.loc[dsph_select,'ra']  
        dec_dwarf  = dsphs.loc[dsph_select,'dec']  
        rhalf_dwarf  = dsphs.loc[dsph_select,'r_h']  

        data_pd['cov'] = 0.0  #assume cov(g,r) = 0.0 for now 
        data_pd['grerr'] = np.sqrt(data_pd['gerr']**2 + data_pd['rerr']**2 - 2.*data_pd['cov'])   
        data_pd['ra'] = data_pd['ra'] * 15.

        data_pd['mag'] = data_pd['r'] - data_pd['ar']
        data_pd['col'] = (data_pd['g'] - data_pd['ag']) - (data_pd['r'] - data_pd['ar'])
        data_pd['magerr'] = data_pd['rerr']
        data_pd['colerr'] = data_pd['grerr']

        #Create a mask to only select stars within the central 2 rhalf and 90% completeness
        n_half = 1
        dist_center = np.sqrt(((data_pd['ra']-ra_dwarf)*np.cos(dec_dwarf*np.pi/180.))**2 +
            (data_pd['dec']-dec_dwarf)**2) 
        data_pd['dist_center'] = dist_center

        if 0:
            mask = dist_center < n_half*rhalf_dwarf/60. # in deg
            plt.subplot(1,3,1)
            plt.scatter(phot['ra'],phot['dec'],color='k',marker='.',s=1)
            plt.scatter(phot['ra'][mask],phot['dec'][mask],color='r',marker='o',s=2)
            plt.xlabel(r'$\alpha$',fontdict={'size':12})
            plt.ylabel(r'$\delta$',fontdict={'size':12})
            plt.xlim(phot['ra'][mask].min()-.03,phot['ra'][mask].max()+.03)
            plt.ylim(phot['dec'][mask].min()-.03,phot['dec'][mask].max()+.03)
            plt.subplot(1,3,2)
            plt.ylabel(r'$r_0$')
            plt.xlabel(r'$(g-r)_0$')
            plt.axis([-0.2,0.75,6.+dmod0,-2+dmod0])
            plt.errorbar(0.0*rerrmean,rbin,xerr=rerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=3.0)
            plt.scatter(phot['col'],phot['mag'],color='k',marker='.',s=1)
            plt.scatter(phot['col'][mask],phot['mag'][mask],color='r',marker='o',s=2)
            plt.subplot(1,3,3)
            n_r , rhist = np.histogram(phot['mag'][mask],bins=50,density=True)
            n_r_sum = np.cumsum(n_r)
            n_r_sum = n_r_sum / n_r_sum.max()
            plt.bar(rhist[:-1],n_r_sum)
            plt.ylabel(r'$r_0$')
            plt.xlabel(r'$N_{cumul}$')
            plt.axis([-2+dmod0,6+dmod0,0,1.1])
            #plt.savefig(os.getenv('HOME')+'/Desktop/select_region.png',bbox_inches='tight')
            plt.show()

        if 'cuts' in kwargs.keys(): 
            if kwargs['cuts'] == True: 
                rmin_box = float(raw_input('enter minimum rmag>> '))
                rmax_box = float(raw_input('enter maximum rmag>> '))
                data_pd = data_pd[(data_pd.dist_center < n_half*rhalf_dwarf/60.) & 
                (data_pd.mag >= rmin_box) & (data_pd.mag <= rmax_box)].reset_index()
            else:
                pass
    else:
        raise sys.exit()
        
    return data_pd

def read_iso_darth(age,feh,afe,system):
    if system == 'sdss': 
        f = open('iso/test.iso','r')
        ids = ['mass','teff','logg','sdss_g','sdss_r','sdss_i']
        iso_pd = pd.read_csv(f,comment='#',names=ids,delim_whitespace=True,
            header=None,skiprows=9,usecols=(1,2,3,6,7,8))
    elif system == 'wfpc2': 
        suffix = '.HST_WFPC2'
        age_str = "{0:0=5d}".format(int(age*1000))
        feh_str = "{0:0=3d}".format(int(abs(feh*100)))
        if feh < 0.0: feh_str = 'fehm'+feh_str
        if feh >= 0.0: feh_str = 'fehp'+feh_str
        if afe == -0.2: afe_str = 'afem2'
        if afe == 0.0: afe_str = 'afep0'
        if afe == 0.2: afe_str = 'afep2'
        if afe == 0.4: afe_str = 'afep4'
        if afe == 0.6: afe_str = 'afep6'
        if afe == 0.8: afe_str = 'afep8' 
        isofile = 'a'+age_str+feh_str+afe_str+suffix
        f = open('iso/'+isofile,'r')
        ids = ['mass','teff','logg','f606w','f814w']
        iso_pd = pd.read_csv(f,comment='#',names=ids,delim_whitespace=True,
            header=None,skiprows=9,usecols=(1,2,3,12,16))
        iso_pd['teff']  = 10.**iso_pd['teff']
        #Convert magnitudes from Vega system to STMAG system
        iso_pd['f606w'] = iso_pd['f606w'] + 23.195 - 22.880
        iso_pd['f814w'] = iso_pd['f814w'] + 22.906 - 21.641
    else:
        pass
    f.close()
    return iso_pd

def read_dsph_data():
    f=open('mwdwarfs_properties_luis.dat')
    g=open('mwdwarfs_properties_luis_2.dat')
    columns = ['dsph','Mv','Mv_err','MV_source','rhelio','erhelio_err','dmod0','dmod0_err',
    'dist_source','r_h','r_h_err','r_h_source','tmp1','tmp2','tmp3','vrad','vrad_err',
    'vgsr','sigma_vel','sigma_vel_err','tmp4','source_kinematics']
    dsphs1 = pd.read_csv(f,names=columns,skiprows=3,comment='#',sep='\s+',header=None,
    usecols=np.arange(22),index_col=0)
    
    dsphs2 = pd.read_csv(g,names=['dsph','ebv','ebv_source','pa','pa_err','pa_source','ellip','ellip_err','ellip_source',
    'ra','dec','source_radec','tmp1','tmp2','tmp3'],skiprows=3,comment='#',sep='\s+',header=None,
    usecols=np.arange(15),index_col=0) 

    f.close()
    g.close()

    del dsphs2['tmp1']
    del dsphs2['tmp2']
    del dsphs2['tmp3']

    dsphs = dsphs1.join(dsphs2,how='inner')  #do not specify on='key' for a given index key IF that key IS the index (here, key=dsph is the index)

    return dsphs

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

#Set env variables for latex-style plotting
if len(sys.argv) != 2: sys.exit()
rc('text', usetex=True)
rc('font', family='serif')

system = 'wfpc2'

dsph_select = str(sys.argv[1])

#Read-in MW dwarf spheroidal data, e.g., Mv, distance modulus, velocity dispersion
#The data comes from a data table I try to maintain updated with high quality data 
#for each quantity.
dsphs = read_dsph_data()

dmod0  = dsphs.loc[dsph_select,'dmod0']  
ra_dwarf  = dsphs.loc[dsph_select,'ra'] 
dec_dwarf  = dsphs.loc[dsph_select,'dec']  
rhalf_dwarf  = dsphs.loc[dsph_select,'r_h']  
print 'The distance modulus is {0:4f}.'.format(dmod0)
print 'The central ra is {0:4f} deg.'.format(ra_dwarf)
print 'The central decl is {0:4f} deg.'.format(dec_dwarf)
print 'The half-light radius is {0:4f} arcmin.'.format(rhalf_dwarf)

#Read in photometry database and extract relevant quantities
phot = read_phot(dsph_select,dataset=system,cuts=True)
phot_raw = read_phot(dsph_select,dataset=system,cuts=False)

#Find representative errors for bins in magnitude
magmin = 18. ; dmag = 0.5
magbin = np.arange(18.,30.,.5) + .5  #excludes endpoint 25.
magerrmean = []

for mag in magbin:
    magerrmean.append(phot[(phot['mag'] > mag - 0.5) & (phot['mag'] < mag + 0.5)].magerr.mean())
 
magerrmean = np.array(magerrmean)
#Print a few elements of the matrix
for i in range(0,2):
    print  'rerr = {0:3f}'.format(phot.loc[i,'magerr'])
    print  'cov = {0:3f}'.format(phot.loc[i,'cov'] - phot.loc[i,'magerr']**2)
    print  'grerr = {0:3f}'.format(phot.loc[i,'colerr'])

#Loop over data points and isochrone points 
#dmod = 19.11  #McConnachie2012
#EBV  = 0.017  #McConnachie2012

#del phot
#phot = photnew

# & (phot.r0 >= rmin_box) & (phot.r0 <= rmax_box)]
    
mass_min = 0.05
mass_max = 0.75  #mass max must be below MSTO - make plot to check for this?

#Now import isochrone: age,feh,afe 

option = raw_input('Choose: 1: plot various ages, 2: plot various [Fe/H], 3: plot various [a/Fe] > ')
option = int(option)

if option == 1:  #ages
    params = [10.0,12.0,14.0]
    params_str = ['Age=10.0','12.0','14.0']

if option == 2:  #[Fe/H]
    params = [-2.5,-2.25,-2.0]
    params_str = ['[Fe/H]=-2.50','-2.25','-2.00']

if option == 3:  #[a/Fe]
    params = [0.0,0.2,0.4]
    params_str = ['[a/Fe]=+0.0','+0.2','+0.4']

colors = ['b','g','r']

#for i,age in enumerate(ages):
#for i,feh in enumerate(fehs):
for i,param in enumerate(params):
    if option==1: iso = read_iso_darth(param,-2.5,0.4,system)
    if option==2: iso = read_iso_darth(14.0,param,0.4,system)
    if option==3: iso = read_iso_darth(14.0,-2.5,param,system)
    isomass0 = iso['mass']
    if system == 'wfpc2':
        isocol0 = iso['f606w'] - iso['f814w']
        isomag0 = iso['f814w'] + dmod0
    elif system == 'sdss':
        isocol0 = iso['sdss_g'] - iso['sdss_r']
        isomag0 = iso['sdss_r'] + dmod0
    else:
        pass

    isocol = isocol0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]
    isomag = isomag0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]
    isomass = isomass0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]

    plt.plot(isocol0,isomag0,lw=2,ls='-',color=colors[i],label=params_str[i])
    #plt.plot(isocol,isomag,lw=1,ls='--')

if system == 'wfpc2':
    col_name = r'$m_{606w} - m_{814w}$'
    mag_name = r'$m_{814w}$'
elif system == 'sdss':
    col_name = r'$(g - r)_0$'
    mag_name = r'$r_0$'
else:
    pass

plt.legend()
plt.ylabel(mag_name)
plt.xlabel(col_name)

if system == 'wfpc2': plt.axis([-1.25,0.75,10.+dmod0,0+dmod0])
if system == 'sdss': plt.axis([-1.25,0.75,6.+dmod0,-2+dmod0])
if system == 'wfpc2': plt.errorbar(-0.9+0.0*magerrmean,magbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=2.0)
if system == 'sdss': plt.errorbar(-0.2+0.0*magerrmean,magbin,xerr=magerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=2.0)

plt.scatter(phot_raw['col'],phot_raw['mag'],color='k',marker='.',s=1,alpha=0.5)
plt.scatter(phot['col'],phot['mag'],color='orange',marker='o',s=2,alpha=0.6)

#plt.savefig(os.getenv('HOME')+'/Desktop/fitting_data.png',bbox_inches='tight')

plt.show()

