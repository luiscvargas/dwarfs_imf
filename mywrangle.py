import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

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
        ebv_sfd98  = dsphs.loc[photfile,'ebv'] 

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

def read_iso_darth(age,feh,afe,system,**kwargs):
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

    #Incorporate optional mass cuts into read_iso_darth so that they are not done 
    #in calling program.
    if 'mass_min' in kwargs.keys():
        mass_min = kwargs['mass_min']   
    else:
        mass_min = 0.0
    if 'mass_max' in kwargs.keys():
        mass_max = kwargs['mass_max']   
    else:
        mass_max = 100.0

    #Future: possibly interpolate isochrone if it does not satisfy the condition
    #of being finely graded in mass or magnitude - need dM intervals to be small
    #relative to the range of mass considered otherwise dN_*(dM) is not accurate.

    iso_pd = iso_pd[(iso_pd['mass'] >= mass_min) & (iso_pd['mass'] <= mass_max)]

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
