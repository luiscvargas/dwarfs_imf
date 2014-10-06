import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import interpolate
import pandas as pd
import pickle
import numpy.lib.recfunctions as nlr


def read_phot(photfile,system,sysmag1,sysmag2):
    
    #Read-in summary properties for MW dSph galaxies
    dsphs = read_dsph_data()

    #A_X(E(B-V) from Schlafly & Finkbeiner, ApJ (2011), E(B-V) 
    #MUST be in old SFD98. Coeffs take into acct E(B-V)_SFD98 -> E(B-V)_SF11 correction.
    ###Alternative: See Yuan, Liu, & Xiang, MNRAS, 430 (2013) for GALEX, SDSS, 2MASS and WISE
    c_wfcp2 = { 'F555W': 2.755, 'F606W': 2.415, 'F814W': 1.549 }  #WFPC2
    c_wfc3  = { 'F110W': 0.881, 'F160W': 0.512, 'F555W': 2.855, 'F606W': 2.488, 'F814W': 1.536 }  #WFC3
    c_acs   = { 'F555W': 2.792, 'F606W': 2.471, 'F814W': 1.526 }  #ACS
    c_sdss  = { 'u': 4.239, 'g': 3.303, 'r':2.285, 'i':1.698, 'z':1.263 } #SDSS
    c_cfht  = { 'u': 4.239, 'g': 3.303, 'r':2.285, 'i':1.698, 'z':1.263 } #Assume = SDSS

    #Get E(B-V) at direction of dSph
    ebv_sfd98  = dsphs.loc[photfile,'ebv'] 

    #Read-in photometric data. Header magnitude titles must match those from the sysmag 
    #labels used in this project, e.g. F606W, not m606w.
    if system == 'wfpc2':
        f = open(os.getenv('DATA')+'/HST/'+photfile+'_wfpc2.db','r')
        data = np.genfromtxt(f,comments='#',names=True)
    elif system == 'wfc3':
        #f = open(os.getenv('DATA')+'/HST/'+photfile+'_wfc3.db','r')
        pass
    elif system == 'acs':
        f = open(os.getenv('DATA')+'/HST/'+photfile+'_acs.db','r')
        data = np.genfromtxt(f,comments='#',names=True)
    elif system == 'sdss':
        #f = open(os.getenv('DATA')+'/SDSS/'+photfile+'_sdss.db','r')
        pass
    elif system == 'cfht':
        f = open(os.getenv('DATA')+'/CFHT/'+photfile+'_cfht.db','r')
        pass

    dtypes_extra = [('covar','f8'),('color','f8'),('colorerr','f8')]
 
    if 'ra' not in data.dtype.names:
        dtypes_extra = dtypes_extra + [ ('ra','f8'),('dec','f8') ]

    #Define additional data columns needed for analysis
    len_data = len(data[sysmag1])
    array = np.zeros( (len_data,), dtype=dtypes_extra )

    #Join additional columns to data numpy struct array
    data = nlr.merge_arrays([data,array],flatten=True)

    #Correct magnitudes for extinction using single value for E(B-V),except for
    #SDSS/CFHT data
    if system == 'wfpc2': c_ext = c_wfpc2
    if system == 'acs'  : c_ext = c_acs
    if system == 'wfc3' : c_ext = c_wfc3
    if system == 'sdss' : c_ext = c_sdss

    if system != 'sdss' and system != 'cfht':
        data[sysmag1] = data[sysmag1] - c_ext[sysmag1] * ebv_sfd98
        data[sysmag2] = data[sysmag2] - c_ext[sysmag2] * ebv_sfd98
    else:
        print '...using extinction values in data table...' 
        data[sysmag1] = data[sysmag1] - data['a'+sysmag1]
        data[sysmag2] = data[sysmag2] - data['a'+sysmag2]
 
    #Assign values to the additional columns, AFTER extinction corrections
    data['covar'] = 0.0  #assume cov(g,r) = 0.0 for now 
    data['color'] = data[sysmag1] - data[sysmag2]
    data['colorerr'] = np.sqrt(data[sysmag1+'err']**2 + data[sysmag2+'err']**2 - 2.*data['covar'])   

    return data

def filter_phot(data,system,sysmag1,sysmag2,**kwargs):

    if 'y1' in kwargs.keys():
        y1_ = kwargs['y1']
    else: y1_ = 24.3
    if 'y2' in kwargs.keys():
        y2_ = kwargs['y2']
    else: y2_ = 28.5
    if 'x1' in kwargs.keys():
        x1_ = kwargs['x1']
    else: x1_ = -0.7
    if 'x2' in kwargs.keys():
        x2_ = kwargs['x2']
    else: x2_ = 0.2

    if system == 'acs':
        x1 = x1_ ; x2 = x2_ ; y1 = y1_ ; y2 = y2_   #y1 = 24.3, y2=28.5
        data = data[(data['color'] >= x1) & (data['color'] <= x2) & 
        (data['F814W'] <= y2) & (data['F814W'] >= y1) & (data['clip'] == 0)]
        if sysmag2 != 'F814W' or sysmag1 != 'F606W':
            print 'Cuts require sysmag1=F606W, sysmag2=F814'
            raise SystemExit

    if system == 'wfpc2':
        x1 = x1_ ; x2 = x2_ ; y1 = y1_ ; y2 = y2_   #y1 = 24.3, y2=28.5
        data = data[(data['color'] >= x1) & (data['color'] <= x2) & 
        (data['F814W'] <= y2) & (data['F814W'] >= y1) & (data['clip'] == 0)]
        if sysmag2 != 'F814W' or sysmag1 != 'F606W':
            print 'Cuts require sysmag1=F606W, sysmag2=F814'
            raise SystemExit

    if system == 'wfc3':
        print 'not cuts defined yet, must pass cuts via **kwargs'
        #raise SystemExit

    elif system == 'sdss':

        dmod0  = dsphs.loc[dsph_select,'dmod0'] 
        ra_dwarf  = dsphs.loc[dsph_select,'ra']  
        dec_dwarf  = dsphs.loc[dsph_select,'dec']  
        rhalf_dwarf  = dsphs.loc[dsph_select,'r_h']  

        ra = data['ra'] * 15.
        dec = data['dec'] * 15.

        #Create a mask to only select stars within the central 2 rhalf and 90% completeness
        n_half = 1
        dist_center = np.sqrt(((ra-ra_dwarf)*np.cos(dec_dwarf*np.pi/180.))**2 +
          (dec-dec_dwarf)**2) 

        rmin_box = float(raw_input('enter minimum rmag>> '))
        rmax_box = float(raw_input('enter maximum rmag>> '))
        data = data[(dist_center < n_half*rhalf_dwarf/60.) & 
        (data[sysmag2] >= rmin_box) & (data[sysmag2] <= rmax_box)]

    elif system == 'cfht':

        dmod0  = dsphs.loc[dsph_select,'dmod0'] 
        ra_dwarf  = dsphs.loc[dsph_select,'ra']  
        dec_dwarf  = dsphs.loc[dsph_select,'dec']  
        rhalf_dwarf  = dsphs.loc[dsph_select,'r_h']  

        ra = data['ra'] * 15.
        dec = data['dec'] * 15.

        #Create a mask to only select stars within the central 2 rhalf and 90% completeness
        n_half = 1
        dist_center = np.sqrt(((ra-ra_dwarf)*np.cos(dec_dwarf*np.pi/180.))**2 +
          (dec-dec_dwarf)**2) 

        if 0:
            mask = dist_center < n_half*rhalf_dwarf/60. # in deg
            plt.subplot(1,3,1)
            plt.scatter(data['ra'],data['dec'],color='k',marker='.',s=1)
            plt.scatter(data['ra'][mask],data['dec'][mask],color='r',marker='o',s=2)
            plt.xlabel(r'$\alpha$',fontdict={'size':12})
            plt.ylabel(r'$\delta$',fontdict={'size':12})
            plt.xlim(data['ra'][mask].min()-.03,data['ra'][mask].max()+.03)
            plt.ylim(data['dec'][mask].min()-.03,data['dec'][mask].max()+.03)
            plt.subplot(1,3,2)
            plt.ylabel(r'$r_0$')
            plt.xlabel(r'$(g-r)_0$')
            plt.axis([-0.2,0.75,6.+dmod0,-2+dmod0])
            plt.errorbar(0.0*rerrmean,rbin,xerr=rerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=3.0)
            plt.scatter(data['color'],data[sysmag2],color='k',marker='.',s=1)
            plt.scatter(data['color'][mask],data[sysmag2][mask],color='r',marker='o',s=2)
            plt.subplot(1,3,3)
            n_r , rhist = np.histogram(data[sysmag2][mask],bins=50,density=True)
            n_r_sum = np.cumsum(n_r)
            n_r_sum = n_r_sum / n_r_sum.max()
            plt.bar(rhist[:-1],n_r_sum)
            plt.ylabel(r'$r_0$')
            plt.xlabel(r'$N_{cumul}$')
            plt.axis([-2+dmod0,6+dmod0,0,1.1])
            #plt.savefig(os.getenv('HOME')+'/Desktop/select_region.png',bbox_inches='tight')
            plt.show()

        rmin_box = float(raw_input('enter minimum rmag>> '))
        rmax_box = float(raw_input('enter maximum rmag>> '))
        data = data[(dist_center < n_half*rhalf_dwarf/60.) & 
          (data[sysmag2] >= rmin_box) & (data[sysmag2] <= rmax_box)]

    else:
        pass

    return data

def read_iso_darth(age,feh,afe,system,**kwargs):
    #get data from pickled isochrone library
    f = open("iso/"+"dartmouth_"+system+".obj",'rb')
    isodata = pickle.load(f)
    f.close()
    isodata = isodata[(isodata['age'] == age) & (isodata['feh'] == feh) &
                      (isodata['afe'] == afe)] 

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

    if mass_min < isodata['mass'].min(): mass_min = isodata['mass'].min()
    if mass_max > isodata['mass'].max(): mass_max = isodata['mass'].max()

    isodata = isodata[(isodata['mass'] >= mass_min) & (isodata['mass'] <= mass_max)]

    #Future: possibly interpolate isochrone if it does not satisfy the condition
    #of being finely graded in mass or magnitude - need dM intervals to be small
    #relative to the range of mass considered otherwise dN_*(dM) is not accurate.

    if 'interpolate' in kwargs.keys():
        if kwargs['interpolate'] == False:
            return isodata  #skip interpolation and return array as is
        else:
            pass
    else:
        pass

    isodatanew = isodata[0]  #create a np struct array of same type as original

    colnames = isodatanew.dtype.names

    isort = np.argsort(isodata['mass'])
    xarr = np.arange(mass_min,mass_max,0.001)

    isodatanew = np.repeat(isodatanew,len(xarr)) #np.repeat or np.tile work equally well here for a 
               #"1-element" array (one element = one row with multiple named columns

    isodatanew[:]['mass'] = xarr
    isodatanew[:]['idx'] = np.arange(len(xarr))
    isodatanew[:]['feh'] = isodata[0]['feh']
    isodatanew[:]['afe'] = isodata[0]['afe']
    isodatanew[:]['age'] = isodata[0]['afe']

    for icol,colname in enumerate(colnames):

        if (colname != 'idx' and colname != 'feh' and colname != 'afe' and colname != 'age' and
           colname != 'mass'):
            
            f = interpolate.splrep(isodata['mass'][isort],isodata[colname][isort])
            yarr = interpolate.splev(xarr,f)

            #plt.plot(xarr,yarr,lw=4,color='blue')
            #plt.plot(isodata['mass'],isodata[colname],lw=1,color='red')
            #plt.show()

            isodatanew[:][colname] = yarr
   
    return isodatanew

def read_iso_darth_txt(age,feh,afe,system,**kwargs):
    
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

    iso_pd = iso_pd[(iso_pd['mass'] >= mass_min) & (iso_pd['mass'] <= mass_max)].reset_index()

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


