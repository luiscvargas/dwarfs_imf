import json
import os
import numpy as np
import glob
import pandas as pd
from astropy.io import fits
import pickle
import numpy.lib.recfunctions as nlr

"""list of photometric systems in dartmouth lib
[1] UBVRI+2MASS+Kepler: Synthetic (Vega)    
[2] Washingon+DDO51   : Synthetic (Vega)    
[3] HST/WFCP2         : Synthetic (Vega)    
[4] HST/ACS-WFC       : Synthetic (Vega)    
[5] HST/ACS-HRC       : Synthetic (Vega)    
[6] HST/WFC3-UVIS+IR  : Synthetic (Vega)    
[7] Spitzer IRAC      : Synthetic (Vega)    
[8] UKIDSS            : Synthetic (Vega)    
[9] WISE              : Synthetic (Vega)    
[10] CFHT/MegaCam      : Synthetic (AB)      
[11] SDSS/ugriz        : Synthetic (AB)      
[12] PanSTARRS         : Synthetic (AB)      
[13] SkyMapper         : Synthetic (AB)      
[14] BV(RI)c+uvby      : VandenBerg+Clem et al"""

"""List of Y options
  Enter choice of Y: [1-3]

  Y=  [1] 0.245 + 1.5 Z
      [2] 0.33           ([Fe/H]<=0 only)
      [3] 0.40           ([Fe/H]<=0 only)"""

"""List of [a/Fe] options
  Enter choice of [a/Fe]: [1-6]
  [a/Fe]=  [1] -0.2
           [2]  0.0
           [3] +0.2
           [4] +0.4   ([Fe/H]<=0 only)
           [5] +0.6   ([Fe/H]<=0 only)
           [6] +0.8   ([Fe/H]<=0 only)"""

""">> ./interp_feh option1 option2 option3 outfile"""

#feh_grid = np.array([-2.25,-1.75,-1.25,-0.75])
#afe_grid = np.array([0.0,0.2,0.4])

feh_grid = np.array([-2.499,-2.25,-2.00,-1.75,-1.50,-1.25,-1.00,-0.75])
afe_grid = np.array([0.0,0.4])
age_grid = [6.0,10.0,14.0]

#system = 'cfht'

system = raw_input("Specify library to make: cfht, sdss, wfpc2, wfc3, acs >> ")

dtypes_params = [('idx','i2'),('feh','f4'),('afe','f4'),('age','f4')]

if system == 'wfpc2': 
    str_sys = '3'
    grid_dir = 'isochrones/HST_WFPC2/'
    outfile = 'dartmouth_wfpc2.obj'

    #['mass','teff','logg','lum','F218W','F255W','F300W','F336W','F439W','F450W','F555W','F606W','F622W','F675W','F791W','F814W','F850LP']
    #usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)
    #dtypes_read = [('mass','f8'),('teff','f8'),('logg','f8'),('lum','f8'),('F218W','f8'),
    #    ('F255W','f8'),('F300W','f8'),('F336W','f8'),('F439W','f8'),('F450W','f8'),('F555W','f8'),('F606W','f8'),('F622W','f8'),
    #    ('F675W','f8'),('F791W','f8'),('F814W','f8'),('F850LP','f8')]

    #['mass','teff','logg','lum','F555W','F606W','F814W']
    usecols=(1,2,3,4,11,12,16)
    dtypes_read = [('mass','f4'),('teff','f4'),('logg','f4'),('lum','f4'),
        ('F555W','f4'),('F606W','f4'),
        ('F814W','f4')]

elif system == 'wfc3': 
    str_sys = '6'
    grid_dir = 'isochrones/HST_WFC3/'
    outfile = 'dartmouth_wfc3.obj'

    #['mass','teff','logg','lum','UVf555w','UVf814w','IRf110w','IRf160w']
    usecols=(1,2,3,4,27,43,69,79)
    outfile = 'dartmouth_wfc3.obj'
    dtypes_read = [('mass','f4'),('teff','f4'),('logg','f4'),('lum','f4'),('F555W','f4'),
        ('F814W','f4'),('F110W','f4'),('F160W','f4')]

elif system == 'acs': 
    str_sys = '4'
    grid_dir = 'isochrones/HST_ACSWF/'
    outfile = 'dartmouth_acs.obj'

    #['mass','teff','logg','lum','F435W','F475W','F502N','F550M','F555W','F606W','F625W','F658N','F660N','F775W','F814W','F850LP']
    #usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
    #dtypes_read = [('mass','f8'),('teff','f8'),('logg','f8'),('lum','f8'),('F435W','f8'),
    #    ('F475W','f8'),('F502N','f8'),('F550M','f8'),('F555W','f8'),('F606W','f8'),('F625W','f8'),('F658N','f8'),('F660N','f8'),
    #    ('F775W','f8'),('F814W','f8'),('F850LPW','f8')]

    #['mass','teff','logg','lum','F555W','F606W','F814W']
    usecols=(1,2,3,4,9,10,15)
    dtypes_read = [('mass','f4'),('teff','f4'),('logg','f4'),('lum','f4'),
        ('F555W','f4'),('F606W','f4'),('F814W','f4')]

elif system == 'sdss':
    str_sys = '11'
    grid_dir = 'isochrones/SDSSugriz/'
    outfile = 'dartmouth_sdss.obj'

    #['mass','teff','logg','lum','sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']
    usecols=(1,2,3,4,5,6,7,8,9)
    dtypes_read = [('mass','f4'),('teff','f4'),('logg','f4'),('lum','f4'),
        ('u','f4'),('g','f4'),('r','f4'),('i','f4'),('z','f4')] 

elif system == 'cfht': 
    str_sys = '10'
    grid_dir = 'isochrones/CFHTugriz/'
    outfile = 'dartmouth_cfht.obj'

    #['mass','teff','logg','lum','cfht_u','cfht_g','cfht_r','cfht_i','chft_i_old']
    usecols=(1,2,3,4,5,6,7,8,9)
    dtypes_read = [('mass','f4'),('teff','f4'),('logg','f4'),('lum','f4'),
        ('u','f4'),('g','f4'),('r','f4'),('i','f4'),('i_old','f4')]

outarr_master = -1

for i,feh in enumerate(feh_grid):
    for j,afe in enumerate(afe_grid):
        if afe == -0.2: str_afe = "1"
        if afe == 0.0: str_afe = "2"
        if afe == 0.2: str_afe = "3"
        if afe == 0.4: str_afe = "4"
        if afe == 0.6: str_afe = "5"
        if afe == 0.8: str_afe = "6"
        os.system("./iso_interp_feh "+str_sys+" 1 "+str_afe+" "+str(feh)+" tmpisochrone")
        os.system("./isolf_split tmpisochrone")
        filelist = glob.glob("a*tmpisochrone")
        for k,isofile in enumerate(filelist):
            params  = np.zeros( (1,), dtype=dtypes_params )
            age = isofile.split("tmpisochrone")
            age = float(age[0][1:])*.001 
            if age in age_grid: 
                g = open(isofile,'r')
                data = np.genfromtxt(g,dtype=dtypes_read,usecols=usecols,comments='#',names=None)
                if feh < -2.49: feh_ = -2.50  #special case for dartmouth, feh cannot be = -2.50 in interp code
                if feh >= -2.49: feh_ = feh  
                #fill in columns with isochrone parameters: feh,afe,age,and a dummy index
                params = np.zeros( (len(data['mass']),), dtype=dtypes_params )
                params[:]['idx'] = np.arange(len(data['mass']))
                params[:]['feh'] = feh_
                params[:]['afe'] = afe
                params[:]['age'] = age
                #convert from VEGAMAG to STMAG     ,    + ZpSTmag - ZpVEGA
                if system == 'wfpc2':
                #conversions from http://www.stsci.edu/hst/wfpc2/analysis/wfpc2_photflam.html
                    data[:]['F555W'] = data[:]['F555W'] + 22.538 - 22.538
                    data[:]['F606W'] = data[:]['F606W'] + 23.195 - 22.880
                    data[:]['F814W'] = data[:]['F814W'] + 22.906 - 21.641
            
                elif system == 'wfc3':
                #conversions from http://www.stsci.edu/hst/wfc3/phot_zp_lbn/
                    data[:]['F110W'] = data[:]['F110W'] + 28.4401 - 26.0628
                    data[:]['F160W'] = data[:]['F160W'] + 28.1875 - 24.6949
                    data[:]['F555W'] = data[:]['F555W'] + 25.7232 - 25.8160
                    #data[:]['F606W'] = data[:]['F606W'] + 26.2266 - 25.9866
                    data[:]['F814W'] = data[:]['F814W'] + 25.9299 - 24.6803
  
                elif system == 'acs':
                #conversions from http://www.stsci.edu/hst/acs/analysis/zeropoints/zpt.py, use 2014-01-01.
                    data[:]['F555W'] = data[:]['F555W'] + 25.665 - 25.717
                    data[:]['F606W'] = data[:]['F606W'] + 26.664 - 26.407
                    data[:]['F814W'] = data[:]['F814W'] + 26.786 - 25.523

                else:
                    pass
 
                outarr = nlr.merge_arrays([params,data],flatten=True)
            
                if (type(outarr_master) == int):
                    outarr_master = outarr
                else: 
                    outarr_master = np.concatenate((outarr_master,outarr))
                    #print outarr.ndim
                    #print outarr_master.ndim
                g.close()

        os.system("rm -rf tmpisochrone")
        os.system("rm -rf *tmpisochrone")

#with open("tmp.dat",'w') as outfile:
#print type(json.dumps(dictlist))

pickle.dump(outarr_master,open(outfile,"wb"))

