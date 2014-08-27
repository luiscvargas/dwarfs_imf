import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
import pandas as pd
import timeit

def read_phot(photfile):
    f = open(os.getenv('DATA')+'CFHT/'+photfile+'_cfht.db','r')
    ids = ['id','ra','dec','g','gerr','r','rerr','chi','sharp',
        'rd','gl','gb','e_gr','a_g','a_r']
    data_pd = pd.read_csv(f,comment="#",delim_whitespace=True,
        header=0,skiprows=5) 
    data_pd['g0'] = data_pd['g'] - data_pd['ag']    
    data_pd['r0'] = data_pd['r'] - data_pd['ar']    
    data_pd['gr'] = data_pd['g'] - data_pd['r']   
    data_pd['gr0'] = data_pd['g0'] - data_pd['r0']    
    f.close()
    return data_pd

def read_iso():
    f = open('iso/test.iso','r')
    ids = ['mass','teff',';logg','g','r','i']
    iso_pd = pd.read_csv(f,comment='#',names=ids,delim_whitespace=True,
        header=0,skiprows=8,usecols=(1,2,3,6,7,8))
    iso_pd['gr'] = iso_pd['g'] - iso_pd['r']
    f.close()
    return iso_pd

    """Alternate way of reading in using astropy
    f = open('iso/test.iso','r')
    iso2 = ascii.read(f,delimeter=' ',comment='#',header_start=None)
    f.close()
     """

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
    dN_arr = (mass_arr**(-1.*alpha)) * dmass_arr
    dN_arr[mass_arr < mass_min] = 0.0
    dN_arr[mass_arr > mass_max] = 0.0
    return dN_arr

#specify a Chabrier LF, but given in dN/dM. The Chabrier IMF is given typically as dN/d(logM)
#dN/dM = (1/ln10)*(1/M)*dN/dlogM, and this is calculated within the function. Finally, return
#dM, as for f_salpeter .
#Careful: denominator in first term has ln10 = np.log(10), but exponential is log10 M, so np.log10(m)
def f_chabrier(mass_arr,mass_min,mass_max,mass_crit,sigma_mass_crit):
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)  
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] 
    dN_arr = ((1./(np.log(10.)*mass_arr)) * (1./(np.sqrt(2.*np.pi)*sigma_mass_crit)) * 
        np.exp(-1. * (np.log10(mass_arr)-np.log10(mass_crit))**2 / (2. * sigma_mass_crit**2)) * 
        dmass_arr)
    dN_arr[mass_arr < mass_min] = 0.0
    dN_arr[mass_arr > mass_max] = 0.0
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

dsph_select = str(sys.argv[1])

#Read-in MW dwarf spheroidal data, e.g., Mv, distance modulus, velocity dispersion
#The data comes from a data table I try to maintain updated with high quality data 
#for each quantity.
dsphs = read_dsph_data()

#Read in photometry database and extract relevant quantities
phot = read_phot(dsph_select)

#The following two series, as well as phot['rerr'] conform the elements of the phot error matrix
phot['cov'] = 0.0  #assume cov(g,r) = 0.0 for now 
phot['grerr'] = np.sqrt(phot['gerr']**2 + phot['rerr']**2 - 2.*phot['cov'])

#Find representative errors for bins in magnitude
rmin = 18. ; dr = 0.5
rbin = np.arange(18.,25.,.5) + .5  #excludes endpoint 25.
rerrmean = []

for r in rbin:
    rerrmean.append(phot[(phot['r'] > r - 0.5) & (phot['r'] < r + 0.5)].rerr.mean())
 
rerrmean = np.array(rerrmean)
#Print a few elements of the matrix
for i in range(0,2):
    print  'rerr = {0:3f}'.format(phot.loc[i,'rerr'])
    print  'cov = {0:3f}'.format(phot.loc[i,'cov'] - phot.loc[i,'rerr']**2)
    print  'grerr = {0:3f}'.format(phot.loc[i,'grerr'])

dmod0  = dsphs.loc[dsph_select,'dmod0']  #(row,column), use_cols = 0 was set in read_csv to index by dsph name
print 'The distance modulus is {0:4f}'.format(dmod0)
#Loop over data points and isochrone points 
#dmod = 19.11  #McConnachie2012
#EBV  = 0.017  #McConnachie2012

mass_min = 0.05
mass_max = 0.9  #mass max must be below MSTO
#Now import isochrone 
iso = read_iso()
isocol0 = iso['gr']
isomag0 = iso['r'] + dmod0
isomass0 = iso['mass']

isocol = isocol0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]
isomag = isomag0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]
isomass = isomass0[(isomass0 >= mass_min) & (isomass0 <= mass_max)]

print isomass
print len(isomass),len(isocol)

"""Here, I can play with interpolating isochrone to a regular grid in say rmag
isomag   = x
#f = interp1d(isomag0,isocol0,kind='cubic')
isocol = f(x)
"""

#Shift isochrone using E(B-V) and some A_X to E(B-V) relation
#For more flexibility shift this to a separate function later.
#EBV  = 0.017  ; A_g  =    ; A_r  = 

#Loop over data points and isochrone points 
i = 0
tic = timeit.default_timer()
for i in range(2000):
#for i in range(len(phot['grerr'])):
    if i % 1000 == 0: print i
    error_cov = np.array([[phot['grerr'][i],0.0],[0.0,phot['rerr'][i]]])
    a = likelihood(phot['grerr'][i],phot['rerr'][i],phot['cov'][i],phot['grerr'][i]-isocol0,phot['r'][i]-isomag0)


"""
markers:  .  ,  o  v  ^  >  <  1  2  3  4  8  s  p  *  h  H  +  x  D  d  |   _  
colors:  (R,G,B)-tuple, OR #aaffhh <html>, OR b,g,r,c,m,y,k,w, OR html names, eg burlywood
"""



"""
plt.scatter(phot['r'],phot['g'],c=['k'],marker='.',s=1)
plt.xlim(16,22)
plt.ylim(16,22)
plt.show()
"""

plt.subplot(1,2,1)
plt.scatter(phot['ra'],phot['dec'],c='k',marker='.',s=1)
plt.xlabel(r'$\alpha$',fontdict={'size':12})
plt.ylabel(r'$\delta$',fontdict={'size':12})
plt.xlim(phot['ra'].min()-.01,phot['ra'].max()+.01)
plt.ylim(phot['dec'].min()-.01,phot['dec'].max()+.01)

#plot CMD on native pixels vs interpolated to fixed mag bin (if interpolated enabled)
plt.subplot(1,2,2)
plt.ylabel(r'$r_0$')
plt.xlabel(r'$(g-r)_0$')
plt.axis([-0.2,0.75,6.+dmod0,-2+dmod0])
plt.errorbar(0.0*rerrmean,rbin,xerr=rerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=3.0)
plt.scatter(phot['gr0'],phot['r0'],color='b',marker='.',s=1)
plt.plot(isocol0,isomag0+dmod0,'r.',linestyle='-',lw=1.0)
plt.show()

"""

plt.subplot(2,3,1)
plt.scatter(phot['ra'],phot['dec'],c='k',marker='.',s=1)
plt.xlabel(r'$\alpha$',fontdict={'size':12})
plt.ylabel(r'$\delta$',fontdict={'size':12})
plt.xlim(phot['ra'].min()-.01,phot['ra'].max()+.01)
plt.ylim(phot['dec'].min()-.01,phot['dec'].max()+.01)

plt.subplot(2,3,2)
plt.scatter(phot['g']-phot['r'],phot['r'],c='k',marker='.',s=1)
plt.xlabel(r'$g-r$',fontdict={'size':12})
plt.ylabel(r'$r$',fontdict={'size':12})
plt.xlim((phot['g']-phot['r']).min()-.05,(phot['g']-phot['r']).max()+.05)
plt.ylim(phot['r'].max()+.05,phot['r'].min()-.05)

plt.subplot(2,3,3)
plt.scatter(phot['r'],phot['rerr'],c='k',marker='.',s=1)
plt.xlabel(r'$r$',fontdict={'size':12})
plt.ylabel(r'$\sigma_r$',fontdict={'size':12})
plt.xlim(phot['r'].min()-.1,phot['r'].max()+.1)
plt.ylim(0.0,min(phot['rerr'].max(),0.8))

plt.subplot(2,3,4)
plt.scatter(phot['r'],phot['chi'],c='k',marker='.',s=1)
plt.xlabel(r'$r$',fontdict={'size':12})
plt.ylabel(r'$\chi_{\nu}^{2}$',fontdict={'size':12})
plt.xlim(phot['r'].min()-.1,phot['r'].max()+.1)
plt.ylim(0,5)

plt.subplot(2,3,5)
plt.scatter(phot['r'],phot['sharp'],c='k',marker='.',s=1)
plt.xlabel(r'$r$',fontdict={'size':12})
plt.ylabel(r'$sharp$',fontdict={'size':12})
plt.xlim(phot['r'].min()-.1,phot['r'].max()+.1)
plt.ylim(-4,4)

#plot CMD on native pixels vs interpolated to fixed mag bin (if interpolated enabled)
plt.subplot(2,3,6)
plt.ylabel(r'$r_0$')
plt.xlabel(r'$(g-r)_0$')
plt.axis([-0.2,0.75,6.+dmod,-2+dmod])
plt.errorbar(0.0*rerrmean,rbin,xerr=rerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=3.0)
plt.scatter(phot['gr0'],phot['r0'],color='b',marker='.',s=1)
plt.plot(isocol0,isomag0+dmod,'r.',linestyle='-',lw=1.0)
plt.show()

"""
