import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
import pandas as pd

def read_phot(photfile):
    f = open(photfile+'_cfht.db','r')
    ids = ['id','ra','dec','g','gerr','r','rerr','chi','sharp',
        'rd','gl','gb','e_gr','a_g','a_r']
    data_pd = pd.read_csv(f,comment="#",names=ids,delim_whitespace=True,
        header=None,skiprows=6) 
    data_pd['g0'] = data_pd['g'] - data_pd['a_g']    
    data_pd['r0'] = data_pd['r'] - data_pd['a_g']    
    data_pd['gr'] = data_pd['g'] - data_pd['r']   
    data_pd['gr0'] = data_pd['g0'] - data_pd['r0']    
    f.close()
    return data_pd

def read_iso():
    f = open('iso/test.iso','r')
    ids = ['LogTeff','LogG','sdss_g','sdss_r']
    iso_pd = pd.read_csv(f,comment='#',names=ids,delim_whitespace=True,
        header=0,skiprows=8,usecols=(2,3,6,7))
    f.close()
    return iso_pd
    """Alternate way of reading in using astropy
    f = open('iso/test.iso','r')
    iso2 = ascii.read(f,delimeter=' ',comment='#',header_start=None)
    f.close()
     """

def likelihood(cmd_point,iso_point,error_cov):
    """Perform calculations as ndarrays and not as matrices;  have
    checked that the behavior and cpu usage is the same"""
    diff = cmd_point - iso_point
    arg = -0.5*(np.dot(diff,np.dot(np.linalg.inv(error_cov),diff)))
    print diff,arg
    return arg

#Set env variables for latex-style plotting
if len(sys.argv) != 2: sys.exit()
rc('text', usetex=True)
rc('font', family='serif')

#Read in photometry database and extract relevant quantities
phot = read_phot(str(sys.argv[1]))

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

#Now import isochrone 
iso = read_iso()
isocol0 = iso['sdss_g'] - iso['sdss_r']
isomag0    = iso['sdss_r']

"""Here, I can play with interpolating isochrone to a regular grid in say rmag
isomag   = x
#f = interp1d(isomag0,isocol0,kind='cubic')
isocol = f(x)
"""

#Shift isochrone using E(B-V) and some A_X to E(B-V) relation
#For more flexibility shift this to a separate function later.
#EBV  = 0.017  #McConnachie2012
#A_g  = 
#A_r  = 

#Loop over data points and isochrone points 
dmod = 19.11  #McConnachie2012
EBV  = 0.017  #McConnachie2012
i = 0
for j in range(len(isomag0)):
    error_cov = np.array([[phot['grerr'][i],0.0],[0.0,phot['rerr'][i]]])
    print likelihood(np.array([phot['gr'][i],phot['r'][i]]),np.array([isocol0[j],isomag0[j]+dmod]),error_cov)
    #dmod = 19.11  #McConnachie2012
    #EBV  = 0.017  #McConnachie2012
    #plt.ylabel('r')
    #plt.xlabel('g-r')
    #plt.axis([-0.2,1.2,12.+dmod,-2+dmod])
    #plt.scatter(phot['gr0'][i],phot['r0'][i],color='green',marker='^',s=20)
    #plt.plot(isocol0,isomag0+dmod,'r.',linestyle='-',lw=1.0)
    #plt.scatter(isocol0[j],isomag0[j]+dmod,color='blue',marker='o',s=15)
    #plt.show()


#plot CMD on native pixels vs interpolated to fixed mag bin (if interpolated enabled)
plt.ylabel(r'$r_0$')
plt.xlabel(r'$(g-r)_0$')
plt.axis([-0.2,0.75,6.+dmod,-2+dmod])
plt.errorbar(0.0*rerrmean,rbin,xerr=rerrmean,yerr=None,fmt=None,ecolor='magenta',elinewidth=3.0)
plt.scatter(phot['gr0'],phot['r0'],color='b',marker='.',s=1)
plt.plot(isocol0,isomag0+dmod,'r.',linestyle='-',lw=1.0)
plt.show()


"""
markers:  .  ,  o  v  ^  >  <  1  2  3  4  8  s  p  *  h  H  +  x  D  d  |   _  
colors:  (R,G,B)-tuple, OR #aaffhh <html>, OR b,g,r,c,m,y,k,w, OR html names, eg burlywood
"""



"""
plt.scatter(phot['r'],phot['g'],c=['k'],marker='.',s=1)
plt.xlim(16,22)
plt.ylim(16,22)
plt.show()


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

plt.show()
"""
    
