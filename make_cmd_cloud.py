#!/usr/bin/python

'''
Created: 2015-01-08
Purpose: Simulate a CMD and generate semi-empirical LF cloud
Inputs : Binary fraction
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mywrangle import *
from myanalysis import *
from my_em import *
from scipy import interpolate

class DartmouthIsochrone(object):

    def __init__(self,feh,afe,age,system):
        try:
            f = open(os.getenv('ASTRO_DIR')+'/dwarfs_imf/iso/'+'dartmouth_'+system+'.obj','rb')
        except:
            raise ValueError("Isochrone library for "+system+" not found!")
        data = pickle.load(f)
        f.close()
        #Check for existence of input parameters in isochrone library
        if age not in np.unique(data['age']):
            raise ValueError("Age not in isochrone library.")
        if feh not in np.unique(data['feh']):
            raise ValueError("[Fe/H] not in isochrone library.")
        if afe not in np.unique(data['afe']):
            raise ValueError("[a/Fe] not in isochrone library.")
        #Select particular isochrone for specified [Fe/H], [a/Fe], and age.
        self.iso = data[(data['age'] == age) & (data['feh'] == feh) &
                      (data['afe'] == afe)]
        self.data = np.zeros(0) #empty placeholder array for interpolated array
        #Assign descriptive variables to object
        self.age = age
        self.afe = afe
        self.feh = feh
        self.system = system
        self.mass_min = self.iso['mass'].min()
        self.mass_max = self.iso['mass'].max()
        self.interp_flag = 0

    def print_params(self):
        print "==Dartmouth Isochrone=="
        print "Age    = {0:.1f} Gyr".format(self.age)
        print "[Fe/H] = {0:.2f}".format(self.feh)
        print "[a/Fe] = {0:.2f}".format(self.afe)
        print "M_min  = {0:.3f} Msun".format(self.mass_min)
        print "M_max  = {0:.3f} Msun".format(self.mass_max)

    def change_min_mass(self,mass_min):
        #If mass_min < min mass available in isochrone, set to min mass available
        if mass_min < self.iso['mass'].min():
            mass_min = self.iso['mass'].min()
        self.mass_min = mass_min

    def change_max_mass(self,mass_max):
        #If mass_max > max mass available in isochrone, set to max mass available
        if mass_max > self.iso['mass'].max():
            mass_max = self.iso['mass'].max()
        self.mass_max = mass_max

    #Future: interpolate isochrone if it does not satisfy the condition
    #of being finely graded in mass or magnitude - need dM intervals to be small
    #relative to the range of mass considered otherwise dN_*(dM) is not accurate.

    def interpolate(self,dm=0.001,diagnose=False):
        #create a np struct array of same type as original.
        #First, sort rows by mass, and interpolate
        isonew = np.copy(self.iso)  # dont use isonew=self.iso!
        isonew = isonew[0]
        isort = np.argsort(self.iso['mass'])

        npts = long((self.mass_max - self.mass_min) / dm)

        #assign size of interpolated array given mass bounds and dm
        massarr = np.linspace(self.mass_min,self.mass_max,npts)

        #check that interpolation would result in more data points that
        #original array, else interpolate.splrep fails.
        if len(massarr) <= len(self.iso['mass']):
            print "No interpolation done; returning..."
            return None
        else:
            print "Proceed to interpolate based on mass..."

        isonew = np.repeat(isonew,len(massarr)) #(aside: np.repeat or np.tile work equally well here)
        isonew['mass'] = massarr
        isonew['idx']  = np.arange(len(massarr))
        isonew['feh']  = self.iso['feh'][0]
        isonew['afe']  = self.iso['afe'][0]
        isonew['age']  = self.iso['age'][0]

        colnames = self.iso.dtype.names

        #tuple is immutable = this line does not work
        #colnames2 = colnames[colnames != 'idx' and colnames != 'feh' and
        #        colnames != 'afe' and colnames != 'age' and colnames != 'mass']

        for icol,colname in enumerate(colnames):

            if (colname != 'idx' and colname != 'feh' and colname != 'afe' and colname != 'age' and
                colname != 'mass'):

                #For each magnitude - mass relation, interpolate
                xx = self.iso['mass'][isort]
                yy = self.iso[colname][isort]
                f = interpolate.splrep(xx,yy)
                magarr = interpolate.splev(massarr,f)

                #plt.plot(massarr,magarr,lw=4,color='blue')
                #plt.plot(self.iso['mass'],self.iso[colname],lw=1,color='red')
                #plt.show()

                isonew[colname] = magarr
            else:
                pass

        #Reassign self.iso using new interpolated array
        self.data = isonew
        self.interp_flag = 1

        if diagnose == True:
            plt.plot(self.iso['F110W'],self.iso['F160W'],'r-',lw=3)
            plt.plot(self.data['F110W'],self.data['F160W'],'b--',lw=1)
            plt.show()

    def has_interp(self):
        if self.interp_flag == 0:
            print "No interpolation done on file"
        else:
            print "Interpolated data located as self.data"
            print "Non-interpolated data located as self.iso"


class SyntheticCMD(object):

    def __init__(self,iso,strmag1,strmag2,abs_mag_min,abs_mag_max,nrequired,fInv,
        modulus=0.0,q=0.0):

        mass_pri_arr = []  #reset mass array
        mass_sec_arr = []  #reset mass array
        mass_arr = []  #reset mass array
        mag1_arr = []
        mag2_arr = []
        ngood = 0

        while ngood < nrequired:
            ranarr = np.random.uniform(size=nrequired)
            mass_raw_arr = interpolate.splev(ranarr,fInv)
            #assign magnitudes to masses for single star/binary case
            for i in range(int(nrequired * (1.-q)/(1.+q))):
                w = np.argmin(abs(iso.data['mass'] - mass_raw_arr[i]))
                mag1_arr.append(iso.data[strmag1][w])
                mag2_arr.append(iso.data[strmag2][w])
                mass_pri_arr.append(mass_raw_arr[i])
                mass_sec_arr.append(0.0)
                mass_arr.append(mass_raw_arr[i])
            for i in range(int(nrequired * (1.-q)/(1.+q)),nrequired-1,2):
                wa = np.argmin(abs(iso.data['mass'] - mass_raw_arr[i]))
                wb = np.argmin(abs(iso.data['mass'] - mass_raw_arr[i+1]))
                if mass_raw_arr[i+1] > mass_raw_arr[i]:
                #swap wa and wb, so wa points to primary
                    wtmp = wa
                    wa = wb
                    wb = wtmp
                mag1_a = iso.data[strmag1][wa]
                mag2_a = iso.data[strmag2][wa]
                mag1_b = iso.data[strmag1][wb]
                mag2_b = iso.data[strmag2][wb]
                gamma1 = 1. + 10.**(0.4*(mag1_a - mag1_b))
                gamma2 = 1. + 10.**(0.4*(mag2_a - mag2_b))
                mag1 = mag1_a - 2.5*np.log10(gamma1)
                mag2 = mag2_a - 2.5*np.log10(gamma2)
                mass_pri = iso.data['mass'][wa]
                mass_sec = iso.data['mass'][wb]
                mass_tot = mass_pri + mass_sec
                mag1_arr.append(mag1)
                mag2_arr.append(mag2)
                mass_pri_arr.append(mass_pri)
                mass_sec_arr.append(mass_sec)
                mass_arr.append(mass_tot)

                #update the number of systems (singles or binaries) within the desired
                #magnitude bins - do NOT make cuts until the end, e.g., cannot chooose
                #not to include stars to list if they do not meet constraints as will
                #bias mass/magnitude distributions.

                #correct for distance modulus assumed for mag_min and mag_max
                #to cut the correct range of magnitudes in absoluate mag space.
            ngood = len([x for x in mag2_arr if (x >= (abs_mag_min)
                and x <= (abs_mag_max))])

        self.mass_pri = np.array(mass_pri_arr[0:nrequired])
        self.mass_sec = np.array(mass_sec_arr[0:nrequired])
        self.mass = np.array(mass_arr[0:nrequired])
        self.mag1 = np.array(mag1_arr[0:nrequired]) + modulus
        self.mag2 = np.array(mag2_arr[0:nrequired]) + modulus
        self.color = self.mag1 - self.mag2
        self.q = q
        self.modulus = modulus
        self.mag1label = strmag1
        self.mag2label = strmag2

    def as_dict(self):
        dict = {}
        dict['mass_pri'] = np.array(self.mass_pri)
        dict['mass_sec'] = np.array(self.mass_sec)
        dict['mass'] = np.array(self.mass)
        dict['mag1'] = np.array(self.mag1)
        dict['mag2'] = np.array(self.mag2)
        dict['color'] = dict['mag1'] - dict['mag2']
        return dict

    def cmdDensity(self,dx,dy):
        from scipy import stats
        import matplotlib.pyplot as plt
        xmin, xmax = min(self.color), max(self.color)
        ymin, ymax = min(self.mag2), max(self.mag2)
        nx = complex(0,(xmax - xmin) / dx)
        ny = complex(0,(ymax - ymin) / dy)
        xg, yg = np.mgrid[xmin:xmax:nx,ymin:ymax:ny]
        posarr = np.vstack([xg.ravel(),yg.ravel()])
        print posarr
        values = np.vstack([self.color,self.mag2])
        kernel = stats.gaussian_kde(values)
        fg = np.reshape(kernel(posarr).T, xg.shape)

        plt.axis([xmin,xmax,ymax,ymin])
        plt.imshow(np.rot90(fg),cmap=plt.cm.gist_earth_r,
            extent=[xmin,xmax,ymin,ymax],aspect='auto',
            interpolation='gaussian')
        cset = plt.contour(xg,yg,fg)
        plt.show()


def simulateScatter(cmd,sigma1,sigma2):
    if isinstance(sigma1,(int,float)):
        mag1 = cmd.mag1 + np.random.randn(len(cmd.mag1)) * sigma1
        mag2 = cmd.mag2 + np.random.randn(len(cmd.mag2)) * sigma2
    else:
        mag1 = cmd.mag1 + np.random.randn(len(cmd.mag1)) * np.polyval(sigma1,cmd.mag1)
        mag2 = cmd.mag2 + np.random.randn(len(cmd.mag2)) * np.polyval(sigma2,cmd.mag2)
    return mag1, mag2

def cmdPlot(cmd,**kwargs):
    font = {'family' : 'serif',
    'color'  : 'black',
    'weight' : 'normal',
    'size'   : 14,
    }
    import matplotlib.pyplot as plt
    plt.scatter(cmd.color, cmd.mag2, marker='o', s=0.5, color='blue')
    xmin = min(cmd.color)-0.1 ; xmax = max(cmd.color)+0.1
    ymin = max(cmd.mag2)+0.1 ; ymax = min(cmd.mag2)-0.1
    if 'xrange' in kwargs.keys():
        xmin = kwargs['xrange'][0]
        xmax = kwargs['xrange'][1]
    if 'yrange' in kwargs.keys():
        ymin = kwargs['yrange'][0]
        ymax = kwargs['yrange'][1]
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xlabel(cmd.mag1label + ' - ' + cmd.mag2label,fontdict=font)
    plt.ylabel(cmd.mag2label,fontdict=font)
    plt.text(xmin+.1,ymax+.5,'q = {0:<g}'.format(cmd.q),fontdict=font)
    plt.show()

def magPlot(cmd,**kwargs):
    font = {'family' : 'serif',
    'color'  : 'black',
    'weight' : 'normal',
    'size'   : 14,
    }
    import matplotlib.pyplot as plt
    plt.scatter(cmd.mag1, cmd.mag2, marker='o', s=0.5, color='blue')
    xmin = min(cmd.mag1)-0.1 ; xmax = max(cmd.mag1)+0.1
    ymin = min(cmd.mag2)-0.1 ; ymax = max(cmd.mag2)+0.1
    if 'xrange' in kwargs.keys():
        xmin = kwargs['xrange'][0]
        xmax = kwargs['xrange'][1]
    if 'yrange' in kwargs.keys():
        ymin = kwargs['yrange'][0]
        ymax = kwargs['yrange'][1]
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xlabel(cmd.mag1label,fontdict=font)
    plt.ylabel(cmd.mag2label,fontdict=font)
    plt.text(xmin+.1,ymax-.1,'q = {0:<g}'.format(q),fontdict=font)
    plt.show()

if __name__ == "__main__":

    #Generate an isochrone object and interpolate to a uniformly-spaced mass array.

    myiso = DartmouthIsochrone(-2.0,0.4,14.0,'sdss')
    myiso.interpolate(dm=0.001,diagnose=False)
    myiso.has_interp()

    strmag1 = 'F110W'
    strmag2 = 'F160W'

    strmag1 = 'g'
    strmag2 = 'r'

    #plt.plot(myiso.data[strmag1]-myiso.data[strmag2],myiso.data[strmag2],lw=2)
    #plt.axis([-1,0,15,0])
    #plt.xlabel(strmag1 + '-' + strmag2)
    #plt.ylabel(strmag2)
    #plt.show()

    #Proceed to draw stars from distribution.
    #Will use rejection sampling as it not clear how to use inverse transformation
    #when binaries are included, except when

    #binary approach 1:
    # dN/dM: both primary and secondary drawn from full IMF distribution, then matched randomly
    # binary fraction, q binary systems: ntot = (1-q)*nsys + 2q*nsys, fraction of stars in binaries =
    # 2q*nsys / ntot = 2q*nsys / (1+q)*nsys , n single stars = ntot * (1-q)*nsys/(1+q)*nsys = (1-q)/(1+q)
    # 1 alt: change boundaries on Mmin, and Mmax.

    isomass = myiso.data['mass']

    alpha = 2.35
    fs = f_salpeter(isomass,alpha)
    fk = f_kroupa(isomass,1.35,1.7,alpha_3=2.30)
    Phi_s = np.cumsum(fs)
    Phi_s = Phi_s / max(Phi_s)
    Phi_k = np.cumsum(fk)
    Phi_k = Phi_k / max(Phi_k)

    #use salpeter
    f_Phiinv = interpolate.splrep(Phi_s,isomass)

    #Set binary fraction = number of binary systems
    q = 1.0
    nrequired = 80000

    #one way to select range of mags
    #another is based on observational constraints
    mass_min = 0.3
    mass_max = 0.8
    w = np.argmin(abs(isomass - mass_min))
    abs_mag_max = myiso.data[strmag2][w]
    w = np.argmin(abs(isomass - mass_max))
    abs_mag_min = myiso.data[strmag2][w]

    #do inverse transform sampling

    cmd = SyntheticCMD(myiso,strmag1,strmag2,abs_mag_min,abs_mag_max,nrequired,
        f_Phiinv,q=1.0)

    cmdPlot(cmd,yrange=[10,2])

    magsub1 = cmd.mag2[(cmd.mag2 >= abs_mag_min) & (cmd.mag2 <= abs_mag_max)]
    magsub2 = cmd.mag2[(cmd.mag2 >= abs_mag_min) & (cmd.mag2 <= abs_mag_max)]
    yval, edges = np.histogram(magsub2,bins=20)
    xval = 0.5 * (edges[1:] + edges[:-1])
    plt.bar(xval,yval,align='center',width=xval[1]-xval[0],color=None)
    plt.show()

    #dx = 0.2
    #cmd.cmdDensity(dx,dx)




    #plt.plot(Phi_s,mass,color='red',label='PL')
    #plt.plot(Phi_k,mass,color='blue',label='BkPL')
    #plt.axis([0,1,0,0.9])
    #plt.legend()
    #plt.show()
