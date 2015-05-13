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

#Set env variables for latex-style plotting
rc('text', usetex=True)
rc('font', family='serif')

#Use Herc data to model error as a function of magnitude
system = 'acs'
sysmag1   = 'F606W'
sysmag2   = 'F814W'
isoage = 14.0
isofeh = -2.5
isoafe =  0.4
dmod0  = 20.63  #dmod to Hercules

#read-in Hercules data
phot = read_phot('Herc',system,sysmag1,sysmag2)
colorarr = phot['F606W'] - phot['F814W']
magarr = phot['F814W']

xmin = -1.0
xmax =  0.5
ymin = 22
ymax = 30
binsize = 0.04

nbinx = int((xmax - xmin) / binsize)
nbiny = int((ymax - ymin) / binsize)

#calculate heat map number counts
xedges = np.linspace(xmin,xmax,nbinx)
yedges = np.linspace(ymin,ymax,nbiny)
H, xedges, yedges = np.histogram2d(colorarr,magarr,bins=[xedges,yedges],normed=False)
H = H / H.max()

X, Y = np.meshgrid(xedges, yedges)

#plot CMD
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(121)
ax1.plot(colorarr,magarr,marker='o',color='blue',ls='None',ms=2)
ax1.set_title('CMD')
ax1.set_xlabel(sysmag1+' - '+sysmag2)
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymax,ymin])

ax2 = fig.add_subplot(122)
img = ax2.pcolormesh(X, Y, H.transpose(), cmap='Blues')
ax2.set_title('CMD Hess Diagram')
ax2.set_xlabel(sysmag1+' - '+sysmag2)
ax2.set_xlim([xmin,xmax])
ax2.set_ylim([ymax,ymin])
#plt.axis([xmin,xmax,ymax,ymin])
plt.colorbar(img, ax=ax2)
#ax.set_aspect('equal')
plt.show()

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