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
from make_cmd_cloud import *

def plot_obs_sim(ax,xrange,yrange,simcolor,simmag,obscolor,obsmag,xlabel="",ylabel="",title=""):

	ax.plot(simcolor, simmag, marker='o',ms=2,color='blue',ls="None")
	ax.plot(obscolor, obsmag,marker='o',color='red',ls='None',ms=2,alpha=0.5)
	ax.set_xlim([xrange[0],xrange[1]])
	ax.set_ylim([yrange[0],yrange[1]])
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)

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
Hobs, xedges, yedges = np.histogram2d(colorarr,magarr,bins=[xedges,yedges],normed=False)
Hobs = Hobs / Hobs.max()

Xobs, Yobs = np.meshgrid(xedges, yedges)

#plot CMD
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(121)
ax1.plot(colorarr,magarr,marker='o',color='blue',ls='None',ms=2)
ax1.set_title('CMD')
ax1.set_xlabel(sysmag1+' - '+sysmag2)
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymax,ymin])

ax2 = fig.add_subplot(122)
imgobs = ax2.pcolormesh(Xobs, Yobs, Hobs.transpose(), cmap='Blues')
ax2.set_title('CMD Hess Diagram')
ax2.set_xlabel(sysmag1+' - '+sysmag2)
ax2.set_xlim([xmin,xmax])
ax2.set_ylim([ymax,ymin])
#plt.axis([xmin,xmax,ymax,ymin])
plt.colorbar(imgobs, ax=ax2)
#ax.set_aspect('equal')
plt.show()

#calculate uncertainties as function of magnitude for input to simulated data.

strmag1 = 'F606W'
strmag2 = 'F814W'

x1 = phot['F606W'][phot['F606Werr'] > 0.0]
x1err = phot['F606Werr'][phot['F606Werr'] > 0.0]
x2 = phot['F814W'][phot['F814Werr'] > 0.0]
x2err = phot['F814Werr'][phot['F814Werr'] > 0.0]

magsort1 = np.argsort(x1)
magsort2 = np.argsort(x2)
p1 = np.polyfit(x1[magsort1],x1err[magsort1],5,cov=False)
p2 = np.polyfit(x2[magsort2],x2err[magsort2],5,cov=False)
magarr1 = np.arange(20.,32.,.01)  
magarr2 = np.copy(magarr1)
magerrarr1_ = np.polyval(p1,magarr1)
#magerrarr1_[magarr1 <= phot['F606W'].min()] = phot['F606Werr'].min()
magerrarr2_ = np.polyval(p2,magarr2)
#magerrarr2_[magarr2 <= phot['F814W'].min()] = phot['F814Werr'].min()
plt.plot(x1,x1err,c='red',marker='o',ms=2,ls='None',alpha=0.3,label=strmag1)
plt.plot(x2,x2err,c='blue',marker='o',ms=2,ls='None',alpha=0.3,label=strmag2)
plt.plot(magarr1, magerrarr1_, ls='-', color='red',lw=3)
plt.plot(magarr2, magerrarr2_, ls='-', color='blue',lw=3)
plt.ylim([0.,0.6])
plt.xlabel("Mag")
plt.ylabel("Uncertainty")
plt.legend(loc=2)
plt.show()

#Create simulated data object

myiso = DartmouthIsochrone(-2.0,0.4,14.0,'acs')
myiso.interpolate(dm=0.001,diagnose=False)
myiso.has_interp()

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
q = 0.3
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
    f_Phiinv,q=q,modulus=dmod0)
simcolor = cmd.color
simmag = cmd.mag2

#Plot Simulated and true data

fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(221)

plot_obs_sim(ax1,[xmin,xmax],[ymax,ymin],simcolor,simmag,colorarr,magarr,
	xlabel=strmag1 + '-' + strmag2,ylabel=strmag2,title="Simulated Data")

mag1, mag2 = simulateScatter(cmd,p1,p2)

ax2 = fig.add_subplot(222)
plot_obs_sim(ax2,[xmin,xmax],[ymax,ymin],mag1-mag2,mag2,colorarr,magarr,
	xlabel=strmag1 + '-' + strmag2,ylabel=strmag2,title="Simulated Data w/Uncertainties")

xedges = np.linspace(xmin,xmax,nbinx)
yedges = np.linspace(ymin,ymax,nbiny)
Hsim, xedges, yedges = np.histogram2d(mag1-mag2,mag2,bins=[xedges,yedges],normed=False)
Hsim = Hsim / Hsim.max()

Xsim, Ysim = np.meshgrid(xedges, yedges)

ax3 = fig.add_subplot(223)
imgsim = ax3.pcolormesh(Xsim, Ysim, Hsim.transpose(), cmap='Blues')
ax3.set_title('Hess Map Simulated')
ax3.set_xlabel(sysmag1+' - '+sysmag2)
ax3.set_xlim([xmin,xmax])
ax3.set_ylim([ymax,ymin])
#plt.axis([xmin,xmax,ymax,ymin])
plt.colorbar(imgsim, ax=ax3)
#ax.set_aspect('equal')

ax4 = fig.add_subplot(224)
img = ax4.pcolormesh(Xobs, Yobs, Hobs.transpose(), cmap='Blues')
ax4.set_title('Hess Map Observations')
ax4.set_xlabel(sysmag1+' - '+sysmag2)
ax4.set_xlim([xmin,xmax])
ax4.set_ylim([ymax,ymin])
#plt.axis([xmin,xmax,ymax,ymin])
plt.colorbar(imgobs, ax=ax4)
#ax.set_aspect('equal')
plt.show()


#magsub1 = cmd.mag2[(cmd.mag2 >= abs_mag_min) & (cmd.mag2 <= abs_mag_max)]
#magsub2 = cmd.mag2[(cmd.mag2 >= abs_mag_min) & (cmd.mag2 <= abs_mag_max)]
#yval, edges = np.histogram(magsub2,bins=20)
#xval = 0.5 * (edges[1:] + edges[:-1])
#plt.bar(xval,yval,align='center',width=xval[1]-xval[0],color=None)
#plt.show()
