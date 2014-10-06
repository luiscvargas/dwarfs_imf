import numpy as np
import matplotlib.pyplot as plt
from mywrangle import *

#first do so for optical (V,I bands)
yy=np.genfromtxt(open('iso/age14feh200afe00.yy.iso','r'),names=True,comments='#',skip_header=1)
da=np.genfromtxt(open('iso/fehm20afep0.UBVRIJHKsKp.darth.iso','r'),names=True,comments='#',skip_header=1)
pa=np.genfromtxt(open('iso/age135feh200afe00.parsec.opt.iso','r'),names=True,comments='#',skip_header=12)

dmod = 18.22

plt.figure(1)

ax = plt.subplot(2,1,1)
line1, = ax.plot(yy['mass'],yy['V']+dmod,color='blue',lw=1.5,label='Yale')
line2, = ax.plot(da['mass'],da['V']+dmod,color='red',lw=1.5,label='Darth')
line3, = ax.plot(pa['mass'],pa['V']+dmod,color='green',lw=1.5,label='Parsec')
ax.set_xlabel('Mass') ; ax.set_ylabel('Johnson V + Dmod(CB)')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,loc=3)
ax.axis([.1,.72,23,28])
ax = plt.subplot(2,1,2)
ax.plot(yy['mass'],yy['V']-yy['VI']+dmod,color='blue',lw=1.5)
ax.plot(da['mass'],da['I']+dmod,color='red',lw=1.5)
ax.plot(pa['mass'],pa['I']+dmod,color='green',lw=1.5)
ax.set_xlabel('Mass') ; ax.set_ylabel('Cousins I + Dmod(CB)')
ax.axis([.15,.65,23,28])

#now for the WFC3 bands F110W, F160W (VegaMag?)
da = read_iso_darth(14.0,-2.0,0.0,'wfc3')
da['F110W'] = da['F110W'] - (28.4401 - 26.0628)  #STMAG-->VEGA
da['F160W'] = da['F160W'] - (28.1875 - 24.6949)  #STMAG-->VEGA
pa=np.genfromtxt(open('iso/age135feh200afe00.parsec.iso','r'),names=True,comments='#',skip_header=12)

plt.figure(2)

ax = plt.subplot(2,1,1)
line2, = ax.plot(da['mass'],da['F110W']+dmod,color='red',lw=1.5,label='Darth')
line3, = ax.plot(pa['mass'],pa['F110W']+dmod,color='green',lw=1.5,label='Parsec')
ax.set_xlabel('Mass') ; ax.set_ylabel('WFC3 F110W + Dmod(CB)')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,loc=3)
ax.axis([.1,.72,23,28])
ax = plt.subplot(2,1,2)
ax.plot(da['mass'],da['F160W']+dmod,color='red',lw=1.5)
ax.plot(pa['mass'],pa['F160W']+dmod,color='green',lw=1.5)
ax.set_xlabel('Mass') ; ax.set_ylabel('WFC3 F160W + Dmod(CB)')
ax.axis([.15,.65,23,28])

plt.show()

