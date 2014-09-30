import numpy as np
import matplotlib.pyplot as plt
from mywrangle import *

#first do so for optical (V,I bands)
yy=np.genfromtxt(open('iso/age14feh200afe00.yy.iso','r'),names=True,comments='#',skip_header=1)
da=np.genfromtxt(open('iso/fehm20afep0.UBVRIJHKsKp.darth.iso','r'),names=True,comments='#',skip_header=1)
pa=np.genfromtxt(open('iso/age135feh200afe00.parsec.opt.iso','r'),names=True,comments='#',skip_header=12)

print da.dtype.names
print pa.dtype.names

#now for the WFC3 bands F110W, F160W (VegaMag?)
da = read_iso_darth(14.0,-2.0,0.0,'wfc3')
da['F110W'] = da['F110W'] - (28.4401 - 26.0628)  #STMAG-->VEGA
da['F160W'] = da['F160W'] - (28.1875 - 24.6949)  #STMAG-->VEGA
pa=np.genfromtxt(open('iso/age135feh200afe00.parsec.iso','r'),names=True,comments='#',skip_header=12)

print da.dtype.names
print pa.dtype.names

