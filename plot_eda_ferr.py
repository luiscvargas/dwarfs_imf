import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')


f = open(getenv('ASTRO_DIR')+'/dwarfs_imf/'+'results/eda2.dat','r')
data = np.genfromtxt(f,comments='#',names=True,delimiter=',')
f.close()

nstar_arr = [1000,2000,5000]
#alpha_arr = [0.5,1.0,1.5,2.0,2.5,3.0]
#alpha_arr = [0.5,1.5,2.5]

nstar_arr = [2000]
alpha_arr = [2.5]
y2max_arr = [28.5,29.0,29.5,30.0]
ferr_arr = [0.1,1.0]
dferr_arr = [.1,.2]

plt.axis([2.00,3.5,28.0,30.5])
plt.xlabel(r"$\alpha$")
plt.ylabel(r"F814W_{max}")
plt.title(r"MC Tests, Salpeter")

color_arr = ['b','r','g','k','yellow','magenta'] 
color_arr = ['b','r','g'] 

for ialpha,alpha in enumerate(alpha_arr):

    for iferr,ferr in enumerate(ferr_arr):

        for iy2max,y2max in enumerate(y2max_arr):

            x = data[(data['alpha_in'] == alpha) & (data['y2max'] == y2max) & (data['ferr'] == ferr)]
            xmean = np.mean(x['alpha_out'])
            xstdev = np.std(x['alpha_out'])
            plt.scatter(x['alpha_out'],x['y2max'],marker='o',color=color_arr[iferr],alpha=0.5,s=4)
            plt.errorbar(xmean,y2max+dferr_arr[iferr],xerr=xstdev,yerr=0.0,color=color_arr[iferr],marker='s',markersize=8,elinewidth=2,capsize=2)

for ialpha,alpha in enumerate(alpha_arr): plt.plot([alpha,alpha],[28.0,30.5],color='k')

plt.show()

