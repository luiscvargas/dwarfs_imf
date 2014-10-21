import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')

imftype = 'kroupa'

#number of stars in mock sample: 
nstars_arr = [1000,2000]

#seed values - sets number of independentm mock samples
seed_arr = 5*np.arange(100) + 101

#depth of sample - maximum F814W magnitude
y2max_arr = [28.5,29.5,30.0]
#y2max_arr = [30.0]

#scaling for photometric uncertainties: 1.0 = error fnc of mag just as in real Herc data
ferr_arr = [1.0]

if imftype == 'salpeter':

    f = open('results/eda_salpeter.dat','r')
    dataall = np.genfromtxt(f,comments='#',names=True,delimiter=',')
    f.close() 
    title = r"MC Tests, Salpeter"
    paramname = r"$\alpha$"

elif imftype == 'chabrier':

    f = open('results/eda_chabrier.dat','r')
    dataall = np.genfromtxt(f,comments='#',names=True,delimiter=',')
    f.close()
    title = r"MC Tests, Chabrier"
    paramname = r"$M_{c}$"

elif imftype == 'kroupa':

    f = open('results/eda_kroupa.dat','r')
    dataall = np.genfromtxt(f,comments='#',names=True,delimiter=',')
    f.close()
    title = r"MC Tests, Kroupa"
    paramname = r"$\alpha_{1}$"

print dataall.dtype.names

plt.title(title)

color_arr = ['b','r','g','k','yellow','magenta'] 

#Cut out variables extraneous to plot
y2maxarr = np.unique(dataall['y2max'])
ferrarr = np.unique(dataall['ferr'])

print y2maxarr
print ferrarr

#####################################################################

data = dataall[(dataall['y2max'] == 29.5) & (dataall['ferr'] == 1.0) & (dataall['param_fit'] > -9.)]

#####define x and y labels for plotting

xlabel = 'param_in'
ylabel = 'nstars'
plt.xlabel(paramname)
plt.ylabel(r"Nstars")

xarr = np.unique(data[xlabel])
yarr = np.unique(data[ylabel])
dx = xarr.max()-xarr.min()
dy = yarr.max()-yarr.min()
xmin = xarr.min() - 0.5*(dx)
xmax = xarr.max() + 0.5*(dx)
ymin = yarr.min() - 0.5*(dy)
ymax = yarr.max() + 0.5*(dy)
plt.axis([xmin,xmax,ymin,ymax])

for ix,x in enumerate(xarr):

    for iy,y in enumerate(yarr):

        datasub = data[(data[xlabel] == x) & (data[ylabel] == y)]
        xmean = np.mean(datasub['param_fit'])
        xstdev = np.std(datasub['param_fit'])
        plt.scatter(datasub['param_fit'],datasub[ylabel],marker='o',color=color_arr[ix],alpha=0.25,s=2)
        plt.errorbar(xmean,y+dy*.05,xerr=xstdev,yerr=0.0,color=color_arr[ix],marker='s',markersize=8,elinewidth=1.5,capsize=2)

for ix,x in enumerate(xarr): plt.plot([x,x],[0,10000],color=color_arr[ix],ls='--')

plt.show()

#####################################################################

data = dataall[(dataall['nstars'] == 2000) & (dataall['ferr'] == 1.0) & (dataall['param_fit'] > -9.)]

#####define x and y labels for plotting

xlabel = 'param_in'
ylabel = 'y2max'
plt.xlabel(paramname)
plt.ylabel(r"F814W_{max}")

xarr = np.unique(data[xlabel])
yarr = np.unique(data[ylabel])
dx = xarr.max()-xarr.min()
dy = yarr.max()-yarr.min()
xmin = xarr.min() - 0.5*(dx)
xmax = xarr.max() + 0.5*(dx)
ymin = yarr.min() - 0.5*(dy)
ymax = yarr.max() + 0.5*(dy)
plt.axis([xmin,xmax,ymin,ymax])

for ix,x in enumerate(xarr):

    for iy,y in enumerate(yarr):

        datasub = data[(data[xlabel] == x) & (data[ylabel] == y)]
        if len(datasub) > 5:
            xmean = np.mean(datasub['param_fit'])
            xstdev = np.std(datasub['param_fit'])
            plt.scatter(datasub['param_fit'],datasub[ylabel],marker='o',color=color_arr[ix],alpha=0.25,s=2)
            plt.errorbar(xmean,y+dy*.05,xerr=xstdev,yerr=0.0,color=color_arr[ix],marker='s',markersize=8,elinewidth=1.5,capsize=2)

for ix,x in enumerate(xarr): plt.plot([x,x],[0,50],color=color_arr[ix],ls='--')

plt.show()

