import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mywrangle import *

def simulate_cmd(nstars,isoage,isofeh,isoafe,dist_mod,inmagarr1,inmagerrarr1,inmagarr2,inmagerrarr2,system,sysmag1,sysmag2,**kwargs):

   rejection_sampling = 0

   testing = 1 #testing = 1 in stand alone test_simulate_cmd.py code
   rc('text', usetex=True)
   rc('font', family='serif')

   if 'testing' in kwargs.keys():
       if kwargs['testing'] == True: 
           testing = 1
           rc('text', usetex=True)
           rc('font', family='serif')


   if 'imftype' not in kwargs.keys(): raise SystemExit

   print "Warning! This program will generate synthetic CMDs"
   print "only for the MS region where the MF = IMF."""

   #raw_input("Press any key to continue>>>")

   #Now import isochrone of given age, [Fe/H], [a/Fe], making desired mass cuts for fitting
   iso = read_iso_darth(isoage,isofeh,isoafe,system)
   #iso = read_iso_darth(isoage,isofeh,isoafe,system,mass_min=mass_min,mass_max=mass_max)

   isomass = iso['mass'] 
   isocol = iso[sysmag1] - iso[sysmag2] 
   isomag = iso[sysmag2] + dist_mod

   if system == 'wfpc2':
       col_name = r'$m_{606w} - m_{814w}$' ; mag_name = r'$m_{814w}$'
   elif system == 'sdss':
       col_name = r'$(g - r)_0$' ; mag_name = r'$r_0$'
   elif system == 'acs':
       col_name = r'$m_{606w} - m_{814w}$' ; mag_name = r'$m_{814w}$'
   else:
       pass

   if 'mass_min' in kwargs.keys():
       mass_min = kwargs['mass_min']
   else: mass_min = 0.05

   if 'mass_max' in kwargs.keys():
       mass_max = kwargs['mass_max']
   else: mass_max = 0.80

   if mass_max <= mass_min: raise SystemExit
   if mass_min <= iso['mass'].min(): mass_min = iso['mass'].min()
   if mass_max >= iso['mass'].max(): mass_max = iso['mass'].max()

   if kwargs['imftype'] == 'salpeter':
       if 'alpha' not in kwargs.keys():
           print "Error: alpha not specified for Salpeter function" 
           raise SystemExit
       alpha_ = kwargs['alpha']

   elif kwargs['imftype'] == 'chabrier':
       if 'mc' not in kwargs.keys():
           print "Error: M_c (kwarg=mc) not specified for Chabrier function" 
           raise SystemExit
       elif 'sigmac' not in kwargs.keys():
           print "Error: sigma_c (kwarg=sigmac) not specified for Chabrier function" 
           raise SystemExit
       pass
       mc_ = kwargs['mc']
       sigmac_ = kwargs['sigmac']

   elif kwargs['imftype'] == 'kroupa':
       if 'alpha1' not in kwargs.keys():
           print "Error: alpha1 (kwarg=alpha1) (slope for 0.08<M<0.5) not specified for Kroupa function" 
           raise SystemExit
       elif 'alpha2' not in kwargs.keys():
           print "Error: alpha1 (kwarg=alpha1) (slope for 0.5<M<1.0) not specified for Kroupa function" 
           raise SystemExit
       pass
       alpha1_ = kwargs['alpha1']
       alpha2_ = kwargs['alpha2']

   else:
       print "Need to specify either Salpeter or Chabrier and their respective params!"
       raise SystemExit
       

   #Find min and max in dN/dM in order to set limits in y-axis for random number
   #generator, in practice do not need to multiply by d(mass) within f_salpeter,
   #f_chabrier, f_kroupa here, as all we want is a relative number of stars per dM,
   #and dM is constant by construction.

   xdum = np.arange(mass_min,mass_max,0.0001)
   if kwargs['imftype'] == 'salpeter':
       ydum = f_salpeter(xdum,mass_min,mass_max,alpha_)
   elif kwargs['imftype'] == 'chabrier':
       ydum = f_chabrier(xdum,mass_min,mass_max,mc_,sigmac_)
   elif kwargs['imftype'] == 'kroupa':
       ydum = f_kroupa(xdum,mass_min,mass_max,alpha1_,alpha2_)

   ydum_cumul = np.cumsum(ydum)

   ydum_cumul = ydum_cumul / max(ydum_cumul)

   #Invert cumulative function

   fcumul = interpolate.splrep(xdum,ydum_cumul)
   gcumul = interpolate.splrep(ydum_cumul,xdum)

   xtmp_fwd = np.arange(mass_min,mass_max,0.01)
   ytmp_fwd = interpolate.splev(xtmp_fwd,fcumul)
   xtmp_inv = np.arange(0.,1.,.01)
   ytmp_inv = interpolate.splev(xtmp_inv,gcumul)
   
   if testing == 1:
       plt.subplot(2,2,1)
       plt.plot(xdum,ydum,color='b',ls='-')
       plt.axis([mass_min,mass_max,0.0,ydum.max()])
       plt.xlabel(r"$M$") ; plt.ylabel(r"$dN/dM$")
       plt.subplot(2,2,2)
       plt.plot(xdum,ydum_cumul,color='b',ls='-')
       plt.plot(xtmp_fwd,ytmp_fwd,'go')
       plt.axis([mass_min,mass_max,0.0,1.0])
       plt.xlabel(r"$M$") ; plt.ylabel(r"$Integral dN/dM$")
       plt.subplot(2,2,3)
       plt.plot(ydum_cumul,xdum,color='b',ls='-')
       plt.plot(xtmp_inv,ytmp_inv,'go')
       plt.axis([0.0,1.0,mass_min,mass_max])
       plt.xlabel(r"$M$") ; plt.ylabel(r"$Integral dN/dM$")
       plt.subplot(2,2,4)
       plt.loglog(xdum,ydum,color='b',ls='-',basex=10,basey=10)
       plt.axis([mass_min,mass_max,ydum.min(),ydum.max()])
       plt.xlabel(r"log\,$M$") ; plt.ylabel(r"log\,$dN/dM$")
       #plt.savefig(os.getenv('HOME')+'/Desktop/mass_dndm_function.png',bbox_inches='tight')
       plt.show()

   #Generate random data points in 2D and stop when 100 points are within 
   #desired region

   #First generate very large array in x and y, so that hopefully there will be at least nstars
   #that fall under dN/dM - M relation.
   if 'start_seed' in kwargs.keys():
       np.random.seed(seed=kwargs['start_seed'])
   else:
       np.random.seed(seed=12345)

   ############Begin rejection sampling

   if (rejection_sampling == 1 or testing == 1):

       #Define limits of masses as lowest isochrone mass and highest isochrone mass *within* the mass cuts
       #specified as input args. If I use instead those input args directly, some points will be slightly
       #outside of range of isochrone masses, as mass_min < M_iso < mass_max is set in read_darth_iso, 
       #causing the spline interpolator to crash later on.

       xrantmparr = np.random.random_sample(nstars*15) * (mass_max - mass_min) + mass_min   
       yrantmparr = np.random.random_sample(nstars*15) * 1.02*ydum.max() 

       xranarr_rej = np.arange(nstars)*0.0
       yranarr_rej = np.arange(nstars)*0.0

       #Now find the pairs (x,y) of simulated data that fall under envelope of dN/dM - M relation.
       count = 0
       for i,xrantmp in enumerate(xrantmparr):
           #print count
           if count == nstars: break
           idx = np.abs(xdum - xrantmp).argmin()
           #if (yrantmparr[i] <= ydum[idx]) & (xdum[idx] > iso['mass'].min()) & (xdum[idx] < iso['mass'].max()):
           #should xdum[idx] go here, or should xrantmp? 
           if (yrantmparr[i] <= ydum[idx]) & (xdum[idx] >= mass_min) & (xdum[idx] < mass_max):
               xranarr_rej[count] = xrantmparr[i]
               yranarr_rej[count] = yrantmparr[i]
               #print count,xrantmparr[i],yrantmparr[i],xranarr[count],yranarr[count]
               count += 1
           else:
               pass

       if len(yranarr_rej[yranarr_rej > 0.0]) < nstars:
           print "Need to generate more samples!"
           raise SystemExit

   ############End rejection sampling

   ##Begin transformation method

   if (rejection_sampling == 0 or testing == 1):
       #note that we now specify Y = U[0,1] instead of X in case of transformation method
       #gcumul is inverse of CDF of dN/dM integrated from M_min to M for a given M. 
       yranarr_transform = np.random.random_sample(nstars)
       xranarr_transform = interpolate.splev(yranarr_transform,gcumul)

   if rejection_sampling == 0: 
       mc_mass_arr_pri = xranarr_transform
   elif rejection_sampling == 1:
       mc_mass_arr_pri = xranarr_rej
   else: 
       print "choose rejection sampling or transformation method!"
       raise SystemExit

   if testing == 1:
       plt.subplot(2,2,1)
       plt.scatter(xrantmparr,yrantmparr,s=1,c='k',marker='.')
       plt.scatter(xranarr_rej,yranarr_rej,s=6,c='r',marker='o')
       plt.plot(xdum,ydum,color='b',ls='-')
       plt.axis([mass_min,mass_max,0.0,ydum.max()])
       plt.xlabel(r"$M$") ; plt.ylabel(r"$dN/dM$")
       plt.subplot(2,2,2)
       plt.scatter(xranarr_rej,yranarr_rej,s=6,c='b',marker='o')
       plt.plot(xdum,ydum,color='b',ls='-')
       plt.axis([mass_min,mass_max,0.0,ydum.max()])
       plt.xlabel(r"$M$") ; plt.ylabel(r"$dN/dM$")
       plt.subplot(2,2,3)
       n_rej , mass_rej = np.histogram(xranarr_rej,bins=20)
       n_transform , mass_transform = np.histogram(xranarr_transform,bins=20)
       mass_rej = (mass_rej[1:] + mass_rej[:-1])/2.
       mass_transform = (mass_transform[1:] + mass_transform[:-1])/2.
       plt.plot(mass_rej,n_rej,c='b',marker='s')
       plt.plot(mass_transform,n_transform,c='g',marker='o')
       plt.axis([mass_min,mass_max,0.0,1.1*n_rej.max()])
       plt.xlabel(r"$M$") ; plt.ylabel(r"$dN$")
       #plt.subplot(2,2,4)
       plt.show()

   #######Add binary fraction

   if 'fb' in kwargs.keys():
       fb = kwargs['fb']
   else:
       fb = -1.0 

   if fb <= 0.0:
       mc_mass_arr_tot = mc_mass_arr_pri

   elif fb > 0.0: 

       #create a set of secondary masses and total masses
  
       mc_mass_arr_sec = np.zeros(len(mc_mass_arr_pri))
       mc_mass_arr_tot = np.zeros(len(mc_mass_arr_pri))

       if 'alpha_sec' in kwargs.keys():
           alpha_sec = kwargs['alpha_sec']
       else: 
           print 'need to specify dN/dM2 in case of non-zero binary fraction'
           raise SystemExit

       fb_arr = np.random.random(len(mc_mass_arr_pri))

       mass_min_sec = iso['mass'].min()  #mass_min_sec must be > mass_min_iso in order to find a matching magnitude

       for imass,mass_pri in enumerate(mc_mass_arr_pri):
 
           if mass_pri <= mass_min_sec + 0.005: 
               print "mass_pri = ",mass_pri
               print "mass of primary too close to lower isochrone mass limit"
               print "no secondary assigned", "fb random value = ",fb_arr[imass]
               raise SystemExit

           if fb_arr[imass] <= fb:
 
               xdum = np.arange(mass_min_sec,mass_pri,0.0001)
               ydum = xdum ** (-1.0*alpha_sec)
               ydum = ydum / max(ydum)

               while 1:
               
                   #Calculate random deviates within range of masses with non-zero dN/dM2, 
                   #max mass is mass of EACH respective primary, NOT maximum mass of all stars 
                   #in sample.
                   xtmp = np.random.random_sample(1) * (mass_pri - mass_min_sec) + mass_min_sec
                   ytmp = np.random.random_sample(1) * 1.02 * max(ydum)
                   w = np.argmin(abs(xtmp - xdum))
                   if ytmp <= ydum[w]: break

               #Assign non-zero masses to secondary mass array, and add masses for total mass array
               mc_mass_arr_sec[imass] = xtmp
               mc_mass_arr_tot[imass] = xtmp + mass_pri

           else:
               mc_mass_arr_sec[imass] = -1.0
               mc_mass_arr_tot[imass] = mass_pri
   

       if testing == 1: 
           #plot distribution of secondary to primary masses - title has fraction of binary systems to total # of systems
           plt.subplot(1,3,1) 
           qarr = mc_mass_arr_sec / mc_mass_arr_pri
           frac, qbins = np.histogram(qarr[qarr > 0.0],bins=20)
           n_single = len(qarr[qarr <= 0.0])
           plt.title(r"Input fb = "+str(fb)+" ; Input alpha sec = "+str(alpha_sec))
           plt.axis([0.0,1.0,0.0,1.1*max(frac)])
           plt.xlabel(r'$q$ = $M_{2}/M_{1}$')
           plt.ylabel(r'N')
           plt.text(0.05,0.9*max(frac),"Output Binary = "+"{0:<5.3f}".format((float(len(mc_mass_arr_pri)-n_single))/float(len(mc_mass_arr_pri))))
           plt.bar(qbins[:-1],frac,qbins[1]-qbins[0],edgecolor='k')
           plt.subplot(1,3,2) 
           n_sec_bins, mass_sec_bins = np.histogram(mc_mass_arr_sec[qarr > 0.0],bins=np.arange(0.,1.,0.05))
           n_pri_bins, mass_pri_bins = np.histogram(mc_mass_arr_pri[qarr > 0.0],bins=np.arange(0.,1.,0.05))
           plt.bar(mass_pri_bins[:-1],n_pri_bins,mass_pri_bins[1]-mass_pri_bins[0],edgecolor='k',alpha=0.5,color='r',label='$M_{1}$')
           plt.bar(mass_sec_bins[:-1],n_sec_bins,mass_pri_bins[1]-mass_pri_bins[0],edgecolor='k',alpha=0.5,color='b',label='$M_{2}$')
           plt.legend(loc='upper right')
           plt.xlabel(r"$M_{1},\, M_{2}$")
           plt.subplot(1,3,3) 
           plt.scatter(mc_mass_arr_pri[qarr > 0.0],mc_mass_arr_sec[qarr > 0.0],marker='o',s=2,color='green')
           plt.xlabel(r"$M_{1}$")
           plt.ylabel(r"$M_{2}$")
           plt.show()

   ##########Begin assignment of magnitudes to individual mock stars (include effect of binaries)

   #Interpolate isochrone magnitude-mass relation
   isort = np.argsort(iso['mass'])  #! argsort = returns indices for sorted array, sort=returns sorted array
   #if testing == 1:
   #    plt.plot(iso['mass'][isort],iso[sysmag2][isort]+dist_mod,'b.',ls='--')
   #    plt.show()
   f1 = interpolate.splrep(iso['mass'][isort],iso[sysmag1][isort]+dist_mod)
   f2 = interpolate.splrep(iso['mass'][isort],iso[sysmag2][isort]+dist_mod)

   #Assign magnitudes to individual components (primary+secondary)

   #Assign magnitudes to each star based on their mass and the mass-magnitude relation calculated above.
   #Get magnitudes for primary component
   mag1ranarr_pri_0 = interpolate.splev(mc_mass_arr_pri,f1)
   mag2ranarr_pri_0 = interpolate.splev(mc_mass_arr_pri,f2)  #band 2 = for system=wfpc2
   der_mag2ranarr_0 = interpolate.splev(mc_mass_arr_pri,f2,der=1)  #band 2 = for system=wfpc2
   if fb > 0:
       #Get magnitudes for secondary component if fb > 0, and then compute magnitudes combining both
       #components
       #Only do this for the fraction of stars with non-zero secondary magnitudes
       mag1ranarr_sec_0 = np.zeros(len(mc_mass_arr_sec))
       mag2ranarr_sec_0 = np.zeros(len(mc_mass_arr_sec))
       mag1ranarr_0 = np.zeros(len(mc_mass_arr_sec))
       mag2ranarr_0 = np.zeros(len(mc_mass_arr_sec))
       der_mag2ranarr__sec_0 = np.zeros(len(mc_mass_arr_sec))   #band 2 = for system=wfpc2
       mag1ranarr_sec_0[mc_mass_arr_sec > 0.0] = interpolate.splev(mc_mass_arr_sec[mc_mass_arr_sec > 0.0],f1)
       mag2ranarr_sec_0[mc_mass_arr_sec > 0.0] = interpolate.splev(mc_mass_arr_sec[mc_mass_arr_sec > 0.0],f2)  
             #band 2 = for system=wfpc2
       der_mag2ranarr__sec_0[mc_mass_arr_sec > 0.0] = interpolate.splev(mc_mass_arr_sec[mc_mass_arr_sec > 0.0],f2,der=1)  #band 2 = for system=wfpc2
       mag1ranarr_0[mc_mass_arr_sec <= 0.0] = mag1ranarr_pri_0[mc_mass_arr_sec <= 0.0]
       mag2ranarr_0[mc_mass_arr_sec <= 0.0] = mag2ranarr_pri_0[mc_mass_arr_sec <= 0.0]
       gamma1 = 1. + 10.**(0.4*(mag1ranarr_pri_0[mc_mass_arr_sec > 0.0] - mag1ranarr_sec_0[mc_mass_arr_sec > 0.0]))
       gamma2 = 1. + 10.**(0.4*(mag2ranarr_pri_0[mc_mass_arr_sec > 0.0] - mag2ranarr_sec_0[mc_mass_arr_sec > 0.0]))
       mag1ranarr_0[mc_mass_arr_sec > 0.0] = mag1ranarr_pri_0[mc_mass_arr_sec > 0.0] - 2.5*np.log10(gamma1)
       mag2ranarr_0[mc_mass_arr_sec > 0.0] = mag2ranarr_pri_0[mc_mass_arr_sec > 0.0] - 2.5*np.log10(gamma2)
   elif fb <=0:
       #Assign primary component magnitude to "total", as no secondary component present  
       mag1ranarr_0 = mag1ranarr_pri_0 
       mag2ranarr_0 = mag2ranarr_pri_0 

       print min(mag1ranarr_pri_0),max(mag1ranarr_pri_0)
       #raise SystemExit

   if testing == 1: 
       plt.subplot(2,2,1)
       #plt.scatter(mag1ranarr_pri_0,mag1ranarr_sec_0,marker='o',s=2,c='k')
       plt.scatter(mag1ranarr_pri_0[mc_mass_arr_sec > 0.0],mag1ranarr_sec_0[mc_mass_arr_sec > 0.0],marker='o',s=3.5,c='b')
       plt.xlabel(r"$mag_{1} (Primary)$"); plt.ylabel(r"$mag_{1} (Secondary)$")
       plt.axis([min(mag1ranarr_pri_0)-.5,max(mag1ranarr_pri_0)+.5,min(mag1ranarr_pri_0)-.5,max(mag1ranarr_pri_0)+4.])
       plt.subplot(2,2,2)
       #plt.scatter(mag2ranarr_pri_0,mag2ranarr_sec_0,marker='o',s=2,c='k')
       plt.scatter(mag2ranarr_pri_0[mc_mass_arr_sec > 0.0],mag2ranarr_sec_0[mc_mass_arr_sec > 0.0],marker='o',s=3.5,c='r')
       plt.xlabel(r"$mag_{2} (Primary)$"); plt.ylabel(r"$mag_{2} (Secondary)$")
       plt.axis([min(mag2ranarr_pri_0)-.5,max(mag2ranarr_pri_0)+.5,min(mag2ranarr_pri_0)-.5,max(mag2ranarr_pri_0)+4.])
       plt.subplot(2,2,3)
       plt.scatter(mag1ranarr_pri_0,mag1ranarr_0-mag1ranarr_pri_0,marker='o',s=2,c='g')
       plt.scatter(mag1ranarr_pri_0[mc_mass_arr_sec > 0.0],mag1ranarr_0[mc_mass_arr_sec > 0.0]-mag1ranarr_pri_0[mc_mass_arr_sec > 0.0],marker='o',s=2,c='r')
       plt.xlabel(r"$mag_{1} (Primary)$"); plt.ylabel(r"$mag_{1} (Combined) - mag_{1} (Primary)$")
       plt.axis([min(mag1ranarr_pri_0)-.5,max(mag1ranarr_pri_0)+.5,-2,1])
       plt.subplot(2,2,4)
       plt.scatter(mag2ranarr_pri_0,mag2ranarr_0-mag2ranarr_pri_0,marker='o',s=2,c='g')
       plt.scatter(mag2ranarr_pri_0[mc_mass_arr_sec > 0.0],mag2ranarr_0[mc_mass_arr_sec > 0.0]-mag2ranarr_pri_0[mc_mass_arr_sec > 0.0],marker='o',s=2,c='r')
       plt.xlabel(r"$mag_{2} (Primary)$"); plt.ylabel(r"$mag_{2} (Combined) - mag_{2} (Primary)$")
       plt.axis([min(mag2ranarr_pri_0)-.5,max(mag2ranarr_pri_0)+.5,-2,1])
       plt.show()

   colorranarr_0  = mag1ranarr_0 - mag2ranarr_0

   #Initialize data magnitude arrays which will include photometric uncertainties.
   mag1ranarr = np.arange(len(mag1ranarr_0))*0.0
   mag2ranarr = np.arange(len(mag1ranarr_0))*0.0
   mag1ranerrarr = np.arange(len(mag1ranarr_0))*0.0
   mag2ranerrarr = np.arange(len(mag1ranarr_0))*0.0

   #Based on mag-errmag relation from input args, assign random Gaussian deviates to each "star".
   for i,imag in enumerate(mag1ranarr_0):
       idx = np.abs(imag - inmagarr1).argmin()
       mag1ranarr[i] = imag + inmagerrarr1[idx]*np.random.normal()
       mag1ranerrarr[i] = inmagerrarr1[idx]
   for i,imag in enumerate(mag2ranarr_0):
       idx = np.abs(imag - inmagarr2).argmin()
       mag2ranarr[i] = imag + inmagerrarr2[idx]*np.random.normal()
       mag2ranerrarr[i] = inmagerrarr2[idx]

   colorranarr = mag1ranarr - mag2ranarr

   if testing == 1:   
       plt.subplot(3,2,1) 
       plt.plot(iso[sysmag2]+dist_mod,iso['mass'],ls='-',color='blue')  #without [isort]
       plt.scatter(mag2ranarr,mc_mass_arr_pri,s=1,marker='o',color='green')
       plt.scatter(mag2ranarr,mc_mass_arr_tot,s=1,marker='.',color='red')
       plt.axis([24.5,mag2ranarr.max(),mc_mass_arr_tot.min()-.02,mc_mass_arr_tot.max()+.02])
       plt.xlabel(r"$F814W$")
       plt.ylabel(r"$M_{tot}$")

       plt.subplot(3,2,2) 
       plt.scatter(mag2ranarr_0,der_mag2ranarr_0,color='green',marker='.',s=1)
       plt.axis([24.5,mag2ranarr.max(),-40,der_mag2ranarr_0.max()+5])
       plt.xlabel(r"$F814W$")

       plt.subplot(3,2,3) 
       plt.plot(isocol,isomag,ls='-',color='red',lw=2)
       plt.xlabel(r"$F606W-F814W$")
       plt.ylabel(r"$F814W$")
       plt.scatter(colorranarr,mag2ranarr,marker='o',s=3,color='b')
       plt.axis([isocol.min()-.25,isocol.max()+.25,dist_mod+12,dist_mod-2])

       plt.subplot(3,2,4) 
       nbins = int((mag2ranarr.max()-mag2ranarr.min())/0.25)
       n_r , rhist = np.histogram(mag2ranarr,bins=nbins)
       plt.bar(rhist[:-1],n_r,rhist[1]-rhist[0],edgecolor='k')
       plt.xlabel(r'$'+sysmag2+'$')
       plt.ylabel(r'$dN$')
       plt.axis([24.5,mag2ranarr.max(),0,1.1*max(n_r)])
       
       ax = plt.subplot(3,2,5) 
       nbins = int((mag2ranarr_0.max()-mag2ranarr_0.min())/0.25)
       n_r , rhist = np.histogram(mag2ranarr_0,bins=nbins)
       rhist_err = np.sqrt(n_r)
       ax.set_yscale("log", nonposy='clip')
       #ax.set_xscale("log", nonposx='clip')
       plt.errorbar(rhist[:-1],n_r,yerr=rhist_err,color='k',marker='o',markersize=1)
       plt.xlabel(r'$'+sysmag2+'$\,(zero scatter)')
       plt.ylabel(r'$dN$')
       plt.axis([24.5,mag2ranarr.max(),.01*max(n_r),2*max(n_r)])

       ax = plt.subplot(3,2,6) 
       nbins = int((mc_mass_arr_tot.max()-mc_mass_arr_tot.min())/0.025)
       n_r , rhist = np.histogram(mc_mass_arr_tot,bins=nbins)
       rhist_err = np.sqrt(n_r)
       ax.set_yscale("log", nonposy='clip')
       ax.set_xscale("log", nonposx='clip')
       plt.errorbar(rhist[:-1],n_r,yerr=rhist_err,color='k',marker='o',markersize=1)
       plt.xlabel(r'log\,M_{tot}')
       plt.ylabel(r'$dN$')
       plt.axis([mc_mass_arr_tot.min(),mc_mass_arr_tot.max(),.2*max(n_r),1.5*max(n_r)])
       plt.plot([0.5,0.5],[.2*max(n_r),1.5*max(n_r)],color='g',ls='-.')
       plt.plot([0.6,0.6],[.2*max(n_r),1.5*max(n_r)],color='magenta',ls='-.')
       plt.plot([0.4,0.4],[.2*max(n_r),1.5*max(n_r)],color='orange',ls='-.')
       plt.show()


   #Now package data into structure numpy array just as real photometric data 

   dtypes_simdata=[('ra','f8'),('dec','f8'),('covar','f8'),('color','f8'),('colorerr','f8'),('clip','f4')]

   #Simulated data right now consists of magnitudes and magnitude errors, where the latter is set 
   #by the input mag-magerr relation arguments to simulate_cmd() module.
   #perhaps later can include RA, Dec if want to model spatial variation.
   if system == 'wfpc2':
       dtypes = [('F555W','f8'),('F606W','f8'),('F814W','f8'),('F555Werr','f8'),('F606Werr','f8'),('F814Werr','f8')]
   elif system == 'wfc3':
       dtypes = [('F110W','f8'),('F160W','f8'),('F555W','f8'),('F606W','f8'),('F814W','f8'),
                 ('F110Werr','f8'),('F160Werr','f8'),('F555Werr','f8'),('F606Werr','f8'),('F814Werr','f8')]
   elif system == 'acs':
       dtypes = [('F555W','f8'),('F606W','f8'),('F814W','f8'),('F555Werr','f8'),('F606Werr','f8'),('F814Werr','f8')]
   elif system == 'sdss':
       dtypes = [('u','f8'),('g','f8'),('r','f8'),('i','f8'),('z','f8'),('uerr','f8'),('gerr','f8'),('rerr','f8'),('gerr','f8'),('rerr','f8')]
   elif system == 'cfht':
       dtypes = [('u','f8'),('g','f8'),('r','f8'),('i','f8'),('z','f8'),('uerr','f8'),('gerr','f8'),('rerr','f8'),('gerr','f8'),('rerr','f8')]

   dtypes = dtypes_simdata + dtypes

   simdata = np.zeros( (nstars,), dtype=dtypes )

   simdata[sysmag1] = mag1ranarr
   simdata[sysmag2] = mag2ranarr
   simdata[sysmag1+'err'] = mag1ranerrarr
   simdata[sysmag2+'err'] = mag2ranerrarr

   simdata['clip'] = 0  #assume cov(g,r) = 0.0 for now 
   simdata['covar'] = 0.0  #assume cov(g,r) = 0.0 for now 
   simdata['color'] = simdata[sysmag1] - simdata[sysmag2]
   simdata['colorerr'] = np.sqrt(simdata[sysmag1+'err']**2 + simdata[sysmag2+'err']**2 - 2.*simdata['covar']) 

   return simdata

#now specify a Salpeter LF, alpha is exponent in linear eqn, alpha = Gamma + 1

def f_salpeter(mass_arr,mass_min,mass_max,alpha,**kwargs):
    #to end sets last element to 0 otherwise one element too few.
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)   
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] #update last element
    dmass_arr = abs(dmass_arr)
    dN_arr = (mass_arr**(-1.*alpha)) * dmass_arr
    #dN_arr[mass_arr < mass_min] = 0.0
    #dN_arr[mass_arr > mass_max] = 0.0
    dN_arr[mass_arr < 0.08] = 0.0
    dN_arr[mass_arr > 100.] = 0.0
    if 'normalize' in kwargs.keys():
        if kwargs['normalize'] == False: #return dN_arr before normalization
            return dN_arr
    #Find normalization - 12-aug-2012
    knorm = 1. / np.sum(dN_arr[(mass_arr >= mass_min) & (mass_arr <= mass_max)])
    dN_arr = knorm * dN_arr
    return dN_arr

#specify a Chabrier LF, but given in dN/dM. The Chabrier IMF is given typically as dN/d(logM)
#dN/dM = (1/ln10)*(1/M)*dN/dlogM, and this is calculated within the function. Finally, return
#dM, as for f_salpeter .
#Careful: denominator in first term has ln10 = np.log(10), but exponential is log10 M, so np.log10(m)
def f_chabrier(mass_arr,mass_min,mass_max,mass_crit,sigma_mass_crit,**kwargs):
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)  
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] 
    dmass_arr = abs(dmass_arr)
    #dN_arr = (1./(np.log(10.)) * (1./(np.sqrt(2.*np.pi)*sigma_mass_crit)) * 
    #    np.exp(-1. * (np.log(mass_arr)-np.log(mass_crit))**2 / (2. * sigma_mass_crit**2)) * 
    #    dmass_arr)
    dN_arr = ((1./(np.log(10.)*mass_arr)) * (1./(np.sqrt(2.*np.pi)*sigma_mass_crit)) * 
        np.exp(-1. * (np.log(mass_arr)-np.log(mass_crit))**2 / (2. * sigma_mass_crit**2)) * 
        dmass_arr)
    #dN_arr = (1./(np.log(10.)) * (1./(np.sqrt(2.*np.pi)*sigma_mass_crit)) * 
    #    np.exp(-1. * (np.log(mass_arr)-np.log(mass_crit))**2 / (2. * sigma_mass_crit**2)) * 
    #    dmass_arr)
    #plt.plot(mass_arr,dN_arr)
    #plt.plot(mass_arr,dN_arr2,c='green')
    #plt.show()
    dN_arr[mass_arr < 0.08] = 0.0
    dN_arr[mass_arr > 100.0] = 0.0
    if 'normalize' in kwargs.keys():
        if kwargs['normalize'] == False: #return dN_arr before normalization
            return dN_arr
    #Find normalization - 12-aug-2012
    #knorm = 1. / dN_arr.sum()
    knorm = 1. / np.sum(dN_arr[(mass_arr >= mass_min) & (mass_arr <= mass_max)])
    dN_arr = knorm * dN_arr
    return dN_arr

#Define a Kroupa-like IMF, i.e. with two broken power laws. For low mass-only systems
#the high mass segment, M > 1 Msun is not important. Here it is fixed to alpha=2.3 as
#in Kroupa 2001. The Kroupa single-star IMF values are: alpha = 1.3 pm 0.5 for 
#0.08 < M < 0.5 Msun, and alpha = 2.3 pm 0.3 for 0.5 < M < 1 Msun.
def f_kroupa(mass_arr,mass_min,mass_max,alpha_1,alpha_2,**kwargs):
    alpha_3 = 2.3
    #to end sets last element to 0 otherwise one element too few.
    dmass_arr = np.ediff1d(mass_arr,to_end=0.0)   
    dmass_arr[len(dmass_arr)-1] = dmass_arr[len(dmass_arr)-2] #update last element
    dmass_arr = abs(dmass_arr)  #makes dM positive
    #Define dN for each of the segments
    f_23 = (1.0**(-1.*alpha_3)) / (1.0**(-1.*alpha_3)) 
    f_12 = (0.5**(-1.*alpha_2)) / (0.5**(-1.*alpha_1)) * f_23
    dN_arr = mass_arr * 0.0 + 0.0
    dN_arr_1 = f_12 * (mass_arr**(-1.*alpha_1)) * dmass_arr
    dN_arr_2 = f_23 * (mass_arr**(-1.*alpha_2)) * dmass_arr
    dN_arr_3 = (mass_arr**(-1.*alpha_3)) * dmass_arr
    #Assign dN_arr_1/2/3 only to their corresponding mass ranges
    dN_arr[(mass_arr < 0.08)] = 0.0
    dN_arr[(mass_arr >= 0.08) & (mass_arr < 0.5)] = dN_arr_1[(mass_arr >= 0.08) & (mass_arr < 0.5)]
    dN_arr[(mass_arr >= 0.50) & (mass_arr < 1.0)] = dN_arr_2[(mass_arr >= 0.50) & (mass_arr < 1.0)]
    dN_arr[(mass_arr >= 1.00)] = dN_arr_3[(mass_arr >= 1.00)]
    dN_arr[(mass_arr > 100.0)] = 0.0
    #Now splice the segments together so that dN/dM is continuous
    if 'normalize' in kwargs.keys():
        if kwargs['normalize'] == False: #return dN_arr before normalization
            return dN_arr
    #Find normalization - 12-aug-2012
    knorm = 1. / np.sum(dN_arr[(mass_arr >= mass_min) & (mass_arr <= mass_max)])
    dN_arr = knorm * dN_arr
    return dN_arr

    
def likelihood_matrix(cmd_point,iso_point,error_cov,c_arr):
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

def likelihood_nocovar(sigma_r,sigma_gr,delta_gr_arr,delta_r_arr):
    #arrays must be ndarr, not python lists
    P = 1.0/(2.0*np.pi*sigma_r*sigma_gr)
    exp_arg = np.exp(-0.5*(delta_gr_arr**2/sigma_gr**2)-
              0.5*(delta_r_arr**2/sigma_r**2)) 
    #print P*exp_arg
    return P*exp_arg

def estimate_required_n(nrequired,age,feh,afe,system,sysmag2,dmod0,magmin,magmax,**kwargs):
    '''Given a number of desired stars N, estimate how many
    should be asked for from the full LF (or from mmin_global to mmax_global if those
    kwargs are passed as inputs) so that approx N 
    stars remain after magnitude cuts. The kwarfs are imftype and alpha, or
    imftype, mc, and sigmac'''

    #Read in isochorne to get magnitude at given mass
    iso = read_iso_darth(age,feh,afe,system)
    
    #Estimate total number of stars between the observed mag bounds
    imax = np.argmin(abs(iso[sysmag2] + dmod0 - magmax))
    imin = np.argmin(abs(iso[sysmag2] + dmod0 - magmin))
    #range of masses for observed CMD
    mass_min_fit = iso['mass'][imax]
    mass_max_fit = iso['mass'][imin]
    #range of masses for entire CMD
    if 'mass_min_global' in kwargs.keys(): 
        mass_min_global = kwargs['mass_min_global']
    else:
        mass_min_global = iso['mass'].min()
    if 'mass_max_global' in kwargs.keys(): 
        mass_max_global = kwargs['mass_max_global']
    else:
        mass_max_global = iso['mass'].max()  

    #check that bounds for mmin/mmax_global are larger than for mmin & mmax calculated from
    #the isochrone
    if mass_min_global > mass_min_fit: 
        print "Min Mass (LF) > Min Mass (Mag cut)",mass_min_global,mass_min_fit
        raise SystemExit
    if mass_max_global < mass_max_fit: 
        print "Max Mass (LF) > Max Mass (Mag cut)",mass_max_global,mass_max_fit
        raise SystemExit

    #and check that they are between the range of isochrone masses present
    if mass_min_global < iso['mass'].min(): 
        mass_min_global = iso['mass'].min()
        print "Min Mass (LF) below isochrone limit",mass_min_global,iso['mass'].min()
        #raise SystemExit
    if mass_max_global > iso['mass'].max(): 
        mass_max_global = iso['mass'].max()
        print "Max Mass (LF) above isochrone limit",mass_min_global,iso['mass'].max()
        #raise SystemExit

    #print out masses
    print "Mmin = ",mass_min_fit
    print "Mmax = ",mass_max_fit
    print "Mmin0 = ",mass_min_global
    print "Mmax0 = ",mass_max_global

    xdum = np.arange(mass_min_global,mass_max_global,0.0001)
    if kwargs['imftype'] == 'salpeter':
        if 'alpha' not in kwargs.keys():
            print "Error: alpha not specified for Salpeter function" 
            raise SystemExit
        alpha_ = kwargs['alpha']
        f_tmp = f_salpeter(xdum,mass_min_global,mass_max_global,alpha_,normalize=False)
        #f_all = f_tmp[0]*(mass_max_global-mass_min_global)/(xdum[1] - xdum[0])
        f_all = np.sum(f_tmp[(xdum >= mass_min_global) & (xdum <= mass_max_global)])
        f_cut = np.sum(f_tmp[(xdum >= mass_min_fit) & (xdum <= mass_max_fit)])

    elif kwargs['imftype'] == 'chabrier':
        if 'mc' not in kwargs.keys():
            print "Error: M_c (kwarg=mc) not specified for Chabrier function" 
            raise SystemExit
        elif 'sigmac' not in kwargs.keys():
            print "Error: sigma_c (kwarg=sigmac) not specified for Chabrier function" 
            raise SystemExit
        pass
        mc_ = kwargs['mc']
        sigmac_ = kwargs['sigmac']
        f_tmp = f_chabrier(xdum,mass_min_global,mass_max_global,mc_,sigmac_,normalize=False)
        f_all = np.sum(f_tmp[(xdum >= mass_min_global) & (xdum <= mass_max_global)])
        f_cut = np.sum(f_tmp[(xdum >= mass_min_fit) & (xdum <= mass_max_fit)])

    elif kwargs['imftype'] == 'kroupa':
        if 'alpha1' not in kwargs.keys():
            print "Error: alpha1 (kwarg=alpha1) (slope for 0.08<M<0.5) not specified for Kroupa function" 
            raise SystemExit
        elif 'alpha2' not in kwargs.keys():
            print "Error: alpha2 (kwarg=alpha2) (slope for 0.5<M<1.0) not specified for Kroupa function" 
            raise SystemExit
        pass
        alpha1_ = kwargs['alpha1']
        alpha2_ = kwargs['alpha2']
        f_tmp = f_kroupa(xdum,mass_min_global,mass_max_global,alpha1_,alpha2_,normalize=False)
        f_all = np.sum(f_tmp[(xdum >= mass_min_global) & (xdum <= mass_max_global)])
        f_cut = np.sum(f_tmp[(xdum >= mass_min_fit) & (xdum <= mass_max_fit)])

    else:
        raise SystemExit

    print 'N_all  =  ',int(nrequired * f_all / f_cut)
   
    return int(nrequired * (f_all / f_cut)), mass_min_fit, mass_max_fit
    
