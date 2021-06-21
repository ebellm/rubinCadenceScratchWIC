#
# readExtinction.py
#

#
# Utilities to read in and use the halpix map of 3D extinction
# generated by compareExtinctions.py and stilism_local.pt
#

import os
import math
import numpy as np

import healpy as hp
from astropy.io import fits

# For querying a particular sight line
from astropy.coordinates import SkyCoord
import astropy.units as u

# matplotlib methods
import matplotlib
import matplotlib.pylab as plt
from matplotlib.ticker import LogLocator
plt.ion()

import requests

# There is no clear way of knowing if we are inside a datalab server or not
# As a workaround, we try to import the datalab package, if we fail we certainly cannot 
# use the datalab functionality
try:
    from dl import storeClient as sc
    datalab=True
except:
    datalab=False

py_folder = os.path.dirname(os.path.abspath(__file__))
extmaps_dir = os.path.join(os.path.dirname(py_folder), 'extmaps')

# Turn the matplotlib version into a float
MATPLOTLIB_VERSION_FLOAT = \
    float('%s%s' % (matplotlib.__version__[0:3], matplotlib.__version__[3::].replace('.','')))

class ebv3d(object):

    """Object to hold and manipulate a merged 3d extinction map generated
by compareExtinctions.py. Also reads in an (nrows, nbins) mask array
but currently does nothing with it."""

    default_map = os.path.join(extmaps_dir,'merged_ebv3d_nside64.fits')
    
    def __init__(self, pathMap=None, Verbose=True):

        # path to map
        if pathMap is None:
            self.pathMap = self.default_map
        else:
            self.pathMap = pathMap[:]

        # Arrays for the map
        self.hpids = np.array([])
        self.dists = np.array([])
        self.ebvs = np.array([])
        self.sfacs = np.array([])
        self.mask = np.array([])

        # extinction information
        self.hdr = []
        self.nside = 64
        self.nested = False
        
        # Distance moduli from distances
        self.dmods = np.array([])

        # R_x factors for the LSST filters. This was generated with
        # the following line, using NOIRLAB Datalab sims_maf and
        # pasting the Dust_values object from the github sims_maf
        # 2021-04-05. From the jupyter notebook:
        #
        # tryThis = Dust_values(3.1, ref_ebv=1.)
        # tryThis.Ax1
        self.R_x = {'u':4.80, 'g':3.67, 'r':2.70, \
                    'i':2.06, 'z':1.59, 'y':1.31}
        
        # verbose?
        self.Verbose = Verbose
        
    def loadMap(self, use_local=None, use_sciserver=None, use_datalab=None):

        """Loads the extinction map"""

        # If all kwargs are None, to load the map there is a priority chain.
        # 1) Check the map (default or user specified) exists in the extmap folder of the repository.
        #    Must check for real map, not gitlfs placeholder.
        # 2) Check Sciserver public folder
        # 3) Check datalab public folder (thalos12) -- this can be used also locally!

        print("Loading the map.")

        # Keep track of whether the map was successfully loaded at one of the steps
        hdul=None

        # Check local map
        if use_local or use_local is None:
            try:
                hdul = fits.open(self.pathMap)
            except OSError:
                if use_local:
                    raise FileNotFoundError('The map file {} could not be loaded.'.format(self.pathMap))
                if self.Verbose:
                    print("The map could not be loaded from {}.".format(self.pathMap))
        
        # Check Sciserver public/shared folder
        if hdul is None and (use_sciserver or use_sciserver is None):
            map_name = os.path.split(self.pathMap)[1]
            if not map_name.endswith('.gz'):
                    map_name = map_name + '.gz'
            map_path = os.path.join('/home/idies/workspace/lsst_cadence/LSST_Shared/extmaps', map_name)
            try:
                hdul = fits.open(map_name)
            except OSError:
                if use_sciserver:
                    raise FileNotFoundError('The map file {} could not be loaded.'.format(map_path))
                if self.Verbose:
                    print("The map could not be loaded from the SciServer shared folder.")
        
        # Check datalab public folder
        if hdul is None and (use_datalab or use_datalab is None):
            if not datalab:
                if self.use_datalab:
                    raise ValueError("The datalab Python package is not available.")
                if self.Verbose:
                    print("The datalab package is not installed, skipping it.")
            else:
                # If importing the datalab package succeeded we can try to query the public map file
                # WARNING: this works even locally, the datalab package will just download the map
                map_name = os.path.split(self.pathMap)[1]
                # On the public space there should be only the compressed maps
                if not map_name.endswith('.gz'):
                    map_name = map_name + '.gz'
                map_path = 'thalos12://public/{}'.format(map_name)
                try:
                    hdul = fits.open(sc.get(map_path,mode='fileobj'))
                except requests.exceptions.MissingSchema:
                    # This exception is raised when the file does not exist on datalab, as far as I have understood
                    if use_datalab:
                        raise FileNotFoundError("The map {} could not be loaded.".format(map_path))
                    if self.Verbose:
                        print("The map could not be loaded from the Data Lab shared folder.")
        
        if hdul is None:
            raise FileNotFoundError("The map could not be loaded!")
        print("Loaded the map.")
        
        self.hdr = hdul[0].header

        # now populate the arrays in turn
        self.hpids = hdul[0].data
        self.dists = hdul[1].data
        self.ebvs = hdul[2].data

        if len(hdul) > 3:
            self.sfacs = hdul[3].data

        if len(hdul) > 4:
            self.mask = hdul[4].data
        
        hdul.close()

        # parse the header for the healpix information we need.
        self.nside = self.hdr['NSIDE']
        self.nested = self.hdr['NESTED']

        
        self.calcDistmods()
        
    def calcDistmods(self):

        """Converts distances in parsecs to distance moduli in magnitudes"""

        self.dmods = 5.0*np.log10(self.dists) - 5.
        
    def getMapNearestDist(self, distPc=3000):

        """Returns the distances and extinctions closest to the supplied
distance"""

        # just use argmin
        imin = np.argmin(np.abs(self.dists - distPc), axis=1)

        # now lift the ebv values at this distance
        iExpand = np.expand_dims(imin, axis=-1)
        distsClosest = np.take_along_axis(self.dists, \
                                          iExpand, \
                                          axis=-1).squeeze()

        ebvsClosest = np.take_along_axis(self.ebvs, \
                                          iExpand, \
                                          axis=-1).squeeze()

        return ebvsClosest, distsClosest

    def getEBVatSightline(self, l=0., b=0., \
                          ebvMap=np.array([]), interp=False, \
                          showEBVdist=False):
        
        """Utility - returns E(B-V) for one or more sightlines in Galactic
        coordinates. Takes as an argument a 2D healpix array of
        quantities (usually this will be reddening returned by
        getMapNearestDist() ). Also returns the nearest coords to the
        requested coords for debug purposes. Arguments:

        l, b = Galactic coordinates of the sight-line(s). Can be
        scalar or vector.

        ebvMap = 2D reddening map to use

        interp: Interpolate using healpy? If False, the nearest
        healpix is used instead.

        showEBVdist = Will usually be used only for debugging
        purposes. If True, this method will plot the run of E(B-V) vs
        distance for the nearest hpid (ignored if interp=True)

        """

        if np.size(ebvMap) < 1:
            return 0., -99., -99.
        
        # find the coords on the sky of the requested sight line, and
        # convert this to healpix
        coo = SkyCoord(l*u.deg, b*u.deg, frame='galactic')

        # Equatorial coordinates of the requested position(s)
        ra = coo.icrs.ra.deg
        dec = coo.icrs.dec.deg

        if interp:
            ebvRet = hp.get_interp_val(ebvMap, ra, dec, \
                                       nest=self.nested, lonlat=True)

            # For backwards compatibility with the "Test" return values
            lTest = np.copy(l)
            bTest = np.copy(b)
            
        else:
            hpid = hp.ang2pix(self.nside, \
                              ra, dec, \
                              nest=self.nested, lonlat=True)
            ebvRet = ebvMap[hpid]
            
            # For debugging: determine the coordinates at this nearest pixel
            raTest, decTest = hp.pix2ang(self.nside, hpid, \
                                         nest=self.nested, lonlat=True)
            cooTest = SkyCoord(raTest*u.deg, decTest*u.deg, frame='icrs')
            lTest = cooTest.galactic.l.degree
            bTest = cooTest.galactic.b.degree

            # also for debugging: show the run of E(B-V) vs distance
            # for the nearest healpix ID:
            if showEBVdist and np.isscalar(hpid):
                fig9 = plt.figure(9)
                fig9.clf()
                ax9 = fig9.add_subplot(111)
                dummy = ax9.plot(self.dists[hpid], self.ebvs[hpid])
                ax9.set_xlabel('Distance, pc')
                ax9.set_ylabel('E(B-V)')
                ax9.grid(which='both', visible=True, alpha=0.5)
                ax9.set_title('hpid: %i, (l,b) = (%.2f, %.2f)' \
                              % (hpid, lTest , bTest))
                
        return ebvRet, lTest, bTest

    def getMaxDistDeltaMag(self, dmagVec, sfilt='r', ipix=None):
        """Compute the maximum distance for an apparent m-M using the maximum
        value of extinction from the map.
        """

        # We do distance modulus = (m-M) - A_x, and calculate the
        # distance from the result. We do this for every sightline
        # at once.
        if ipix is not None:
            ebvsMax = self.R_x[sfilt] * self.ebvs[ipix,-1]
        else:
            ebvsMax = self.R_x[sfilt] * self.ebvs[:,-1]
        distModsFar = dmagVec - ebvsMax

        distsFar = 10.0**(0.2*distModsFar + 1.)

        return distsFar

    def getDeltaMag(self, sFilt='r',ipix=None):

        """Converts the reddening map into an (m-m0) map for the given
        filter"""

        if not sFilt in self.R_x.keys():
            sFilt = 'r'
        Rx = self.R_x[sFilt]
        if ipix is not None:
            mMinusm0 = self.dmods[np.newaxis,ipix] + Rx * self.ebvs[ipix,:]
            # make 3d so that form the outside nothing changes
        else:
            mMinusm0 = self.dmods[np.newaxis,:] + Rx * self.ebvs

        return mMinusm0[0]

    def getDistanceAtMag(self, deltamag=15.2, sfilt='r', ipix=None, \
                         extrapolateFar=True):

        """Returns the distances at which the combination of distance and
extinction produces the input magnitude difference (m-M) = deltamag. Arguments:

        ARGUMENTS:

        deltamag = target (m-m_0). Can be scalar or array. If array,
        must have the same number of elements as the healpix map
        (i.e. hp.nside2npix(64) )

        sfilt = filter at which we want deltamag

        ipix = Pixels for which to perform the evaluation.

        extrapolateFar - for distances beyond the maximum
        distance in the model, treat the extinction as constant beyond
        that maximum distance and compute the distance at which
        delta-mag is achieved. (Defaults to True: only set to False if
        you know what you are doing!!)

        RETURNS:

        distances, mMinusM, bFar   -- where:

        distances = npix-length array giving the distances in parsecs

        mMinusM = npix-length array giving the magnitude differences

        bFar = npix-length boolean indicating whether the maximum
        distance indicated by a sight line was beyond the range of
        validity of the extinction model.

            If ipix is provided, either as a single int or a list|array or
            ints representing Healpix pixel indices, only the number of
            pixels requested will be queried. Arrays will be returne in any
            case, even when one single pixel is requested.
        """

        # A little bit of parsing... if deltamag is a scalar,
        # replicate it into an array. Otherwise just reference the
        # array that was passed in. For the moment, trust the user to
        # have inputted a deltamag vector of the right shape.
        if ipix is not None:
            ipix = np.atleast_1d(ipix)
            npix = ipix.shape[0]
        else:
            npix = self.ebvs.shape[0]
        if np.isscalar(deltamag):
            dmagVec = np.repeat(deltamag, npix)
        else:
            dmagVec = deltamag

        if np.size(dmagVec) != npix:
            print("ebv3d.getDistanceAtMag WARN - size mismatch:", \
                  npix, np.shape(dmagVec))
            return np.array([]), np.array([]), np.array([])

        # Now we need apparent minus absolute magnitude:
        mMinusM = self.getDeltaMag(sfilt,ipix=ipix)

        # Now we find elements in each row that are closest to the
        # requested deltamag:
        iMin = np.argmin(np.abs(mMinusM - dmagVec[:,np.newaxis]), axis=1)
        iExpand = np.expand_dims(iMin, axis=-1)

        # select only m-M at needed distance
        mMinusM = np.take_along_axis(mMinusM, iMin[:,np.newaxis], -1).flatten()

        # now find the closest distance...
        if ipix is not None:
            distsClosest = np.take_along_axis(self.dists[ipix], \
                                              iExpand, \
                                              axis=-1).squeeze()
            # To keep things similar to the case of querying the whole map,
            #  we need to return an array also in case ipix is a single pixel.
            distsClosest = np.atleast_1d(distsClosest)
            # if npix>1:
            #     distsClosest = distsClosest.squeeze()
        else:
            distsClosest = np.take_along_axis(self.dists, \
                                              iExpand, \
                                              axis=-1).squeeze()

        # 2021-04-09: started implementing distances at or beyond the
        # maximum distance. Points for which the closest delta-mag is
        # in the maximum distance bin are picked.
        if ipix is not None:
            bFar = iMin == self.dists[ipix].shape[-1]-1
        else:
            bFar = iMin == self.dists.shape[-1]-1
        if not extrapolateFar:
            return distsClosest, mMinusM, bFar
        
        # For distances beyond the max, we use the maximum E(B-V)
        # along the line of sight to compute the distance.

        distsFar = self.getMaxDistDeltaMag(mMinusM, sfilt, ipix)

        # Now we swap in the far distances
        distsClosest[bFar] = distsFar[bFar]

        # ... Let's return both the closest distances and the map of
        # (m-M), since the user might want both.
        return distsClosest, mMinusM, bFar
    
    def getInterpolatedProfile(self, gall, galb, dist):
        gall = np.atleast_1d(gall)
        galb = np.atleast_1d(galb)
        dist = np.atleast_2d(dist).T # required for subtraction to 2d array with shape (N,N_samples)
        N = len(dist)
        
        coo = SkyCoord(gall*u.deg,galb*u.deg,frame='galactic')
        RAs = coo.icrs.ra.deg
        DECs = coo.icrs.dec.deg
        hpids, weights = hp.get_interp_weights(self.nside, RAs, DECs, self.nested, lonlat=True)
        # hpids, weights = hp.get_interp_weights(64, gall, galb, False, lonlat=True)
        ebvout = np.zeros(N)
        distout = np.zeros(N)
        for i in range(hpids.shape[0]):
            pid = hpids[i]
            w = weights[i]
            distID = np.argmin(np.abs(self.dists[pid]-dist),axis=1)
            ebvout_i = self.ebvs[pid,distID] * w
            ebvout += ebvout_i
            distout_i = self.dists[pid,distID] * w
            distout += distout_i

    def showMollview(self, hparr=np.array([]), fignum=4, \
                     subplot=(1,1,1), figsize=(10,6),\
                     cmap='Set2', numTicks=9, \
                     clobberFigure=True, \
                     sTitle='', sUnit='TEST UNIT', \
                     sSuptitle='', \
                     coord=['C','G'], norm='linear', \
                     gratColor='0.2', gratAlpha=0.5, \
                     margins=(0.05, 0.05, 0.05, 0.05), \
                     fontsize=-1, \
                     minval=None, maxval=None, sfilt='r'):


        """Plot mollweide view using customized colorbar ticks. Returns the
figure. Arguments:

        hparr = healpix array to show

        fignum = matplotlib figure number

        subplot = subplot string for figure. Default (1,1,1)

        figsize = figure size

        cmap = colornam

        nticks = number of ticks to use (TODO: set this from the
        colornap)

        fontsize = desired colorbar label font size. Set to negative to accept defaults

        minval, maxval = minmax values to show

        sfilt = filter string if rescaling colorbar labels

        """

        # the number of ticks and fontsize are overridden with
        # defaults if the colormap is one of the set below.
        labelsize = 10
        Dnticks = {'Set1':10, 'Set2':9, 'Set3':13, 'tab10':11, \
                   'Paired':13, 'Pastel2':9, 'Pastel1':10, \
                   'Accent':9, 'Dark2':10}
        Dlsize =  {'Set1':9,  'Set2':10, 'Set3':7.5, 'tab10':9, \
                   'Paired':7.5, 'Pastel2':9, 'Pastel1':9, \
                   'Accent':9, 'Dark2':9}

        # Set the number of ticks and the fontsize, allowing for
        # reversed colormaps
        labelsize = 8 # default
        cmapStem = cmap.split('_r')[0]
        if cmapStem in Dnticks.keys():
            numTicks = Dnticks[cmapStem]
            labelsize = Dlsize[cmapStem]

        # allow user-input override fontsize for labels
        if fontsize > 0:
            labelsize=fontsize

        # Is the input sensible?
        if np.size(hparr) < 1:
            return None

        fig = plt.figure(fignum, figsize=figsize)
        if clobberFigure:
            fig.clf()

        hp.mollview(hparr, fignum, coord=coord, nest=self.nested, \
                    sub=subplot, \
                    title=sTitle, unit=sUnit, cmap=cmap, norm=norm, \
                    margins=margins, \
                    min=minval, max=maxval)

        # Now we use Alessandro's nice method for handling the
        # colorbar:
        cbar =  plt.gca().images[-1].colorbar

        # 2021-11-04 the following methods will only work if cbar is
        # not None (as I am finding when using rubin_sim). So:
        if cbar is not None:

            cmin, cmax = getColorbarLimits(cbar)

            # The colorbar has log scale, which means that cmin=0 is
            # not valid this should be handled by mollview, if not
            # cmin is replaced by the smallest non-zero value of the
            # array vecSho
            if cmin==0:
                cmin=np.amin(sfilt[sfilt!=0])
                # Set tick positions and labels
            cmap_ticks = np.linspace(cmin,cmax,num=numTicks)

            cbar.set_ticks(cmap_ticks,True)
            cmap_labels = ["{:5.0f}".format(t) for t in cmap_ticks]
            cbar.set_ticklabels(cmap_labels)
            cbar.ax.tick_params(labelsize=labelsize) 
            # Change the position of the colorbar label
            text = [c for c in cbar.ax.get_children() \
                        if isinstance(c,matplotlib.text.Text) if c.get_text()][0]
            # print(text.get_position())
            text.set_y(-3.) # valid for figsize=(8,6)

        # now show a graticule
        hp.graticule(color=gratColor, alpha=gratAlpha)

        # set supertitle if set
        if len(sSuptitle) > 0:
            fig.suptitle(sSuptitle)

        return fig
            
    def showDistanceInterval(self, fignum=5, cmap='viridis'):

        """Utility - shows the map of distance resolutions for close and far
points. Currently this just uses the difference betweeh bin 1 and 0 as
the bin spacing for L+19, and between bins -2 and -1 for the bovy et
al. spacing.

        """

        ddistClose = self.dists[:,1] - self.dists[:,0]
        ddistFar = self.dists[:,-1] - self.dists[:,-2]

        fig5=plt.figure(fignum, figsize=(10,3))
        fig5.clf()

        # set the margins
        margins = (0.02, 0.05, 0.05, 0.00)
                
        hp.mollview(ddistClose, fignum, coord=['C','G'], \
                    nest=self.nested, \
                    title='Nearest distance bin width, pc', \
                    unit=r'$\Delta d$, pc', \
                    cmap=cmap, sub=(1,2,1), \
                    margins=margins)

        hp.mollview(ddistFar, fignum, coord=['C','G'], \
                    nest=self.nested, \
                    title='Farthest distance bin width, pc', \
                    unit=r'$\Delta d$, pc', \
                    cmap=cmap, sub=(1,2,2), \
                    margins=margins)

###
def getColorbarLimits(cbar=None):

    """Gets the colorbar limits for matplotlib colorbar, respecting
    the matplotlib version"""

    if cbar is None:
        return 0., 1.

    if MATPLOTLIB_VERSION_FLOAT < 3.3:
        return cbar.get_clim()
    else:
        return cbar.mappable.get_clim()

def testReadExt(showExtn=False, sfilt='r', showDeltamag=False, \
                figName='test_mapDust.png', \
                pathMap=None, norm='log'):

    """Tests whether we can read the extinction map we just created. If
showExtn, then the extinction at filter sfilt is shown. If showDeltamag, then the quantity (m-M) is plotted, including extinction. pathMap is the path to the E(B-V) vs distance map. Example call:

    readExtinction.testReadExt(sfilt='r', showExtn=True, \
    figName='testmap_Ar.png')

    """
    if pathMap is None:
        pathMap = os.path.join(extmaps_dir,'merged_ebv3d_nside64.fits')
        print("Using the default map at",pathMap)

    ebv = ebv3d(pathMap)
    ebv.loadMap()

    # print(np.shape(ebv.mask))
    
    # ebvThis, distThis = ebv.getMapNearestDist(dpc)

    
    ## try showing the scale factors
    #hp.mollview(ebv.sfacs, 3, coord=['C', 'G'], nest=ebv.nested)

    # print(ebv.sfacs[0:10], np.min(ebv.sfacs), np.max(ebv.sfacs))
    
    fig2=plt.figure(2, figsize=(14,8))
    fig2.clf()

    # fig3=plt.figure(3, figsize=(7,7))
    # fig3.clf()

    # Must change the margins, the default ones are too thin and text
    # overflows from the image
    # follow the scheme -> (left,bottom,right,top)
    # margins = (0.075, 0.075, 0.075, 0.05) # from mollview function
    margins = (0.0, 0.02, 0.00, 0.02) # valid for figsize=(14,8)

    rx = 1.
    sUnit = 'E(B-V), mag'
    cmap='plasma_r'
    if showExtn and sfilt in ebv.R_x.keys():
        rx = ebv.R_x[sfilt]
        sUnit = r'A$_%s$, mag' % (sfilt)
        # cmap = 'Greys'
        
    dpcs = [252., 1503.5, 4000., 7500.]
    for iDist in range(len(dpcs)):
        ebvThis, distThis = ebv.getMapNearestDist(dpcs[iDist])

        vecSho = ebvThis*rx
        
        if showDeltamag and showExtn:
            dmod = 5.0*np.log10(distThis) - 5.
            vecSho += dmod
            sUnit = r'(m-M)$_%s$' % (sfilt)

        # if log scheme requested, make vecSho a masked array
        if norm.find('log') > -1:
            vecSho = np.ma.masked_less_equal(vecSho, 0.)
            # on my system, mollview still doesn't handle masked
            # arrays well...
            vecSho[vecSho.mask] = np.ma.min(vecSho)
            
        # Show the dust map
        hp.mollview(vecSho, 2, coord=['C','G'], nest=ebv.nested, \
                    title='Requested distance %.1f pc' % (dpcs[iDist]), \
                    unit=sUnit, \
                    cmap=cmap, sub=(2,2,iDist+1), \
                    norm=norm, margins=margins)

        cbar = plt.gca().images[-1].colorbar
        cmin, cmax = getColorbarLimits(cbar)
        # The colorbar has log scale, which means that cmin=0 is not valid
        # this should be handled by mollview, if not cmin is replaced by the
        # smallest non-zero value of the array vecSho
        if cmin==0:
            cmin=np.amin(vecSho[vecSho!=0])
        # Set tick positions and labels
        cmap_ticks = np.logspace(math.log10(cmin),math.log10(cmax),num=5)
        cbar.set_ticks(cmap_ticks,True)
        cmap_labels = ["{:4.3g}".format(t) for t in cmap_ticks]
        cbar.set_ticklabels(cmap_labels)
        # Change the position of the colorbar label
        text = [c for c in cbar.ax.get_children() if isinstance(c,matplotlib.text.Text) if c.get_text()][0]
        print(text.get_position())
        text.set_y(-2.5) # valid for figsize=(14,8)

        hp.graticule(alpha=0.5, color='0.25')
        
        # show the distance between the nearest distance bin and the
        # requested distance
        #hp.mollview(distThis - dpcs[iDist], 3, \
        #            coord=['C','G'], nest=ebv.nested, \
        #            title='Requested distance %.1f pc' % (dpcs[iDist]), \
        #            unit='Distance - requested, pc', \
        #            cmap='RdBu_r', sub=(2,2,iDist+1))

    fig2.suptitle('NSIDE=%i' % (ebv.nside))
    fig2.savefig(figName)

def testDeltamags(sfilt='r', dmagOne=13., \
                  figName='test_deltamag.png', \
                  cmap='viridis', norm='linear', \
                  pathMap=None, \
                  dmagVec=np.array([]), testMethod=False, \
                  testFigureMethod=False, testFarDistances=True, \
                  maxDistShow=None):

    """Use the extinction map to find the distance at which a particular
delta-mag is found.

    2021-03-07 new arguments: 

    testMethod: tests using the method in object ebv3d to find the distance. Currently defaults to False so that the notebooks on the repository will work.

    testFigureMethod: tests using the method inside the object to show a healpy mollview. Currently defaults to False so that hte notebook on the repository will work.

    Example call:

    Find the distance in parsecs at which (m-M)_i = 15.2, using a
    stepped colormap until I work out how to add tickmarks to
    the colorbar...:

    readExtinction.testDeltamags('i', 15.2, cmap='Set2', \
    figName='testmap_delta_i_set1.png')

    """

    if pathMap is None:
        pathMap = os.path.join(extmaps_dir,'merged_ebv3d_nside64.fits')
        print("Using the default map at",pathMap)

    ebv = ebv3d(pathMap)
    ebv.loadMap()

    if not testMethod:
    
        # for the supplied filter choice, build an (m-M)_x map from the
        # reddening and the distance moduli
        mMinusM = ebv.getDeltaMag(sfilt)

        # We pretend that we have one target delta-magnitude for every
        # healpix, by replicating our program deltamag into an npix-length
        # array
        if np.size(dmagVec) < 1:
            dmagVec = np.repeat(dmagOne, np.shape(mMinusM)[0])
    
        # now find the elements in each row that are closest to the
        # requested deltamag
        iMin = np.argmin(np.abs(mMinusM - dmagVec[:,np.newaxis]), axis=1)
        iExpand = np.expand_dims(iMin, axis=-1)

        # print("INFo:", np.shape(mMinusM), np.shape(iMin))
        # return
    
        # get the distances closest to this
        distsClosest = np.take_along_axis(ebv.dists, \
                                          iExpand, \
                                          axis=-1).squeeze()
    else:
        if np.size(dmagVec) < 1:
            dmagInp = dmagOne
        else:
            dmagInp = np.copy(dmagVec)

        distsClosest, mMinusM, bFar \
            = ebv.getDistanceAtMag(dmagInp, sfilt, \
                                   extrapolateFar=testFarDistances)

    if testFigureMethod:
        figThis = ebv.showMollview(distsClosest, 4, cmap=cmap, norm=norm, \
                                   coord=['C','G'], sUnit='Distance (pc)')
        return

    
    fig4=plt.figure(4, figsize=(8,6))
    fig4.clf()
    sTitle = r'Distance at $\Delta$%s=%.2f (%s scale)' \
             % (sfilt, dmagOne, norm)
    hp.mollview(distsClosest, 4, coord=['C','G'], nest=ebv.nested, \
                title=sTitle, \
                unit='Distance (pc)', \
                cmap=cmap, norm=norm, \
                max=maxDistShow)

    cbar = plt.gca().images[-1].colorbar
    cmin, cmax = getColorbarLimits(cbar)
    # The colorbar has log scale, which means that cmin=0 is not valid
    # this should be handled by mollview, if not cmin is replaced by the
    # smallest non-zero value of the array vecSho
    if cmin==0:
        cmin=np.amin(sfilt[sfilt!=0])
    # Set tick positions and labels
    cmap_ticks = np.linspace(cmin,cmax,num=9)
    cbar.set_ticks(cmap_ticks,True)
    cmap_labels = ["{:5.0f}".format(t) for t in cmap_ticks]
    cbar.set_ticklabels(cmap_labels)
    cbar.ax.tick_params(labelsize=10) 
    # Change the position of the colorbar label
    text = [c for c in cbar.ax.get_children() if isinstance(c,matplotlib.text.Text) if c.get_text()][0]
    print(text.get_position())
    text.set_y(-3.) # valid for figsize=(8,6)

    # show a graticule
    hp.graticule(color='0.2', alpha=0.5)

    fig4.suptitle('NSIDE=%i, Filter:%s' % (ebv.nside, sfilt))
    fig4.savefig(figName)

def testShowDistresol(pathMap=None):
    """Test our method to show the distance resolution"""

    if pathMap is None:
        pathMap = os.path.join(extmaps_dir,'merged_ebv3d_nside64.fits')
        print("Using the default map at",pathMap)

    ebv=ebv3d(pathMap)
    ebv.loadMap()
    ebv.showDistanceInterval()

def testGetOneSightline(l=0., b=0., dpc=3000., \
                            pathMap=None, \
                        interpCoo=False, showVsDistance=True):

    """Test getting the E(B-V) map at a particular
    sight-line. Currently the nearest healpix to the requested
    position is returned. Arguments:

    l, b = Galactic coordinates requested. Can be scalars or arrays.

    dpc = The distance requested at which the EBV is to be evaluated.

    pathMap = path to the E(B-V) map.

    interpCoo: If True, the E(B-V) values are interpolated from the
    nearest entries in the healpixel map. Otherwise the nearest
    healpix ID(s) will be queried.

    showVsDistance: if True, AND interpCoo is False, AND l, b are
    both scalars, then the run of E(B-V) vs distance is plotted for
    the nearest hpid to the requested coordinates.

    """
    
    # Commentary 2021-04-08: I think the discrepancy in tests between
    # the loaded E(B-V) and the E(B-V) values generated by
    # compareExtinctions.hybridsightline were due to a different value
    # of pixFillFac being used. When generating the extinction map
    # currently on my home area, a value of 0.80 was used (to come in
    # a bit from the pixel corners). This test routine now prints
    # selected header metadata to clarify the arguments to send to
    # hybridSightline when testing.

    # Here is a recommended sequence for testing:
    
    # readExtinction.testGetOneSightline(1, 2., dpc=4000.,
    # interpCoo=False)

    # <Then read the coordinates of the nearest hpid to those
    # requested. In this case, the nearest coords are 0.79, 1.87. So
    # we put those back in:>

    # readExtinction.testGetOneSightline(0.79, 1.87, dpc=4000.,
    # interpCoo=False)

    # Now, we use hybridsightline to re-generate the extinction map at
    # those coordinates, using the metadata printed to screen from the
    # previous command to set up the hybrid sightline in the same was
    # as was used to generate the map. In this case, pixFillFac was
    # 0.8, nL, nB were both 4.

    # compareExtinctions.hybridSightline(0.79, 1.87, nl=4, nb=4,
    # setLimDynamically=False, useTwoBinnings=True,
    # nBinsAllSightlines=300, doPlots=True, pixFillFac=0.8)

    # At THIS point, the E(B-V) you see at the requested distance
    # should match that returned by testGetOneSightline() at the
    # specified coordinates.

    # With all that said, here is the test routine:

    if pathMap is None:
        pathMap = os.path.join(extmaps_dir,'merged_ebv3d_nside64.fits')
        print("Using the default map at",pathMap)
    # load the map and compute the E(B-V) at the distance
    ebv = ebv3d(pathMap)
    ebv.loadMap()

    # Report some information to the screen about the map
    print("ebv metadata info: NESTED: %i, NSIDE:%i, pixFillFac:%.2f, nL=%i, nB=%i" \
          % (ebv.hdr['NESTED'], ebv.hdr['NSIDE'], ebv.hdr['fracPix'], ebv.hdr['nl'], ebv.hdr['nb']))

    # Also report additional metadata which might or might not appear
    # in the version on my webspace, but which is now generated by
    # loopSightlines. See compareExtinctions.loopSightlines for the
    # header keywords that are now added to the 3d map.
    print("Additional header keywords:")
    for skey in ['Rv', 'mapvers', 'PlanckOK', 'dmaxL19', \
                 'bridgL19', 'bridgwid']:
        try:
            print(skey, ebv.hdr[skey])
        except:
            pass
    
    ebvs, dists = ebv.getMapNearestDist(dpc)

    # Now we've obtained the map at a given distance, we can query
    # particular sight-lines. Let's try the coords requested
    ebvHere, lTest, bTest \
        = ebv.getEBVatSightline(l, b, ebvs, \
                                interp=interpCoo, \
                                showEBVdist=showVsDistance)

    # For testing purposes, we can also find the distance
    # corresponding to our sight line using exactly the same method:
    distHere, _, _ = ebv.getEBVatSightline(l, b, dists, \
                                           interp=interpCoo, \
                                           showEBVdist=True)

    # If the coords are scalars, report to screen.
    if np.isscalar(l):
    
        print("Info: E(B-V), distance at (l,b, distance) nearest to (%.2f, %.2f, %.1f) are %.2f, %.1f" \
              % (l, b, dpc, ebvHere, distHere))

        print("Info: nearest Galactic coords to requested position: %.2f, %.2f" \
        % (lTest, bTest))

    else:
        print("INFO: requested l:", l)
        print("INFO: requested b:", b)
        print("INFO: nearest l:", lTest)
        print("INFO: nearest b:", bTest)
        print("INFO: returned E(B-V)", ebvHere)
        print("INFO: returned distances:", distHere)


def testInteprolateProfile(gall,galb,dist,ebvmap=None):
    """Function demonstrating interpolation over nearby sightlines
    to determine EBV at a given (l,b). The value of EBV will be taken
    from the closest distance bins.

    Args:
        gall (float or iterable): Galactic longitude(s).
        galb (float or iterable): Galactic latitude
        dist (float or iterable): Distance for EBV dtermination.
        ebvmap (ebv3d, optional): Instance of the ebv3d class to be used. 
            If None, a new one will be initiaized. Defaults to None.

    Returns:
        array, array: Balue(s) of EBV and corresponding distance.
    """
    
    if ebvmap is None:
        ebvmap = ebv3d()
        ebvmap.loadMap()
    
    ebvout, distout = ebvmap.getInterpolatedProfile(gall, galb, dist)
    
    return ebvout, distout
