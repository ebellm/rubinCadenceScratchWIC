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

# matplotlib methods
import matplotlib
import matplotlib.pylab as plt
from matplotlib.ticker import LogLocator
plt.ion()

class ebv3d(object):

    """Object to hold and manipulate a merged 3d extinction map generated
by compareExtinctions.py. Also reads in an (nrows, nbins) mask array
but currently does nothing with it."""
    
    def __init__(self, pathMap='merged_ebv3d_nside64.fits', \
                 Verbose=True):

        # path to map
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
        
    def loadMap(self):

        """Loads the extinction map"""

        if not os.access(self.pathMap, os.R_OK):
            if self.Verbose:
                print("loadMap WARN - cannot read path %s" \
                      % (self.pathMap))
            return

        hdul = fits.open(self.pathMap)
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

    def getDeltaMag(self, sFilt='r'):

        """Converts the reddening map into an (m-M) map for the given
        filter"""

        if not sFilt in self.R_x.keys():
            sFilt = 'r'
        Rx = self.R_x[sFilt]
        mMinusM = self.dmods[np.newaxis,:] + Rx * self.ebvs

        return mMinusM[0]

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

        
def testReadExt(showExtn=False, sfilt='r', showDeltamag=False, \
                figName='test_mapDust.png', \
                pathMap='merged_ebv3d_nside64.fits'):

    """Tests whether we can read the extinction map we just created. If
showExtn, then the extinction at filter sfilt is shown. If showDeltamag, then the quantity (m-M) is plotted, including extinction. pathMap is the path to the E(B-V) vs distance map. Example call:

    readExtinction.testReadExt(sfilt='r', showExtn=True, \
    figName='testmap_Ar.png')

    """

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
            
        # Show the dust map
        hp.mollview(vecSho, 2, coord=['C','G'], nest=ebv.nested, \
                    title='Requested distance %.1f pc' % (dpcs[iDist]), \
                    unit=sUnit, \
                    cmap=cmap, sub=(2,2,iDist+1), \
                    norm='log', margins=margins)

        cbar = plt.gca().images[-1].colorbar
        cmin, cmax = cbar.get_clim()
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
                  pathMap='merged_ebv3d_nside64.fits', \
                  dmagVec=np.array([])):

    """Use the extinction map to find the distance at which a particular
delta-mag is found. Example call:

    Find the distance in parsecs at which (m-M)_i = 15.2, using a
    stepped colormap until I work out how to add tickmarks to
    the colorbar...:


    readExtinction.testDeltamags('i', 15.2, cmap='Set2', \
    figName='testmap_delta_i_set1.png')

    """

    ebv = ebv3d(pathMap)
    ebv.loadMap()

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

    fig4=plt.figure(4, figsize=(8,6))
    fig4.clf()
    sTitle = r'Distance at $\Delta$%s=%.2f (%s scale)' \
             % (sfilt, dmagOne, norm)
    hp.mollview(distsClosest, 4, coord=['C','G'], nest=ebv.nested, \
                title=sTitle, \
                unit='Distance (pc)', \
                cmap=cmap, norm=norm)

    cbar = plt.gca().images[-1].colorbar
    cmin, cmax = cbar.get_clim()
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

def testShowDistresol(pathMap='merged_ebv3d_nside64.fits'):

    """Test our method to show the distance resolution"""

    ebv=ebv3d(pathMap)
    ebv.loadMap()
    ebv.showDistanceInterval()
