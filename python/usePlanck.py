#
# usePlanck.py
#

#
# Uses Gregory Green's "dustmaps" to create 2D healpix maps of Planck
#

import numpy as np

import healpy as hp
from dustmaps.planck import PlanckQuery

from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy.io import fits

import matplotlib.pylab as plt
plt.ion()

def go(nside=64, nested=False, cooFrame='icrs'):

    """Queries planck"""

    # generate sight lines for each healpix
    npix = hp.nside2npix(nside)
    lpix = np.arange(npix)
    ra, dec = hp.pix2ang(nside, lpix, nested, lonlat=True)

    # Generate sky coordinates
    coo = SkyCoord(ra*u.deg, dec*u.deg, frame=cooFrame)

    # can we send skycoords straight into planck query?
    planck = PlanckQuery()

    print("Querying Planck...")
    ebvPlanck = planck(coo)

    # Write this to fits, including header info
    hdu0 = fits.PrimaryHDU(ebvPlanck)
    hdu0.header['NSIDE'] = nside
    hdu0.header['NESTED'] = nested
    hdu0.header['frame'] = cooFrame
    hdu0.header['quant'] = 'E(B-V)'

    sNested = 'ring'
    if nested:
        sNested = 'nested'
    pathOut = 'planck_ebv_nside%i_%s.fits.gz' % (nside, sNested)

    hdu0.writeto(pathOut, overwrite=True)

def plotmap(pathmap='planck_ebv_nside64_ring.fits.gz', fignum=3, \
            nside=64, quant='ebv'):

    """Shows a sky map of the planck map"""

    # load the planck data
    planck, hdr = fits.getdata(pathmap, 0, header=True)

    # construct the figure string
    try:
        nside=hdr['NSIDE']
    except:
        pass

    try:
        quant=hdr['quant']
    except:
        pass
    
    stitle = '%s, NSIDE %i' % (quant, nside)
    
    fig3 = plt.figure(fignum)
    fig3.clf()
    hp.mollview(planck, fignum, coord=['C', 'G'], title=stitle, \
                unit=hdr['quant'], cmap='plasma_r', norm='hist')
