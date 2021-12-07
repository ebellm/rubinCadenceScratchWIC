import os
from readExtinction_MAF import ebv3d, extmaps_dir
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from plotting import getColorbarLimits
import math
import matplotlib


def testReadExt(
    showExtn=False, sfilt="r", showDeltamag=False, figName="test_mapDust.png", pathMap=None, norm="log"
):

    """Tests whether we can read the extinction map we just created. If
showExtn, then the extinction at filter sfilt is shown. If showDeltamag, then the quantity (m-M) is plotted,
including extinction. pathMap is the path to the E(B-V) vs distance map. Example call:

    readExtinction.testReadExt(sfilt='r', showExtn=True, \
    figName='testmap_Ar.png')

    """
    if pathMap is None:
        pathMap = os.path.join(extmaps_dir, "merged_ebv3d_nside64.fits")
        print("Using the default map at", pathMap)

    ebv = ebv3d(pathMap)
    ebv.loadMap()

    # print(np.shape(ebv.mask))

    # ebvThis, distThis = ebv.getMapNearestDist(dpc)

    ## try showing the scale factors
    # hp.mollview(ebv.sfacs, 3, coord=['C', 'G'], nest=ebv.nested)

    # print(ebv.sfacs[0:10], np.min(ebv.sfacs), np.max(ebv.sfacs))

    fig2 = plt.figure(2, figsize=(14, 8))
    fig2.clf()

    # fig3=plt.figure(3, figsize=(7,7))
    # fig3.clf()

    # Must change the margins, the default ones are too thin and text
    # overflows from the image
    # follow the scheme -> (left,bottom,right,top)
    # margins = (0.075, 0.075, 0.075, 0.05) # from mollview function
    margins = (0.0, 0.02, 0.00, 0.02)  # valid for figsize=(14,8)

    rx = 1.0
    sUnit = "E(B-V), mag"
    cmap = "plasma_r"
    if showExtn and sfilt in ebv.R_x.keys():
        rx = ebv.R_x[sfilt]
        sUnit = r"A$_%s$, mag" % (sfilt)
        # cmap = 'Greys'

    dpcs = [252.0, 1503.5, 4000.0, 7500.0]
    for iDist in range(len(dpcs)):
        ebvThis, distThis = ebv.getMapNearestDist(dpcs[iDist])

        vecSho = ebvThis * rx

        if showDeltamag and showExtn:
            dmod = 5.0 * np.log10(distThis) - 5.0
            vecSho += dmod
            sUnit = r"(m-M)$_%s$" % (sfilt)

        # if log scheme requested, make vecSho a masked array
        if norm.find("log") > -1:
            vecSho = np.ma.masked_less_equal(vecSho, 0.0)
            # on my system, mollview still doesn't handle masked
            # arrays well...
            vecSho[vecSho.mask] = np.ma.min(vecSho)

        # Show the dust map
        hp.mollview(
            vecSho,
            2,
            coord=["C", "G"],
            nest=ebv.nested,
            title="Requested distance %.1f pc" % (dpcs[iDist]),
            unit=sUnit,
            cmap=cmap,
            sub=(2, 2, iDist + 1),
            norm=norm,
            margins=margins,
        )

        cbar = plt.gca().images[-1].colorbar
        cmin, cmax = getColorbarLimits(cbar)
        # The colorbar has log scale, which means that cmin=0 is not valid
        # this should be handled by mollview, if not cmin is replaced by the
        # smallest non-zero value of the array vecSho
        if cmin == 0:
            cmin = np.amin(vecSho[vecSho != 0])
        # Set tick positions and labels
        cmap_ticks = np.logspace(math.log10(cmin), math.log10(cmax), num=5)
        cbar.set_ticks(cmap_ticks, True)
        cmap_labels = ["{:4.3g}".format(t) for t in cmap_ticks]
        cbar.set_ticklabels(cmap_labels)
        # Change the position of the colorbar label
        text = [c for c in cbar.ax.get_children() if isinstance(c, matplotlib.text.Text) if c.get_text()][0]
        print(text.get_position())
        text.set_y(-2.5)  # valid for figsize=(14,8)

        hp.graticule(alpha=0.5, color="0.25")

        # show the distance between the nearest distance bin and the
        # requested distance
        # hp.mollview(distThis - dpcs[iDist], 3, \
        #            coord=['C','G'], nest=ebv.nested, \
        #            title='Requested distance %.1f pc' % (dpcs[iDist]), \
        #            unit='Distance - requested, pc', \
        #            cmap='RdBu_r', sub=(2,2,iDist+1))

    fig2.suptitle("NSIDE=%i" % (ebv.nside))
    fig2.savefig(figName)


def testDeltamags(
    sfilt="r",
    dmagOne=13.0,
    figName="test_deltamag.png",
    cmap="viridis",
    norm="linear",
    pathMap=None,
    dmagVec=np.array([]),
    testMethod=False,
    testFigureMethod=False,
    testFarDistances=True,
    maxDistShow=None,
):

    """Use the extinction map to find the distance at which a particular
delta-mag is found.

    2021-03-07 new arguments:

    testMethod: tests using the method in object ebv3d to find the distance. Currently defaults to False so
    that the notebooks on the repository will work.

    testFigureMethod: tests using the method inside the object to show a healpy mollview. Currently defaults
    to False so that hte notebook on the repository will work.

    Example call:

    Find the distance in parsecs at which (m-M)_i = 15.2, using a
    stepped colormap until I work out how to add tickmarks to
    the colorbar...:

    readExtinction.testDeltamags('i', 15.2, cmap='Set2', \
    figName='testmap_delta_i_set1.png')

    """

    if pathMap is None:
        pathMap = os.path.join(extmaps_dir, "merged_ebv3d_nside64.fits")
        print("Using the default map at", pathMap)

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
        iMin = np.argmin(np.abs(mMinusM - dmagVec[:, np.newaxis]), axis=1)
        iExpand = np.expand_dims(iMin, axis=-1)

        # print("INFo:", np.shape(mMinusM), np.shape(iMin))
        # return

        # get the distances closest to this
        distsClosest = np.take_along_axis(ebv.dists, iExpand, axis=-1).squeeze()
    else:
        if np.size(dmagVec) < 1:
            dmagInp = dmagOne
        else:
            dmagInp = np.copy(dmagVec)

        distsClosest, mMinusM, bFar = ebv.getDistanceAtMag(dmagInp, sfilt, extrapolateFar=testFarDistances)

    if testFigureMethod:
        ebv.showMollview(distsClosest, 4, cmap=cmap, norm=norm, coord=["C", "G"], sUnit="Distance (pc)")
        return

    fig4 = plt.figure(4, figsize=(8, 6))
    fig4.clf()
    sTitle = r"Distance at $\Delta$%s=%.2f (%s scale)" % (sfilt, dmagOne, norm)
    hp.mollview(
        distsClosest,
        4,
        coord=["C", "G"],
        nest=ebv.nested,
        title=sTitle,
        unit="Distance (pc)",
        cmap=cmap,
        norm=norm,
        max=maxDistShow,
    )

    cbar = plt.gca().images[-1].colorbar
    cmin, cmax = getColorbarLimits(cbar)
    # The colorbar has log scale, which means that cmin=0 is not valid
    # this should be handled by mollview, if not cmin is replaced by the
    # smallest non-zero value of the array vecSho
    if cmin == 0:
        cmin = np.amin(sfilt[sfilt != 0])
    # Set tick positions and labels
    cmap_ticks = np.linspace(cmin, cmax, num=9)
    cbar.set_ticks(cmap_ticks, True)
    cmap_labels = ["{:5.0f}".format(t) for t in cmap_ticks]
    cbar.set_ticklabels(cmap_labels)
    cbar.ax.tick_params(labelsize=10)
    # Change the position of the colorbar label
    text = [c for c in cbar.ax.get_children() if isinstance(c, matplotlib.text.Text) if c.get_text()][0]
    print(text.get_position())
    text.set_y(-3.0)  # valid for figsize=(8,6)

    # show a graticule
    hp.graticule(color="0.2", alpha=0.5)

    fig4.suptitle("NSIDE=%i, Filter:%s" % (ebv.nside, sfilt))
    fig4.savefig(figName)


def testShowDistresol(pathMap=None):
    """Test our method to show the distance resolution"""

    if pathMap is None:
        pathMap = os.path.join(extmaps_dir, "merged_ebv3d_nside64.fits")
        print("Using the default map at", pathMap)

    ebv = ebv3d(pathMap)
    ebv.loadMap()
    ebv.showDistanceInterval()


def testGetOneSightline(gall=0.0, galb=0.0, dpc=3000.0, pathMap=None, interpCoo=False, showVsDistance=True):

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
        pathMap = os.path.join(extmaps_dir, "merged_ebv3d_nside64.fits")
        print("Using the default map at", pathMap)
    # load the map and compute the E(B-V) at the distance
    ebv = ebv3d(pathMap)
    ebv.loadMap()

    # Report some information to the screen about the map
    print(
        "ebv metadata info: NESTED: %i, NSIDE:%i, pixFillFac:%.2f, nL=%i, nB=%i"
        % (ebv.hdr["NESTED"], ebv.hdr["NSIDE"], ebv.hdr["fracPix"], ebv.hdr["nl"], ebv.hdr["nb"])
    )

    # Also report additional metadata which might or might not appear
    # in the version on my webspace, but which is now generated by
    # loopSightlines. See compareExtinctions.loopSightlines for the
    # header keywords that are now added to the 3d map.
    print("Additional header keywords:")
    for skey in ["Rv", "mapvers", "PlanckOK", "dmaxL19", "bridgL19", "bridgwid"]:
        try:
            print(skey, ebv.hdr[skey])
        except KeyError:
            pass

    ebvs, dists = ebv.getMapNearestDist(dpc)

    # Now we've obtained the map at a given distance, we can query
    # particular sight-lines. Let's try the coords requested
    ebvHere, lTest, bTest = ebv.getEBVatSightline(
        gall, galb, ebvs, interp=interpCoo, showEBVdist=showVsDistance
    )

    # For testing purposes, we can also find the distance
    # corresponding to our sight line using exactly the same method:
    distHere, _, _ = ebv.getEBVatSightline(gall, galb, dists, interp=interpCoo, showEBVdist=True)

    # If the coords are scalars, report to screen.
    if np.isscalar(gall):

        print(
            "Info: E(B-V), distance at (l,b, distance) nearest to (%.2f, %.2f, %.1f) are %.2f, %.1f"
            % (gall, galb, dpc, ebvHere, distHere)
        )

        print("Info: nearest Galactic coords to requested position: %.2f, %.2f" % (lTest, bTest))

    else:
        print("INFO: requested l:", gall)
        print("INFO: requested b:", galb)
        print("INFO: nearest l:", lTest)
        print("INFO: nearest b:", bTest)
        print("INFO: returned E(B-V)", ebvHere)
        print("INFO: returned distances:", distHere)


def testInteprolateProfile(gall, galb, dist, ebvmap=None):
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
