"""Utilities to read in and use the HEALPix map of 3D extinction.
Authors:
- Will Clarkson (wiclarks@umich.edu)
- Alessandro Mazzi (alessandro.mazzi@unipd.it)
"""

import os
import socket
from typing import Tuple, Union, List

import astropy.units as u
import healpy as hp
import matplotlib.pylab as plt
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from rubin_sim.maf.slicers import baseSlicer
from rubin_sim.maf.maps import baseMap

# To know if we are inside datalab, check the host name
hostname = socket.gethostname()
if "datalab" in hostname:
    datalab_server = True
else:
    datalab_server = False

# Check if we can use the datalab functionality
try:
    from dl import storeClient as sc

    datalab_available = True
except ImportError:
    datalab_available = False

py_folder = os.path.dirname(os.path.abspath(__file__))
extmaps_dir = os.path.join(os.path.dirname(py_folder), "extmaps")


class ebv3d(object):

    default_map = os.path.join(extmaps_dir, "merged_ebv3d_nside64_defaults.fits.gz")

    def __init__(self, mapPath: str = None, load=False, Verbose=True) -> None:
        """Class that loads and manipulates a merged 3d extinction map.
        Also reads in an (nrows, nbins) mask array, but currently does nothing with it.
        The default extinction map is `../extmaps/merged_ebv3d_nside64.fits`.

        Args:
            map_path (str, optional): Name of the file containing the map. Defaults to None.
            load (bool, optional): Determines whether the map is automatically loaded. Defaults to False.
            Verbose (bool, optional): Print additional information. Defaults to True.
        """

        # R_x factors for the LSST filters. This was generated with
        # the following line, using NOIRLAB Datalab sims_maf and
        # pasting the Dust_values object from the github sims_maf
        # 2021-04-05. From the jupyter notebook:
        #
        # tryThis = Dust_values(3.1, ref_ebv=1.)
        # tryThis.Ax1
        self.R_x = {"u": 4.80, "g": 3.67, "r": 2.70, "i": 2.06, "z": 1.59, "y": 1.31}

        # verbose?
        self.Verbose = Verbose

        # path to map
        if mapPath is None:
            self.mapPath = self.default_map
        else:
            self.mapPath = mapPath[:]

        self.initialized = False

        ### MAF SETTINGS
        self.keynames = ["ebv"]
        if load:
            self.loadMap()

    def loadMap(self, use_local: bool = None, use_sciserver: bool = None, use_datalab: bool = None) -> None:
        """Loads the extinction map from different sources.

        There are three methods to load the map, from three different sources. With the default arguments
        (all parameters equal to `None`) all of them are used in sequence, such that if one fails, the next
        one is used. The sequence is as follows.

        1) Load from the the given path. If using maps in the `extmap` folder, beware that `git` might have
        downloaded only a placeholder, please check and if required use
        `git lfs pull --include extmaps/<mapname>'.
        2) Load from a Sciserver public folder. The file name of the map, without parent folders, is used.
        3) Load from a Data Lab public folder. The file name of the map, without parent folders, is used.
        Beware that this works also when not using the Data Lab server! The map will be downloaded to memory.

        To change the behavior described above, three arguments exist (`use_*`). Setting any of them to
        `False` will exclude the source from being used. Setting one of them to `True` will throw an
        exception if the map cannot be loaded from the requested source.

        If the map cannot be loaded using any of the sources, an exception is raised.

        Args:
            use_local (bool, optional): Use a local file. Defaults to True.
            use_sciserver (bool, optional): Use a Sciserver file. Defaults to None.
            use_datalab (bool, optional): Use a Data Lab file. Defaults to None.
        """

        # Keep track of whether the map was successfully loaded at one of the steps
        hdul = None

        print("Loading the map.")

        # Check local map
        if use_local or use_local is None:
            if self.Verbose:
                print("Loading the map from local path.")
            try:
                hdul = fits.open(self.mapPath)
            except OSError:
                msg = "The map {} could not be loaded.".format(self.mapPath)
                if use_local:
                    raise FileNotFoundError(msg)
                if self.Verbose:
                    print(msg)

        # Check Sciserver public/shared folder
        # Works only on a Sciserver machine
        if hdul is None and (use_sciserver or use_sciserver is None):
            if self.Verbose:
                print("Loading the map from Sciserver.")
            map_name = os.path.split(self.mapPath)[1]
            # On sciserver maps are gzipped
            if not map_name.endswith(".gz"):
                map_name = map_name + ".gz"
            map_path = os.path.join("/home/idies/workspace/lsst_cadence/LSST_Shared/extmaps", map_name)
            try:
                hdul = fits.open(map_name)
            except OSError:
                msg = "The map {} could not be loaded from Sciserver.".format(map_name)
                if use_sciserver:
                    raise FileNotFoundError(msg)
                if self.Verbose:
                    print(msg)

        # Check Data Lab public folder
        # Works always as long as `noaodatalab` is installed
        if hdul is None and (use_datalab or use_datalab is None):
            if self.Verbose:
                print("Loading the map from Data Lab.")
            if not datalab_available and use_datalab:
                raise ImportError("The `noaodatalab` package is not available, cannot load the map.")
            else:
                # If importing the datalab package succeeded we can try to query the public map file on Data
                # Lab space.
                # WARNING: this works even locally, the datalab package will just download the map in memory.
                map_name = os.path.split(self.mapPath)[1]
                # On the public space there should be only the compressed maps
                if not map_name.endswith(".gz"):
                    map_name = map_name + ".gz"
                map_path = "thalos12://public/{}".format(map_name)
                try:
                    hdul = fits.open(sc.get(map_path, mode="fileobj"))
                except requests.exceptions.MissingSchema:
                    msg = "The map {} could not be loaded from Data Lab.".format(map_name)
                    # This exception is raised when the file does not exist on datalab, as far as I have
                    # understood
                    if use_datalab:
                        raise FileNotFoundError(msg)
                    if self.Verbose:
                        print(msg)
        # If no source provided a map, cannot continue
        if hdul is None:
            raise FileNotFoundError("The map could not be loaded!")

        print("Loaded the map.")

        self.hdr = hdul[0].header
        # get extinction map information
        # parse the header for the healpix information we need.
        self.nside = self.hdr["NSIDE"]
        self.nested = self.hdr["NESTED"]

        # now populate the arrays in turn
        self.hpids = hdul[0].data
        self.dists = hdul[1].data
        self.ebvs = hdul[2].data

        if len(hdul) > 3:
            self.sfacs = hdul[3].data
        else:
            self.sfacs = None

        if len(hdul) > 4:
            self.mask = hdul[4].data
        else:
            self.mask = None

        hdul.close()

        self._calcDistmods()

        self.initialized = True

    def _calcDistmods(self) -> None:
        """Converts distances in parsecs to distance moduli in magnitudes"""
        self.dmods = 5.0 * np.log10(self.dists) - 5.0

    def getMapNearestDist(self, distPc=3000) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the distances and extinctions closest to the supplied distance.

        Args:
            distPc (int, optional): Target distance. Defaults to 3000.

        Returns:
            Tuple[np.ndarray,np.ndarray]: Tuple containing two arrays, one for E(B-V) and another one for
            their distances.
        """

        # just use argmin
        imin = np.argmin(np.abs(self.dists - distPc), axis=1)

        # now lift the ebv values at this distance
        iExpand = np.expand_dims(imin, axis=-1)
        distsClosest = np.take_along_axis(self.dists, iExpand, axis=-1).squeeze()
        ebvsClosest = np.take_along_axis(self.ebvs, iExpand, axis=-1).squeeze()

        return ebvsClosest, distsClosest

    # TODO Make into function independent on the map.
    def getEBVatSightline(
        self,
        gall: Union[float, np.ndarray],
        galb: Union[float, np.ndarray],
        ebvMap: np.ndarray = None,
        interp=False,
        showEBVdist=False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns E(B-V) for one or more sightlines in Galactic coordinates. Takes as an argument a 2D
        healpix array of quantities (usually this will be reddening returned by getMapNearestDist() ). Also
        returns the nearest coords to the requested coords for debug purposes

        Args:
            l (Union[float,np.ndarray]): Galactic longitude of the sightline(s).
            b (Union[float,np.ndarray]): Galactic latitude of the sightline(s).
            ebvMap (np.ndarray): 2D reddening map to use. Defaults to None.
            interp (bool, optional): Interpolate the result to the exact coordinates using HEALPix routines.
                If False, returns the value at the closest pixel. Defaults to False.
            showEBVdist (bool, optional): Only for debugging purposes. If True, plots the run of E(B-V) vs
                distance for the nearest hpid (ignored if interp=True). Defaults to False.

        Returns:
            Tuple[np.ndarray,np.ndarray,np.ndarray]: Tuple containing: E(B-V) values, galactic longitude and
            galactic latitude. The latter two are the input ones if `interp=True`, otherwise they correspond
            to the closest pixel.
        """

        if ebvMap is not None:
            # Accept only 2D extinction maps.
            if np.size(ebvMap) < 1:
                return np.array([0.0]), np.array([-99.0]), np.array([-99.0])
            # Accept only full extinction maps.
            if not len(ebvMap) == hp.nside2npix(self.nside):
                raise ValueError(
                    "The extinction map is not complete! It contains {} of {} pixels at nside {}]".format(
                        len(ebvMap), hp.nside2npix(self.nside), self.nside
                    )
                )
        else:
            ebvMap = self.ebvs

        # find the coords on the sky of the requested sight line, and
        # convert this to healpix
        coo = SkyCoord(gall * u.deg, galb * u.deg, frame="galactic")

        # Equatorial coordinates of the requested position(s)
        ra = coo.icrs.ra.deg
        dec = coo.icrs.dec.deg

        if interp:
            ebvRet = hp.get_interp_val(ebvMap, ra, dec, nest=self.nested, lonlat=True)
            # For backwards compatibility with the "Test" return values
            lTest = np.copy(gall)
            bTest = np.copy(galb)
        else:
            hpid = hp.ang2pix(self.nside, ra, dec, nest=self.nested, lonlat=True)
            ebvRet = ebvMap[hpid]
            # For debugging: determine the coordinates at this nearest pixel
            raTest, decTest = hp.pix2ang(self.nside, hpid, nest=self.nested, lonlat=True)
            cooTest = SkyCoord(raTest * u.deg, decTest * u.deg, frame="icrs")
            lTest = cooTest.galactic.l.degree
            bTest = cooTest.galactic.b.degree
            # also for debugging: show the run of E(B-V) vs distance
            # for the nearest healpix ID:
            if showEBVdist and np.isscalar(hpid):
                fig9 = plt.figure(9)
                fig9.clf()
                ax9 = fig9.add_subplot(111)
                ax9.plot(self.dists[hpid], self.ebvs[hpid])
                ax9.set_xlabel("Distance, pc")
                ax9.set_ylabel("E(B-V)")
                ax9.grid(which="both", visible=True, alpha=0.5)
                ax9.set_title("hpid: %i, (l,b) = (%.2f, %.2f)" % (hpid, lTest, bTest))

        return ebvRet, lTest, bTest

    def getMaxDistDeltaMag(
        self, dmagVec: Union[float, np.ndarray], sfilt="r", ipix: Union[float, np.ndarray] = None
    ) -> np.ndarray:
        """Computes the maximum distance for an apparent m-M using the maximum
        value of extinction from the map.

        Args:
            dmagVec (Union[float, np.ndarray]): Apparent m-M value(s).
            sfilt (str, optional): Filter to use. Defaults to 'r'.
            ipix (Union[float, np.ndarray], optional): Pixel(s) to use. Defaults to None.

        Returns:
            Union[float, np.ndarray]: Values of maximum distance.
        """
        # We do distance modulus = (m-M) - A_x, and calculate the distance from the result. We do this for
        # every sightline at once.
        if ipix is not None:
            ebvsMax = self.R_x[sfilt] * self.ebvs[ipix, -1]
        else:
            ebvsMax = self.R_x[sfilt] * self.ebvs[:, -1]
        distModsFar = dmagVec - ebvsMax
        distsFar = 10.0 ** (0.2 * distModsFar + 1.0)
        return np.atleast_1d(distsFar)

    def getDeltaMag(self, sFilt="r", ipix: Union[float, np.ndarray] = None) -> Union[float, np.ndarray]:
        """Converts the reddening map into an (m-m0) map for the given filter.

        Args:
            sfilt (str, optional): Filter to use. Defaults to 'r'.
            ipix (Union[float, np.ndarray], optional): Pixel(s) to use. Defaults to None.

        Returns:
            Union[float, np.ndarray]: Values of maximum distance.
        """
        if sFilt not in self.R_x.keys():
            sFilt = "r"
        Rx = self.R_x[sFilt]
        if ipix is not None:
            mMinusm0 = self.dmods[np.newaxis, ipix] + Rx * self.ebvs[ipix, :]
            # make 3d so that from the outside nothing changes
        else:
            mMinusm0 = self.dmods[np.newaxis, :] + Rx * self.ebvs

        return mMinusm0[0]

    def getDistanceAtMag(
        self, deltamag=15.2, sfilt="r", ipix: Union[int, np.ndarray] = None, extrapolateFar=True
    ):
        """Returns the distances at which the combination of distance and extinction produces the input
        magnitude difference (m-M) = deltamag.

        If ipix is provided, either as a single int or a list/array of ints representing Healpix pixel
        indices, only the number of pixels requested will be queried. Arrays will be returned in any case,
        even when one single pixel is requested.

        Args:
            deltamag (Union[float, np.ndarray], optional): Target (m-m_0). If array, must have the same
                number of elements as the extinction map (i.e. hp.nside2npix(self.nside)). Defaults to 15.2.
            sfilt (str, optional): Filter to use. Defaults to 'r'.
            ipix (Union[float, np.ndarray], optional): Pixel(s) to use. Defaults to None.
            extrapolateFar (bool, optional): For distances beyond the maximum distance in the model, treat
                the extinction as constant beyond that maximum distance and compute the distance at which
                delta-mag is achieved. Only set to False if you know what you are doing! Defaults to True.

        Returns:
            Tuple[np.ndarray,np.ndarray,np.ndarray]: Tuple containg the array of distances in parsecs, the
                array of magnitude differences and the array of bools indicating whether the maximum
                distance indicated by a sight line was beyond the range of validity of the extinction model
                and the result was obtained via extrapolation
        """

        # A little bit of parsing...
        # If deltamag is a scalar, replicate it into an array.
        # Otherwise just reference the array that was passed in.
        # For the moment, trust the user to have inputted a deltamag vector of the right shape.
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
            print("WARNING - ebv3d.getDistanceAtMag - size mismatch:", npix, np.shape(dmagVec))
            return np.array([]), np.array([]), np.array([])

        # Now we need apparent minus absolute magnitude:
        mMinusM = self.getDeltaMag(sfilt, ipix=ipix)

        # Now we find elements in each row that are closest to the
        # requested deltamag:
        iMin = np.argmin(np.abs(mMinusM - dmagVec[:, np.newaxis]), axis=1)
        iExpand = np.expand_dims(iMin, axis=-1)

        # select only m-M at needed distance
        mMinusM = np.take_along_axis(mMinusM, iMin[:, np.newaxis], -1).flatten()

        # now find the closest distance...
        if ipix is not None:
            distsClosest = np.take_along_axis(self.dists[ipix], iExpand, axis=-1).squeeze()
            # To keep things similar to the case of querying the whole map,
            #  we need to return an array also in case ipix is a single pixel.
            distsClosest = np.atleast_1d(distsClosest)
            # if npix>1:
            #     distsClosest = distsClosest.squeeze()
        else:
            distsClosest = np.take_along_axis(self.dists, iExpand, axis=-1).squeeze()

        # 2021-04-09: started implementing distances at or beyond the
        # maximum distance. Points for which the closest delta-mag is
        # in the maximum distance bin are picked.
        if ipix is not None:
            bFar = iMin == self.dists[ipix].shape[-1] - 1
        else:
            bFar = iMin == self.dists.shape[-1] - 1
        if extrapolateFar:
            # For distances beyond the max, we use the maximum E(B-V)
            # along the line of sight to compute the distance.

            distsFar = self.getMaxDistDeltaMag(mMinusM, sfilt, ipix)

            # Now we swap in the far distances
            distsClosest[bFar] = distsFar[bFar]

        # ... Let's return both the closest distances and the map of
        # (m-M), since the user might want both.
        return distsClosest, mMinusM, bFar

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

    def getInterpolatedEBV(self, gall, galb, dist):
        """Get an interpolated extinction profile for one or more sightlines at one or more distances.

        Beware of the shapes!!

        - For one sightline and one distance: all arguments floats or 1D arrays of length 1.
        - For one sightline at multiple distances: `gall` and `galb` floats or 1D arrays of length 1, `dist`
            1D array of unspecified length.
        - For multiple sightlines, each one single distance distance: `gall`, `galb` and `dist` 1D arrays
            with same length.
        - For multiple sightlines at one distance: `gall` and `galb` arrays of length N, `dist` 1D array of
            length N created from e.g. np.repeat.
        - For multiple sightlines, each multiple distances: NOT IMPLEMENTED

        Args:
            gall (Union[float,np.ndarray]): Galactic longitudes.
            galb (Union[float,np.ndarray]): Galactic longitudes.
            dist (Union[float,np.ndarray]): Target distances.

        Returns:
            Tuple[np.ndarray,np.ndarray]: Tuple containing E(B-V) values and respective distances.
        """
        gall = np.atleast_1d(gall)
        galb = np.atleast_1d(galb)

        if not np.shape(gall) == np.shape(galb):
            raise ValueError("The number of longitudes and latitudes provided are not equal.")

        if (np.size(dist) > 1) and (np.size(gall) > 1):
            raise ValueError("Cannot get more than one distance sample when asking for multiple sightlines.")
            return np.array(-99), np.array(-99)

        dist = np.atleast_2d(dist).T  # required for subtraction to 2d array with shape (N,N_samples)
        N = max(len(gall), len(dist))

        coo = SkyCoord(gall * u.deg, galb * u.deg, frame="galactic")
        RAs = coo.icrs.ra.deg
        DECs = coo.icrs.dec.deg
        hpids, weights = hp.get_interp_weights(self.nside, RAs, DECs, self.nested, lonlat=True)
        # hpids, weights = hp.get_interp_weights(64, gall, galb, False, lonlat=True)
        ebvout = np.zeros(N)
        distout = np.zeros(N)
        for i in range(hpids.shape[0]):
            pid = hpids[i]
            w = weights[i]
            distID = np.argmin(np.abs(self.dists[pid] - dist), axis=1)
            ebvout_i = self.ebvs[pid, distID] * w
            ebvout += ebvout_i
            distout_i = self.dists[pid, distID] * w
            distout += distout_i
        return ebvout, distout

    def showDistanceInterval(self, fignum=5, cmap="viridis"):
        """Utility function that shows the map of distance resolutions for close and far points. Currently
        this just uses the difference betweeh bin 1 and 0 as the bin spacing for L+19, and between bins -2
        and -1 for the bovy et al. spacing.

        Args:
            fignum (int, optional): Number of the figure to use. Defaults to 5.
            cmap (str, optional): Colormap to use. Defaults to 'viridis'.
        """

        ddistClose = self.dists[:, 1] - self.dists[:, 0]
        ddistFar = self.dists[:, -1] - self.dists[:, -2]

        fig5 = plt.figure(fignum, figsize=(10, 3))
        fig5.clf()

        # set the margins
        margins = (0.02, 0.05, 0.05, 0.00)

        hp.mollview(
            ddistClose,
            fignum,
            coord=["C", "G"],
            nest=self.nested,
            title="Nearest distance bin width, pc",
            unit=r"$\Delta d$, pc",
            cmap=cmap,
            sub=(1, 2, 1),
            margins=margins,
        )

        hp.mollview(
            ddistFar,
            fignum,
            coord=["C", "G"],
            nest=self.nested,
            title="Farthest distance bin width, pc",
            unit=r"$\Delta d$, pc",
            cmap=cmap,
            sub=(1, 2, 2),
            margins=margins,
        )


class ebv3dMap(baseMap):
    def __init__(self, extmap: ebv3d = None, mapPath=None):
        if extmap is None:
            extmap = ebv3d(mapPath=mapPath)
            if mapPath is not None:
                extmap.loadMap(use_local=True)
        else:
            if not extmap.initialized:
                raise ValueError("The class has not been initialized.")

        self.extmap = extmap

        ### MAF SETTINGS
        self.keynames = ["ebv"]

    def run(self, slicePoints: baseSlicer, dist=None) -> baseSlicer:
        """Entry point for MAF.

        Args:
            slicePoints (baseSlicer): Slice points to use.

        Returns:
            baseSlicer: Slicer with E(B-V) for each point.
        """
        # Code takes inspiration from
        # https://github.com/lsst/rubin_sim/blob/main/rubin_sim/maf/maps/dustMap.py

        # If the slicer has nside, it's a healpix slicer so we can read the map directly
        if "nside" in slicePoints:
            nside = slicePoints["nside"]
            pix = slicePoints["sid"]
            # slicePoints['ebv'] = EBVhp(slicePoints['nside'], pixels=slicePoints['sid'],
            #    mapPath=self.mapPath)
            # ! Healpix slices have RING scheme
            ra, dec = hp.pix2ang(nside, pix, lonlat=True)
        # Not a healpix slicer, look up values based on RA,dec with possible interpolation
        else:
            ra, dec = slicePoints["ra"], slicePoints["dec"]
            coords = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
            gall = coords.galactic.l.degree
            galb = coords.galactic.b.degree

        if "dist" not in slicePoints:
            raise ValueError("Must provide distance for 3D map evaluation.")

        ebv, dist = self.getInterpolatedEBV(gall, galb, slicePoints["dist"])

        slicePoints["ebv"] = ebv

        return slicePoints


class MaxDistDeltaMagMap(baseMap):
    def __init__(self, extmap: ebv3d = None, mapPath=None):
        if extmap is None:
            extmap = ebv3d(mapPath=mapPath)
            if mapPath is not None:
                extmap.loadMap(use_local=True)
        else:
            if not extmap.initialized:
                raise ValueError("The class has not been initialized.")

        self.extmap = extmap

        ### MAF SETTINGS
        self.keynames = ["maxdist"]

    def run(self, slicePoints: baseSlicer) -> baseSlicer:
        """Entry point for MAF.

        Args:
            slicePoints (baseSlicer): Slice points to use.

        Returns:
            baseSlicer: Slicer with E(B-V) for each point.
        """

        maxdist = self.extmap.getMaxDistDeltaMag(
            slicePoints["dmag"], slicePoints["sfilt"], slicePoints["sid"]
        )

        slicePoints["maxdist"] = maxdist

        return slicePoints


class DistanceAtMAgMap(baseMap):
    def __init__(self, extmap: ebv3d = None, mapPath=None):
        if extmap is None:
            extmap = ebv3d(mapPath=mapPath)
            if mapPath is not None:
                extmap.loadMap(use_local=True)
        else:
            if not extmap.initialized:
                raise ValueError("The class has not been initialized.")

        self.extmap = extmap

        ### MAF SETTINGS
        self.keynames = ["dist"]

    def run(self, slicePoints: baseSlicer) -> baseSlicer:
        """Entry point for MAF.

        Args:
            slicePoints (baseSlicer): Slice points to use.

        Returns:
            baseSlicer: Slicer with E(B-V) for each point.
        """

        maxdist = self.extmap.getDistanceAtMag(slicePoints["mag"], slicePoints["sfilt"], slicePoints["sid"])

        slicePoints["maxdist"] = maxdist

        return slicePoints
