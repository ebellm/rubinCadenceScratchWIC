# plots #

Random plots resulting from routines in progress.

2021-04-02 to reproduce the single line of sight plot in this folder, cd to a folder that contains stilism_cube_2.h5 and this version of stilism_local.py, and type

```
import compareExtinctions
compareExtinctions.hybridSightline(0, 4, figName='test_l0b4_ebvCompare.png', nl=7, nb=7, tellTime=True) 
```

2021-04-05 To reproduce the version with the Planck comparison, do the following:
```
l19 = stilism_local.LallementDustMap('19',3.1)
compareExtinctions.hybridSightline(-90., 65., nl=4, nb=4, setLimDynamically=False, useTwoBinnings=True, nBinsAllSightlines=500, nside=64, objL19=l19, figName='test_l270_b65_nside64_sub.png')
```
