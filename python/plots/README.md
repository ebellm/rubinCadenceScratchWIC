# plots #

Random plots resulting from routines in progress.

2021-04-02 to reproduce the single line of sight plot in this folder, cd to a folder that contains stilism_cube_2.h5 and this version of stilism_local.py, and type

```
import compareExtinctions
compareExtinctions.hybridSightline(0, 4, figName='test_l0b4_ebvCompare.png', nl=7, nb=7, tellTime=True) 
```
