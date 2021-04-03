# python #

2021-04-02 currently contains prototype routines for abutting together the predictions of Lallement+2019 and Bovy's "mwdust" module. 

Lallement et al. 2019 is queried using the routine **stilism_local.py**, written by Alessandro Mazzi and modified lightly by Will.

**WATCHOUT** - to run Bovy et al.'s **mwdust**, the environment variable DUST_DIR must be set. For more information, see the README at the **mwdust** github repository: 
https://github.com/jobovy/mwdust

For info about stilism, particularly the relevant papers and caveats, see here: https://stilism.obspm.fr/ but notice this seems to point to an older version of the map (which 
reports the gradient in E(B-V)). The updated version (which also reports extinction as A_555nm, or roughly A_V), can be found at Vizier: 
http://cdsarc.unistra.fr/viz-bin/cat/J/A+A/625/A135

