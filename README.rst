ASAP-Stereo
===========

Wraps the `asp_scripts`_ in python. Note: currently requires python 3.6 or newer.

.. _asp_scripts: https://github.com/USGS-Astrogeology/asp_scripts


Installation
------------
NOTE: you need to use python3.6 or newer!

Clone the repository, cd into the project and run: python setup.py
install.

You must also have all of the other dependencies for ASP and the asp_scripts installed and available in the path to run.


Usage
-----

Once installed, asap-stereo is available in the command line as the command `asap`.
Try this out, and see the list of sub-commands that are available in the output of running `asap`.
When you want to produce a DEM, make a new directory and `cd` into it, all commands run relative to the current working directory.



CTX-HiRISE Workflow
-------------------

1. Pick two CTX and/or HiRISE images as stereo pairs and note their product ids, which can be partial.

You can either start with CTX or HiRISE, but note that the third HiRISE step requires a reference DEM
like the once produced from CTX.

For CTX:
--------
1. `asap ctx-one PRODUCTID PRODUCTID`
   where "PRODUCTID" are the product ids you want to process
2. `asap ctx-two STEREO_CONF PEDR_LIST`
   where "STEREO_CONF" is the absolute path to your stereo.conf file,
   "PEDR_LIST" is a file that contains the absolute paths for all of the MOLA PEDR files.
3. `asap ctx-three MAX_DISP`
   where "MAX_DISP" is the maximum allowable displacement
   in meters that you find between MOLA and the `dem` folder products from step two.
   To find a value for the displacement, load the `PRODUCTID_PRODUCTID/results_map_ba/dem/PRODUCTID_PRODUCTID_map_ba-DEM.tif` file into
   a GIS environment and compare the vertical and horizontal displacement to a MOLA basemap.

Final products from step 3 are located in the `PRODUCTID_PRODUCTID/results_map_ba/dem_align` folder. The
`PRODUCTID_PRODUCTID_map_ba_align_24-DEM-adj.tif` is the final DEM product, and the map projected images are `PRODUCTID_PRODUCTID_map_ba_align_24-DRG.tif` and `PRODUCTID_PRODUCTID_map_ba_align_6-DRG.tif`

For HiRISE:
-----------
1. `asap hirise-one PRODUCTID PRODUCTID`
   where "PRODUCTID" are the product ids you want to process
2. `asap hirise-two STEREO_CONF`
   where "STEREO_CONF" is the absolute path to your stereo.conf file
3. `asap hirise-three MAX_DISP REF_DEM`
   where "MAX_DISP" is the maximum allowable displacement
   in meters that you find between MOLA and the `dem` folder products from step two.
   To find a value for the displacement, load the `PRODUCTID_PRODUCTID/results/dem/PRODUCTID_PRODUCTID-DEM.tif` file into
   a GIS environment and compare the vertical and horizontal displacement to the `-DEM.tif` CTX final file corresponding to the image.
   "REF_DEM" is the absolute path to the whatever non-geoid corrected DEM you want to align to HiRISE (same one used for displacement estimate).

Final products from step 3 are located in the `PRODUCTID_PRODUCTID/results/dem_align` folder. The
`PRODUCTID_PRODUCTID_align_1-DEM-adj.tif` is the final DEM product, and the map projected images are `PRODUCTID_PRODUCTID_align_1-DRG.tif` and `PRODUCTID_PRODUCTID_align_025-DRG.tif`
