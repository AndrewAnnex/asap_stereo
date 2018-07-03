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

Final products are located in the `dem_align` folder.

For HiRISE:
-----------
1. `asap hirise-one PRODUCTID PRODUCTID`
   where "PRODUCTID" are the product ids you want to process
2. `asap hirise-two STEREO_CONF`
   where "STEREO_CONF" is the absolute path to your stereo.conf file
3. `asap ctx-three MAX_DISP REF_DEM`
   where "MAX_DISP" is the maximum allowable displacement
   in meters that you find between MOLA and the `dem` folder products from step two.
   and "REF_DEM" is the absolute path to the whatever non-geoid corrected DEM you want to align to HiRISE.
   For CTX processed above it will be the "1-DEM.tif" file in the "dem_align" final products from the last step.

Final products are located in the `dem_align` folder.
