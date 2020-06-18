ASAP-Stereo (Ames Stereo Automated Pipeline)
=============================================

ASAP-Stereo is workflow wrapper for the `NASA Ames Stereo Pipeline`_, it is a reimplementation of the `asp_scripts`_ workflow to produce
co-registered CTX and HiRISE Digital Elevation Models (DEMs) from stereo image pairs in python and jupyter notebooks.

ASAP-Stereo currently only support the origional MRO CTX/HiRISE stereo pair workflow from the asp_scripts, but could be built upon to make workflows for other imaging datasets.

*NOTE:* Advanced knowledge of the Ames Stereo Pipeline, Bash environment, and ISIS3 is a near pre-requisit for using ASAP-Stereo.
No guarentees are implied for product quality or correctness, and it is not endorsed by the ASP developers. Proceed with caution.


.. _asp_scripts: https://github.com/USGS-Astrogeology/asp_scripts
.. _NASA Ames Stereo Pipeline: https://github.com/NeoGeographyToolkit/StereoPipeline

Goals
-----

The aim of ASAP-Stereo is to enable reproducable workflows for ASP. Through rich logging and the jupyer notebook based workflows, users can distribute
the complete log of steps they ran to produce a digital elevation model, and that other users can run to duplicate results.
ASAP-Stereo, like the asp_scripts project, provides high level functions to execute many individual steps of the pipeline to reduce the complexity of producing a CTX or DEM to two steps.

In addition, workflows are broken down to dozens of granular steps that users can re-run with different parameters to fix issues as they arrise.

High Level Workflow
-------------------

ASAP-Stereo provides both a command line interface and jupyter notebook based workflows that are also executable using the
command line interface, accessible using the `asap` command.
Generally, users should first make the CTX DEM for a location they are interested in and verify it is correctly registered to MOLA topography. Then users
can proceed to produce a HiRISE DEM, and aligning it to the CTX DEM in the 2nd step for good co-registration of the HiRISE DEM.

The advantage of the CLI interface is that there are many individual steps in the workflow that sometimes
need to be re-run with different parameters when things don't go well. However, often this is only one or
two steps in the workflow, such as stereo correlation or point cloud alignment that can be run in relative
isolation of the other steps in the pipeline. Different steps require varying degrees of granularity as
some steps are much faster than others.

The advantage of the Jupyter Notebook workflows are that the configuration parameters and preview images are richly displayed
along side logs of the commands that were run so that workflows are both
reproduceable and easy to interact with. Should things go wrong for a processing step, the Jupyter Notebooks
are easy to modify so steps can be re-run easily. The Jupyter Notebook workflow uses the same CLI interface
for ASAP-Stereo so the notebooks are the prefered way to interact with ASAP-Stereo.

Differences from asp_scripts
----------------------------
There are a few minor to major differences between the workflow in ASAP-Stereo and the asp_scripts default workflow that are
partially listed here.

1. `SLURM`_ support was removed, but could likely be added back.
2. Bundle Adjustment was added to the HiRISE workflow, possibly of marginal benefit.
3. PEDR data, and pedr2tab is optional as the ODE Rest API is used to get relevant data.
4. Image resolutions are encoded in file names in meters with an underscore for decimals (ie 0_25.tif would be 25 cm per pixel).
5. Hillshade alignment of the HiRISE dem to the target DEM is used to improve/speed up alignment of HiRISE.

.. _SLURM: https://slurm.schedmd.com

Installation
------------
Clone the repository, cd into the project and run: python setup.py install (or develop).

You must also have all of the other dependencies for ASP and the asp_scripts installed and available in the path to run.
You will also need GDAL command line tools, Imagemagick, and ISIS3 installed. It is recommendend to use anaconda to isolate the ASAP-Stereo
environment.

Usage
=====

Once installed, asap-stereo is available in the command line as the command `asap`.
Try this out, and see the list of sub-commands that are available in the output of running `asap`.
When you want to produce a DEM, make a new directory and `cd` into it, all commands run relative to the current working directory.

New CTX-HiRISE Workflow (with notebooks)
----------------------------------------
`nohup` is recommended for each of the following steps for long running processes.

Pick two CTX and/or HiRISE images as stereo pairs and note their product ids, which can be partial.

You can either start with CTX or HiRISE, but note that the third HiRISE step requires a reference DEM
like the once produced from CTX.

Detailed help for each command can be viewed by running: `asap PREFIX COMMAND -- --help`,
for example `asap ctx notebook_pipeline_make_dem -- --help`.

For CTX:
~~~~~~~~
1. `asap ctx notebook_pipeline_make_dem PRODUCTID1 PRODUCTID2 STEREO_CONF`
2. `asap ctx notebook_pipeline_align_dem`

Final products from step 2 are located in the `PRODUCTID_PRODUCTID/results_map_ba/dem_align` folder. The
`PRODUCTID_PRODUCTID_map_ba_align_24_0-DEM-adj.tif` is the final DEM product,
and the map projected images are `PRODUCTID_PRODUCTID_map_ba_align_24_0-DRG.tif`
and `PRODUCTID_PRODUCTID_map_ba_align_6_0-DRG.tif`

For HiRISE:
~~~~~~~~~~~
1. `asap hirise notebook_pipeline_make_dem ID1 ID2 STEREO_CONF`
2. `asap hirise notebook_pipeline_align_dem REFDEM`
    where REFDEM is the path to the CTX DEM (non "adj.tif") created before.

Final products from step 3 are located in the `PRODUCTID_PRODUCTID/results/dem_align` folder. The
`PRODUCTID_PRODUCTID_align_1_0-DEM-adj.tif` is the final DEM product, and the map projected images
are `PRODUCTID_PRODUCTID_align_1_0-DRG.tif` and `PRODUCTID_PRODUCTID_align_0_25-DRG.tif`

Estimating Max-Disperity
------------------------
The maximum disperity parameter used in both `align_dem` steps can be estimated by loading the reference and target
DEM products into a GIS environment (like QGIS) to determine the distance in the x, y, and z axes between the two products.
It is good practice to add a hundred meter margin to this estimate. For CTX use the PEDR CSV file to estimate it from MOLA,
for HiRISE use the final DEM.tif (non-geoid corrected) for the corresponding CTX pair.
