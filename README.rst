ASAP-Stereo (Ames Stereo Automated Pipeline)
=============================================

.. image:: https://readthedocs.org/projects/asap-stereo/badge/?version=main
   :target: https://asap-stereo.readthedocs.io/en/main/?badge=main
   :alt: Documentation Status
   
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4171570.svg
   :target: https://doi.org/10.5281/zenodo.4171570
   :alt: Cite using doi.org/10.5281/zenodo.4171570

ASAP-Stereo is workflow wrapper for the `NASA Ames Stereo Pipeline`_, it is a reimplementation of the `asp_scripts`_ workflow to produce
co-registered CTX and HiRISE Digital Elevation Models (DEMs) from stereo image pairs in python and jupyter notebooks.

ASAP-Stereo currently only support the original MRO CTX/HiRISE stereo pair workflow from the asp_scripts, but could be built upon to make workflows for other imaging datasets.

*NOTE:* Advanced knowledge of the Ames Stereo Pipeline, Bash environment, and ISIS3 is a near prerequisite for using ASAP-Stereo.
No guarantees are implied for product quality or correctness, and it is not endorsed by the ASP developers. Proceed with caution.

Please cite!

.. _asp_scripts: https://github.com/USGS-Astrogeology/asp_scripts
.. _NASA Ames Stereo Pipeline: https://github.com/NeoGeographyToolkit/StereoPipeline

Docs
----

Documentation is available at https://asap-stereo.readthedocs.io 

Goals
-----

The aim of ASAP-Stereo is to enable reproducible workflows for ASP. Through rich logging and the jupyter notebook based workflows, users can distribute
the complete log of the steps they ran to produce a digital elevation model, and from that other users can run the same steps to duplicate results.
ASAP-Stereo, like the asp_scripts project, provides high level functions to execute many individual steps of the pipeline to reduce the complexity of producing a CTX or DEM to two steps.

In addition, workflows are broken down to dozens of granular steps that users can re-run with different parameters to fix issues as they arise.

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
reproducible and easy to interact with. Should things go wrong for a processing step, the Jupyter Notebooks
are easy to modify so steps can be re-run easily. The Jupyter Notebook workflow uses the same CLI interface
for ASAP-Stereo so the notebooks are the preferred way to interact with ASAP-Stereo.

Differences from asp_scripts
----------------------------
There are a few minor to major differences between the workflow in ASAP-Stereo and the asp_scripts default workflow that are
partially listed here.

1. Bundle Adjustment was added to the HiRISE workflow, possibly of marginal benefit.
2. PEDR data, and pedr2tab is optional as the ODE Rest API is used to get relevant data.
3. Image resolutions are encoded in file names in meters with an underscore for decimals (ie 0_25.tif would be 25 cm per pixel).
4. Hillshade alignment of the HiRISE dem to the target DEM is used to improve/speed up alignment of HiRISE to CTX.

.. _SLURM: https://slurm.schedmd.com

Installation
------------
ASAP can be installed using pip in a conda environment hosting ASP and ISIS. See the documentation for more details. 

Clone the repository, cd into the project and run: python setup.py install (or develop).

You must also have all of the other dependencies for ASP and the asp_scripts installed and available in the path to run.
You will also need GDAL command line tools,and ISIS3 installed. It is recommended to use anaconda to isolate the ASAP-Stereo
environment.

Usage
=====

Please checkout the documentation available at https://asap-stereo.readthedocs.io for more complete documentation and usage than below. 

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
`asap ctx notebook_pipeline_make_dem PRODUCTID1 PRODUCTID2 STEREO_CONF`


Final products from step 2 are located in the `PRODUCTID_PRODUCTID/results_map_ba/dem_align` folder. The
`PRODUCTID_PRODUCTID_map_ba_align_24_0-DEM-adj.tif` is the final DEM product,
and the map projected images are `PRODUCTID_PRODUCTID_map_ba_align_24_0-DRG.tif`
and `PRODUCTID_PRODUCTID_map_ba_align_6_0-DRG.tif`.

For HiRISE:
~~~~~~~~~~~
`asap hirise notebook_pipeline_make_dem ID1 ID2 STEREO_CONF REFDEM`

where REFDEM is the path to the CTX DEM (not the geoid adjusted "adj.tif" DEM) created before.

Final products from step 3 are located in the `PRODUCTID_PRODUCTID/results/dem_align` folder. The
`PRODUCTID_PRODUCTID_align_1_0-DEM-adj.tif` is the final DEM product, and the map projected images
are `PRODUCTID_PRODUCTID_align_1_0-DRG.tif` and `PRODUCTID_PRODUCTID_align_0_25-DRG.tif`.

Estimating Max-Disparity (--maxdisp)
------------------------------------
*note:*  As of 0.2.0 disparity is estimated for the user by ASAP, this note retained for cases when overrides needed.

The maximum disparity parameter used in both workflows above can be estimated by loading the reference and target
DEM products into a GIS environment (like QGIS) to determine the distance in the x, y, and z axes between the two products.
It is good practice to add a hundred meter margin to this estimate. For CTX use the PEDR CSV file to estimate it from MOLA,
for HiRISE use the final DEM.tif (non-geoid corrected) for the corresponding CTX pair.
