============
Tutorial
============

ASAP CTX Notebook Workflow Step-by-Step
---------------------------------------


Overview
^^^^^^^^
Now that we have run the Jupyter Notebook based workflows through the command line interface, we can look at each step that was run and describe what happened in more detail.
Note that the function docstrings are also available to describe the parameters of a given step, and what that step does.
Below is an export of all the codeblocks in the notebook workflow, additional markdown cells are included in the files but are not important to reproduce here.
This workflow replicates the same workflow used by the asp_scripts project.

Part 1: notebook_pipeline_make_dem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First define all the parameters for the notebook for papermill. The notebook includes a cell metadata tag for papermill to allow these parameters to be defined at runtime.
First we need the left and right image ids, the left image typically has the lower emission angle.
The pedr_list variable points to the local copy of a file containing a list of all the paths to all of the PEDR data, but can be left as None to use the ODE REST API which is much faster anyways.
The config1 and config2 parameters are paths to stereo.default files the user has to configure the Ames Stereo Pipeline.
Output_path is typically left blank to default to the current working directory.

.. code:: ipython3

    left  = None
    right = None
    pedr_list = None
    config1 = None
    config2 = None
    output_path = None

Check if config2 was defined, if it was not just use the first config file again

.. code:: ipython3

    if config2 == None:
        config2 = config1

For next two lines, print out the config into the notebook file.

.. code:: ipython3

    !cat {config1}

.. code:: ipython3

    !cat {config2}

Import a few python things to use later on, including the Image function to render images out into the notebook

.. code:: ipython3

    from IPython.display import Image
    from asap_stereo import asap

If the user did not specify a output directory, make one. Note this step only does something if the output_path is explicitly set to None.
By default from the command-line interface ASAP will use the current working directory.

.. code:: ipython3

    default_output_dir = '~/auto_asap/ctx/'
    left, right = asap.CTX().get_ctx_order(left, right)
    if output_path == None:
        output_path = default_output_dir + f'a_{left}_{right}'

Make that directory if needed.

.. code:: ipython3

    !mkdir -p {output_path}

Make sure the notebook is now running in that directory

.. code:: ipython3

    %cd {output_path}


Step 1: Download images
~~~~~~~~~~~~~~~~~~~~~~~~

Now we are getting to the heart of the notebook workflow. First use step-one to download our left and right images using the moody tool.
At the end of the command you can see we are using standard bash to redirect stdout and stderr to two log files, the first a log just for this step, the second a cumulative log file for the whole job.

.. code:: ipython3

    !asap ctx step-one {left} {right} 2>&1 | tee -i -a ./1_download.log ./full_log.log

Step 2: Preprocessing through ISIS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we replicate the preprocessing from the asp_scripts project/ames stereo pipeline using ISIS commands.
This step will run these steps in the following order: mroctx2isis, spiceinit, spicefit, ctxcal, ctxevenodd.

.. code:: ipython3

    !asap ctx step-two  2>&1 | tee -i -a ./2_ctxedr2lev1eo.log ./full_log.log

Step 3: Metadata init
~~~~~~~~~~~~~~~~~~~~~

Now we create a number of metadata files used by the asp_scripts project to simplify future command calls.
We also copy our preprocessed CTX cub files into a new working directory where all the stereo products will be computed.
This new directory name uses both image IDs joined by an underscore '{left_id}_{right_id}', for example: "B03_010644_1889_XN_08N001W_P02_001902_1889_XI_08N001W".

.. code:: ipython3

    !asap ctx step-three

Step 4: Bundle adjustment
~~~~~~~~~~~~~~~~~~~~~~~~~

We will use the `parallel_bundle_adjust <https://stereopipeline.readthedocs.io/en/latest/bundle_adjustment.html#bundle-adjustment>`_ command from Ames Stereo Pipeline to refine the spacecraft position and orientation.
The user can later re-run this step with more advanced options or GCPs if so desired.

.. code:: ipython3

    !asap ctx step-four 2>&1 | tee -i -a ./2_bundle_adjust.log ./full_log.log

Step 5: Stereo first run (steps 1-3 of stereo in ASP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can start making our first dem, we pass in the stereo config file to `parallel_stereo <https://stereopipeline.readthedocs.io/en/latest/tools/parallel_stereo.html>`_.
We split this into two parts (step 5 & 6) as we may want to run each part with slightly different parameters or give us a chance to inspect the outputs before the final step which can be long running.
In the future Step 5 & & maybe reconfigured into the 4 sub-steps for further improvement to the workflow.

.. code:: ipython3

    !asap ctx step-five {config1}  2>&1 | tee -i -a ./3_lev1eo2dem.log ./full_log.log

Step 6: Stereo first run (step 4 of stereo in ASP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run step 4, see step 5 above for more information.

.. code:: ipython3

    !asap ctx step-six {config1}  2>&1 | tee -i -a ./3_lev1eo2dem.log ./full_log.log

Step 7: Produce low resolution DEM for map projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have made a point cloud, but it is preliminary so we will use it to make a 100 mpp DEM to map-project the CTX images to, to produce a better 2nd pass DEM.

.. code:: ipython3

    !asap ctx step-seven --mpp 100 --just_dem True --dem_hole_fill_len 50 2>&1 | tee -i -a ./4_make_100m_dem.log ./full_log.log

Step 8: Make GoodPixelMap and Hillshade Previews
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We make image previews of the DEM using the next few steps to check for issues with our first pass DEM.
First we will render out the goodpixelmap image and then the hillshade of the DEM to look for issues with the topography.

.. code:: ipython3

    !asap ctx step-eight

Use some python to specify a new file name for the png version

.. code:: ipython3

    both = f'{left}_{right}'
    img = f'./{both}/results_ba/{both}_ba-GoodPixelMap.tif'
    out = img.replace('.tif', '.png')

Use gdal_translate to produce a png version of the hillshade image.

.. code:: ipython3

    !gdal_translate -of PNG -co worldfile=yes {img} {out}

Display the image in the notebook.

.. code:: ipython3

    Image(filename=out)

Now again for the hillshade

.. code:: ipython3

    both = f'{left}_{right}'
    img = f'./{both}/results_ba/dem/{both}_ba_100_0-DEM-hillshade.tif'
    out = img.replace('.tif', '.png')

Convert to a png file again.

.. code:: ipython3

    !gdal_translate -of PNG -co worldfile=yes {img} {out}

Display the image in the notebook.

.. code:: ipython3

    Image(filename=out)

Step 9: Mapproject ctx against 100m DEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !asap ctx step-nine 2>&1 | tee -i -a ./5_mapproject_to_100m_dem.log ./full_log.log

Calculate Better DEM using prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !asap ctx step-ten {config2} 2>&1 | tee -i -a ./6_next_level_dem.log ./full_log.log

.. code:: ipython3

    !asap ctx step-eleven {config2} 2>&1 | tee -i -a ./6_next_level_dem.log ./full_log.log

.. code:: ipython3

    !asap ctx step-seven --folder results_map_ba

.. code:: ipython3

    !asap ctx step-eight --folder results_map_ba

Get PEDR Shots for PC alignment (Step 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !asap ctx step-twelve {pedr_list}  2>&1 | tee -i -a ./7_pedr_for_pc_align.log ./full_log.log


Good Pixel Preview
##################

.. code:: ipython3

    both = f'{left}_{right}'
    img = f'./{both}/results_map_ba/{both}_ba-GoodPixelMap.tif'
    out = img.replace('.tif', '.png')

.. code:: ipython3

    !gdal_translate -of PNG -co worldfile=yes {img} {out}

.. code:: ipython3

    Image(filename=out)

Hillshade of higher res DEM
###########################

.. code:: ipython3

    both = f'{left}_{right}'
    img = f'./{both}/results_map_ba/dem/{both}_ba_24_0-DEM-hillshade.tif'
    out = img.replace('.tif', '.png')

.. code:: ipython3

    !gdal_translate -of PNG -co worldfile=yes {img} {out}

.. code:: ipython3

    Image(filename=out)


.. code:: ipython3

    !cat ./{left}_{right}/PEDR2TAB.PRM

.. code:: ipython3

    !cat ./{left}_{right}/{left}_{right}_pedr4align.csv | wc -l