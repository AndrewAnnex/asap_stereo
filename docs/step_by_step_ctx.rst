=======================================
Notebook Workflow Step-by-Step for CTX
=======================================

Now that we have run the Jupyter Notebook based workflows through the command line interface, we can look at each step that was run and describe what happened in more detail.
Note that the function docstrings are also available to describe the parameters of a given step, and what that step does.
Below is an export of all the codeblocks in the notebook workflow, additional markdown cells are included in the files but are not important to reproduce here.
This workflow replicates the same workflow used by the asp_scripts project.

Part 1: notebook_pipeline_make_dem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First define all the parameters for the notebook for papermill. The notebook includes a cell metadata tag for papermill to allow these parameters to be defined at runtime.
First we need the left and right image ids, the left image typically has the lower emission angle.
ASAP will check the metadata of the images to ensure the correct order is provided.
The config1 and config2 parameters are paths to stereo.default files the user has to configure the Ames Stereo Pipeline.
The first config file is the only required parameter, config2 gives you to use higher quality parameters for the 2nd pass CTX DEM.
The "dem_gsd" and "img_gsd" parameters control the number of pixels per pixel the final DEM and orthorectified images have.
These default to 24 and 6 meters per pixel which works for generally any CTX image pair.
Generally, most CTX images are captured at around 5.5 meters per pixel (GSD) so we pick 6 mpp as a reasonable default.
By convention, the DEM post spacing `should be at least 3X the image GSD <https://stereopipeline.readthedocs.io/en/latest/tools/point2dem.html?highlight=post%20spacing#post-spacing>`_.
ASAP defaults to 4X the image GSD to be a bit more conservative, resulting in 24 meters per pixel.
Output_path is typically left blank to default to the current working directory.
The maxdisp parameter controls the maximum expected disparity (distance) between the intermediate CTX DEM and the reference topography. Leaving this as 'None' will allow ASAP to estimate the disparity for you.
The downsample parameter allows you to downsample the imagery by a factor of the value to reduce processing times, a downsample of 4 will reduce the number of pixels by a factor of 4.
The pedr_list variable points to the local copy of a file containing a list of all the paths to all of the MOLA PEDR data.
By default this is set to None to use the ODE REST API to grab the necessary PEDR data, which is much faster anyways.

.. code:: ipython3

    left  = None
    right = None
    config1 = None
    config2 = None
    dem_gsd  = 24.0
    img_gsd  = 6.0
    output_path = None
    maxdisp = None
    downsample = None
    pedr_list = None

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
    from pathlib import Path
    from asap_stereo import asap
    import math

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

    !asap ctx step-1 {left} {right} 2>&1 | tee -i -a ./1_download.log ./full_log.log

Step 2: Preprocessing through ISIS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we replicate the preprocessing from the asp_scripts project/ames stereo pipeline using ISIS commands.
This step will run these steps in the following order: mroctx2isis, spiceinit, spicefit, ctxcal, ctxevenodd.

.. code:: ipython3

    !asap ctx step-2  2>&1 | tee -i -a ./2_ctxedr2lev1eo.log ./full_log.log

Step 3: Metadata init
~~~~~~~~~~~~~~~~~~~~~

Now we create a number of metadata files used by the asp_scripts project to simplify future command calls.
We also copy our preprocessed CTX cub files into a new working directory where all the stereo products will be computed.
This new directory name uses both image IDs joined by an underscore '{left_id}_{right_id}', for example: "B03_010644_1889_XN_08N001W_P02_001902_1889_XI_08N001W".

.. code:: ipython3

    !asap ctx step-3

Step 4: Bundle adjustment
~~~~~~~~~~~~~~~~~~~~~~~~~

We will use the `parallel_bundle_adjust <https://stereopipeline.readthedocs.io/en/latest/bundle_adjustment.html#bundle-adjustment>`_ command from Ames Stereo Pipeline to refine the spacecraft position and orientation.
The user can later re-run this step with more advanced options or GCPs if so desired.

.. code:: ipython3

    !asap ctx step-4 2>&1 | tee -i -a ./2_bundle_adjust.log ./full_log.log

Step 5: Stereo first run (steps 1-3 of stereo in ASP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can start making our first dem, we pass in the stereo config file to `parallel_stereo <https://stereopipeline.readthedocs.io/en/latest/tools/parallel_stereo.html>`_.
We split this into two parts (step 5 & 6) as we may want to run each part with slightly different parameters or give us a chance to inspect the outputs before the final step which can be long running.
In the future Step 5 & & maybe reconfigured into the 4 sub-steps for further improvement to the workflow.

.. code:: ipython3

    !asap ctx step-5 {config1}  2>&1 | tee -i -a ./3_lev1eo2dem.log ./full_log.log

Step 6: Stereo first run (step 4 of stereo in ASP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run step 4, see step 5 above for more information.

.. code:: ipython3

    !asap ctx step-6 {config1}  2>&1 | tee -i -a ./3_lev1eo2dem.log ./full_log.log

Step 7: Produce low resolution DEM for map projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have made a point cloud, but it is preliminary so we will use it to make a 100 mpp DEM to map-project the CTX images to, to produce a better 2nd pass DEM.

.. code:: ipython3

    !asap ctx step-7 --mpp 100 --just_dem True --dem_hole_fill_len 50 2>&1 | tee -i -a ./4_make_100m_dem.log ./full_log.log

Step 8: Make GoodPixelMap and Hillshade Previews
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We make image previews of the DEM using the next few steps to check for issues with our first pass DEM.
First we will render out the good pixel map image and then the hillshade of the DEM to look for issues with the topography.

.. code:: ipython3

    !asap ctx step-8

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

We now map-project our ctx images against our low resolution DEM to reduce image distortion for our 2nd pass DEM.

.. code:: ipython3

    !asap ctx step-9 --mpp {img_gsd} 2>&1 | tee -i -a ./5_mapproject_to_100m_dem.log ./full_log.log

Step 10: Stereo second run (steps 1-3 of stereo in ASP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same as step 5, just using the new map projected images this time.

.. code:: ipython3

    !asap ctx step-10 {config2} 2>&1 | tee -i -a ./6_next_level_dem.log ./full_log.log

Step 11: Stereo second run (step 4 of stereo in ASP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same as step 6, just using the new map projected images this time.

.. code:: ipython3

    !asap ctx step-11 {config2} 2>&1 | tee -i -a ./6_next_level_dem.log ./full_log.log

Step 7&8 again: create preview DEMs and Hillshade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have made our second point cloud, so we should export some visuals as before.
The parameter '--folder' just specifies that we are saving things into a different directory this time around.

.. code:: ipython3

    !asap ctx step-7 --folder results_map_ba

.. code:: ipython3

    !asap ctx step-8 --folder results_map_ba

Step 12: Get PEDR Shots for PC alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final important step in the make_dem workflow is to get the MOLA PEDR data for the region we care about.
Again, our data is not completely done until it has been aligned to the MOLA topography.
If we had GCPs in the bundle adjust stage this would not be as big of an issue, but since it is relatively easy to align to MOLA we don't
need to go through the process of producing GCPs.

.. code:: ipython3

    !asap ctx step-12 {pedr_list}  2>&1 | tee -i -a ./7_pedr_for_pc_align.log ./full_log.log

Make Final GoodPixelMap and Hillshade Previews
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing too surprising here, just export PNG versions of the images we care about to see the DEM at this stage of the processing.

.. code:: ipython3

    both = f'{left}_{right}'
    img = f'./{both}/results_map_ba/{both}_ba-GoodPixelMap.tif'
    out = img.replace('.tif', '.png')

.. code:: ipython3

    !gdal_translate -of PNG -co worldfile=yes {img} {out}

.. code:: ipython3

    Image(filename=out)

.. code:: ipython3

    both = f'{left}_{right}'
    img = f'./{both}/results_map_ba/dem/{both}_ba_24_0-DEM-hillshade.tif'
    out = img.replace('.tif', '.png')

.. code:: ipython3

    !gdal_translate -of PNG -co worldfile=yes {img} {out}

.. code:: ipython3

    Image(filename=out)

One additional bit here, for the MOLA data, show the PEDR2TAB template if created and the amount of PEDR data we have to align to.
If the final line is less than a few hundred we could be in a bad situation.

.. code:: ipython3

    !cat ./{left}_{right}/PEDR2TAB.PRM

.. code:: ipython3

    !cat ./{left}_{right}/{left}_{right}_pedr4align.csv | wc -l

Now that we have finished the first half of the workflow we can inspect the output products for issues before moving forwards.
If there are issues noted in the log or after a particular step, that step can be re-run with different parameters until a good solution is found.

At this point, we have a completed DEM! However, it's absolute position in space maybe off from the correct position.
Therefore, we must now perform a point cloud alignment to align our DEM with reference topography, in this case MOLA PEDR data `to correct the position of the CTX DEM <https://stereopipeline.readthedocs.io/en/latest/next_steps.html?highlight=ortho#alignment-to-point-clouds-from-a-different-source>`_.
In older versions of ASAP, this point is the dividing line between the make_dem and align_dem pipelines.

The "maxdisp" parameter in particular deserves attention.
It is the number passed to `pc_align's --max-displacement <https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html>`_ parameter in the Ames Stereo Pipeline.
Basically, it is the value of the distance you expect to move the CTX DEM to become aligned to your reference DEM (in this case, the PEDR data).
It is generally worth estimating this number using a GIS to sample points in both the DEM and reference file, and seeing how far away they are from each other.
But, CTX can be well behaved with ASP, so we pick a default of 500 meters which can be large enough for many situations.


Step 13: Align the DEM to MOLA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most important step in the 2nd half of the workflow as all the remaining steps are just producing final science products and visuals for the logs.
This step runs `pc_align <https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html>`_ using the provided max displacement (aka disparity). If the
logs indicate a larger displacement was observed than the user provided value it will need to be re-run using larger values or with other advanced parameters.
If users see issues it is generally easyier to re-run the pipeline at this step repeatedly in the command line or inside the Jupyter notebook.

.. code:: ipython3

    !asap ctx step_13 {maxdisp} 2>&1 | tee -i -a ./8_pc_align.log ./full_log.log

Step 14: Make the final CTX DEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the previous step everything after is simple and easy as we now have a final aligned point cloud from which DEMs and ortho images can be made.
That is all the rest of the steps do, they generate final DEMs with the geoid adjustment to produce science ready DEMs and ortho images for mapping.

.. code:: ipython3

    !asap ctx step_14 --mpp {demgsd}  2>&1 | tee -i -a ./9_dems_orthos.log ./full_log.log

Step 15: Adjust final CTX DEM to Geoid (Areoid)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !asap ctx step_15 2>&1 | tee -i -a ./10_geoid_adjustment.log  ./full_log.log

Make the final CTX Hillshade and Orthos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !asap ctx step_8 --folder results_map_ba --output_folder dem_align 2>&1 | tee -i -a ./11_hillshade.log ./full_log.log


.. code:: ipython3

    img = './' + str(next(Path('./').glob('./*/results_map_ba/dem_align/*_ba_align_24_0-DEM-hillshade.tif')))
    out = img.replace('.tif', '.png')


.. code:: ipython3

    !gdal_translate -of PNG -co worldfile=yes {img} {out}


.. code:: ipython3

    Image(filename=out)


.. code:: ipython3

    !asap ctx step_14 --mpp {imggsd} --just_ortho True  2>&1 | tee -i -a ./12_img_full_ortho.log ./full_log.log

