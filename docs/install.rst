============
Installation
============

This guide assumes that the user has already installed anaconda with conda version newer than 4.9.

ASAP requires that Install both Ames Stereo Pipeline and ISIS are available to the user in the PATH, and the user will need to have downloaded the relevant ISIS data folders.
Below we provide a guide assuming the user has a working ISIS installation in some other conda environment, and a a local copy of the pre-compiled Ames Stereo Pipeline.
Please refer to the `Ames Stereo Pipeline Installation <https://stereopipeline.readthedocs.io/en/latest/installation.html#precompiled-binaries-linux-and-macos>`_ documentation for details on how to download the precompiled ASP.

We will create a new conda environment for ASAP that includes the bin directories in the PATH for ASAP and the ISIS data folder environment variables.
In the variables section, you will need to update the parameters for your environment.
The ISISROOT directory is the conda environment folder for your working ISIS installation, for example "~/anaconda/envs/isis/", it can also be found by activating the environment and printing the $CONDA_PREFIX variable.
The two ISIS data directories are explained in the `ISIS installation documentation <https://github.com/USGS-Astrogeology/ISIS3>`_.
The ASPROOT directory similarly points to the unzipped archive of the Ames Stereo Pipeline.


.. code-block:: yaml

   name: asap_0_2_0
   channels:
     - conda-forge
     - defaults
   dependencies:
     - python=3.8
     - nb_conda_kernels
     - jupyterlab
     - requests
     - fire
     - tqdm
     - sh
     - papermill
     - rasterio
     - pyproj
     - shapely
     - pvl
     - pip 
     - pip:
       - moody>=0.0.4
       - asap_stereo=0.2.0
   variables:
     ISISROOT: /path/to/your/isis/conda/env/
     ISISDATA: /path/to/your/isis/data/dir/
     ISISTESTDATA: /path/to/optional/isis/test/data/dir/
     ASPROOT: /path/to/your/precompiled/StereoToolkitDirectory/


Using the command::

    conda env create --file asap.yml

This will create a new conda environment called "asap_0_2_0". The environment however is not ready to use yet.
We need to update the path to ensure ASP and ISIS programs are available for ASAP to use.
To do this run the following commands::

    conda activate asap_0_2_0
    conda env config vars set PATH=$PATH:$ASPROOT/bin:$ISISROOT/bin
    conda deactivate

This will update the PATH variable in the conda environment to include both ISIS and ASP bin folders.
At this point you are ready to start using asap!

To test if asap is installed correctly, run each of the following after activating the new environment::

    asap
    asap ctx
    asap hirise
    asap common
    
Each of these commands should show the help page for the asap modules and shouldn't print any errors.


