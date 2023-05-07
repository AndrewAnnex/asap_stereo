============
Installation
============

This guide assumes that the user has already installed anaconda with conda version newer than 4.9.

ASAP requires that both Ames Stereo Pipeline and ISIS are available to the user in the PATH, and the user will need to have downloaded the relevant ISIS data folders.

To start it is necessary to:

1. Create a dedicated ISIS conda environment following the instructions at https://github.com/DOI-USGS/ISIS3
2. Download and extract the pre-compiled Ames Stereo Pipeline binary following the first step in `Ames Stereo Pipeline Installation <https://stereopipeline.readthedocs.io/en/latest/installation.html#precompiled-binaries-linux-and-macos>`_   

Now we will make a 2nd conda environment dedicated for ASAP using the following conda yml file environment specfication (for background on this subject please read `the related conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_). 

In the variables section at the bottom, you will need to update the parameters for your environment.
The ISISROOT directory is the conda environment folder for your working ISIS installation, for example "~/anaconda/envs/isis/", it can also be found by activating the environment and printing the $CONDA_PREFIX variable.
The two ISIS data directories are explained in the `ISIS installation documentation <https://github.com/DOI-USGS/ISIS3>`_.
The ASPROOT directory similarly points to the unzipped archive of the Ames Stereo Pipeline.


.. code-block:: yaml

   name: asap_0_3_0
   channels:
     - conda-forge
     - defaults
   dependencies:
     - python>=3.11
     - ale
     - black=19.10b0
     - pvl
     - nb_conda_kernels
     - jupyterlab
     - requests
     - fire
     - tqdm
     - papermill
     - rasterio
     - pyproj
     - shapely
     - sh
     - pip
     - pip:
       - moody>=0.2.0
       - asap_stereo=0.3.1
   variables:
     ISISROOT: /path/to/your/isis/conda/env/
     ISISDATA: /path/to/your/isis/data/dir/
     ISISTESTDATA: /path/to/optional/isis/test/data/dir/
     ASPROOT: /path/to/your/precompiled/StereoToolkitDirectory/


Once the above yml file has been updated, create the environment using the command::

    conda env create --file asap.yml

This will create a new conda environment called "asap_0_3_0". The environment however is not ready to use yet.
We need to update the PATH for this conda environment to ensure ASP and ISIS programs are available for ASAP to use.
To do this run the following commands::

    conda activate asap_0_3_0
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


