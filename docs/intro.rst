============
Introduction
============

ASAP-Stereo (Ames Stereo Automated Pipeline) is a Python API
that orchestrates the NASA Ames Stereo Pipeline (ASP_) for HiRISE and CTX stereo image processing. 
ASAP implements functions that use various command-line tools from GDAL, ASP, and USGS ISIS to 
produce Digital Elevation Models (DEMs) following a workflow of steps described by
Mayer and Kite 2016 in their tool asp_scripts_. 
The workflows generally follow a procedure of first downloading and calibrating the data, 
producing stereo products using ASP, and then finally aligning the stereo products with 
reference elevation data and the geoid. The commonality in workflow procedures and the
number of steps involved mean that users will significantly benefit from the workflow's 
automation. ASAP also provides both a command-line interface for the whole API 
and Jupyter Notebooks, accessible through the command-line interface, that executes the workflows 
for CTX and HiRISE (described below). By providing all three interfaces, users can execute workflows,
and then re-run individual steps to improve stereo product quality and 
run the ASAP workflows in a semi-automatic fashion. 

.. _asp_scripts: https://github.com/USGS-Astrogeology/asp_scripts
.. _ASP: https://github.com/NeoGeographyToolkit/StereoPipeline

ASAP Core Functionality
-----------------------
ASAP uses the sh_ python library to access external command-line tools 
as if they were Python functions without any direct subprocess management code in ASAP.
Although using Python to call other subprocesses is not new, 
the sh project makes it remarkably easy to do this with minimal code.
Using sh enables ASAP to reuse existing command-line tools 
and to use Python for tasks it is best suited. 

.. _sh: https://github.com/amoffat/sh


ASAP's Commandline Interface (CLI)
----------------------------------

ASAP's CLI is automatically created from the Python classes
and functions using the python-fire_ project,
requiring no code changes to the Python API.
Python-fire allows ASAP to simultaneously provide both the 
API and CLI with no additional code to maintain.
For example, using python-fire, it is possible to enable tab-completion
for ASAP's CLI or drop into interactive python sessions. 
The python-fire_ GitHub page is an excellent resource to 
learn more about the project. 

.. _python-fire: https://github.com/google/python-fire


ASAP Jupyter Notebook Workflows
-------------------------------

ASAP provides Jupyter Notebook template workflows to users,
and they are the easiest way to use ASAP. 
ASAP uses the Papermill_ project to execute the notebook workflows, 
that project can turn any Jupyter Notebook into a parameterized command-line tool.
Papermill uses the workflows provided by ASAP as templates
that are then updated to use the user's specified images and parameters.
When executed, the notebooks capture all the logs from the various steps 
of ASAP and the underlying tools into the notebook file. 
In addition to the logs, the notebooks also include images generated during 
the workflow's steps for quick quality inspection by the user.
Users can share notebooks, allowing for reproducibility and traceability in their scientific data.
More details about the notebook workflows will be provided in the tutorials section of the documentation.


.. _Papermill: https://github.com/nteract/papermill
