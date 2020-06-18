#!/usr/bin/env bash
set -x
# Summary:
# Script to take Level 1eo CTX stereopairs, run them through NASA Ames stereo Pipeline.
# The script uses ASP's bundle_adjust tool to perform bundle adjustment on each stereopair separately.
# This script is capable of processing many stereopairs in a single run and uses GNU parallel
#  to improve the efficiency of the processing and reduce total wall time.


# Dependencies:
#      NASA Ames Stereo Pipeline
#      USGS ISIS3
#      GDAL
#      GNU parallel
# Optional dependency:
#      Dan's GDAL Scripts https://github.com/gina-alaska/dans-gdal-scripts
#        (used to generate footprint shapefile based on initial DEM)


# Just a simple function to print a usage message
print_usage (){
    echo ""
    echo "Usage: $0 -s <stereo.default> -p <productIDs.lis>"
    echo " Where <productIDs.lis> is a file containing a list of the IDs of the CTX products to be processed."
    echo " Product IDs belonging to a stereopair must be listed sequentially."
    echo " The script will search for CTX Level 1eo products in the current directory before processing with ASP."
    echo " "
    echo "<stereo.default> is the name and absolute path to the stereo.default file to be used by the stereo command."
}

### Check for sane commandline arguments

if [[ $# = 0 ]] || [[ "$1" != "-"* ]]; then
    # print usage message and exit
    print_usage
    exit 0
elif  [[ "$1" = "-"* ]]; then  # Else use getopts to parse flags that may have been set
    while getopts ":p:s:" opt; do
        case $opt in
          p)
              prods=$OPTARG
                if [ ! -e "$OPTARG" ]; then
                  echo "$OPTARG not found" >&2
                  # print usage message and exit
                  print_usage
                  exit 1
              fi
              ;;
          s)
              config=$OPTARG
              if [ ! -e "$OPTARG" ]; then
                  echo "$OPTARG not found" >&2
                  # print usage message and exit
                  print_usage
                  exit 1
              fi
              # Export $config so that GNU parallel can use it later
              export config=$OPTARG
              ;;
          \?)
              # Error to stop the script if an invalid option is passed
              echo "Invalid option: -$OPTARG" >&2
              exit 1
              ;;
          :)
              # Error to prevent script from continuing if flag is not followed by at least 1 argument
              echo "Option -$OPTARG requires an argument." >&2
              exit 1
              ;;
        esac
    done
fi

# If we've made it this far, commandline args look sane and specified files exist

# Check that ISIS has been initialized by looking for pds2isis,
#  if not, initialize it
# https://stackoverflow.com/questions/592620/check-if-a-program-exists-from-a-bash-script
if ! hash pds2isis 2>/dev/null; then
    echo "Initializing ISIS3"
    # shellcheck source=/dev/null
    source $ISISROOT/scripts/isis3Startup.sh
    # Quick test to make sure that initialization worked
    # If not, print an error and exit
    if ! hash pds2isis 2>/dev/null; then
        echo "ERROR: Failed to initialize ISIS3" 1>&2
        exit 1
    fi
fi

#######################################################
# figure out the number of cores/threads the pc has
# we assume we have hyperthreading so the cores should
# just be half the thread count
export num_threads_asp=`getconf _NPROCESSORS_ONLN`
export num_cores_asp=$((num_threads_asp / 2 ))
export num_procs_asp=$((num_cores_asp / 4 ))
#######################################################

######
echo "Start asp_ctx_lev1eo2dem $(date)"
#######################################################
## Housekeeping and Creating Some Support Files for ASP
#######################################################
# Create a 3-column, space-delimited file containing list of CTX stereo product IDs and the name of the corresponding directory that will be created for each pair
# For the sake of concision, we remove the the 2 character command mode indicator and the 1x1 degree region indicator from the directory name
awk '{printf "%s ", $0}!(NR % 2){printf "\n"}' $prods | sed 's/ /_/g' | awk -F_ '{print($1"_"$2"_"$3"_"$4"_"$5" "$6"_"$7"_"$8"_"$9"_"$10" "$1"_"$2"_"$3"_"$6"_"$7"_"$8)}' > stereopairs.lis

# Extract Column 3 (the soon-to-be- directory names) from stereopairs.lis and write it to a file called stereodirs.lis
# This file will be specified as an input argument for asp_ctx_map2dem.sh or asp_ctx_para_map2dem.sh
awk '{print($3)}' stereopairs.lis > stereodirs.lis

# Make directories named according to the lines in stereodirs.lis
awk '{print("mkdir -p "$1)}' stereodirs.lis | sh

# Now extract each line from stereopairs.lis (created above) and write it to a textfile inside the corresponding subdirectory we created on the previous line
# These files are used to ensure that the input images are specified in the same order during every step of `stereo` in ASP
awk '{print $1" "$2 > $3"/stereopair.lis"}' stereopairs.lis

####################################################################################
# Bundle adjust *all* of the input products
# You must modify the hard-coded bundle adjustment prefix inside the FOR loop below if you want to bundle adjust all of the input products together
# rather than running bundle_adjust separately on each stereopair.
# The practical difference between these two approaches is non-trivial. Consider yourself warned
#awk '{print($1".lev1eo.cub")}' $prods | tr '\n' ' ' | tr -s ' ' | awk '{print("bundle_adjust "$0" -o ba_results/ba")}' | sh
###################################################################################

# TODO: Test that the Level1eo cubes exist before trying to move them, throw error and exit if they don't exist
# Move the Level 1eo cubes into the directory named for the stereopair they belong to
awk '{print("cp -n "$1"*.lev1eo.cub "$3)}' stereopairs.lis | sh
awk '{print("cp -n "$2"*.lev1eo.cub "$3)}' stereopairs.lis | sh

# If this script is run as part of a job on a computing cluster using SLURM, we write the nodelist to a file named "nodelist.lis" so parallel_stereo can use it
if hash scontrol 2>/dev/null; then
    scontrol show hostname $SLURM_NODELIST | tr ' ' '\n' > nodelist.lis
fi

##  Run ALL stereo in series for each stereopair using `parallel_stereo`
# This is not the most resource efficient way of doing this but it's a hell of a lot more efficient compared to using plain `stereo` in series
for i in $( cat stereodirs.lis ); do
    cd $i || exit 1
    # Store the names of the Level1 EO cubes in variables
    L=$(awk '{print($1".lev1eo.cub")}' stereopair.lis)
    R=$(awk '{print($2".lev1eo.cub")}' stereopair.lis)
    Ladj=$(awk '{print($1".lev1eo.adjust")}' stereopair.lis)
    Radj=$(awk '{print($2".lev1eo.adjust")}' stereopair.lis)
    # Run ASP's bundle_adjust on the given stereopair
    echo "Begin bundle_adjust on \"$i\" at $(date)"
    bundle_adjust $L $R -o adjust/ba
    if [[ $? -ne 0 ]] || [[ ! -e "adjust/ba-${Ladj}" ]] || [[ ! -e "adjust/ba-${Radj}" ]]; then
        echo "Failure running bundle_adjust of $i at $(date)"
        exit 1
    else
        echo "Success running bundle_adjust of $i at $(date)"
    fi
    echo "Finished bundle_adjust on \"$i\" at $(date)"
    # Note that we specify ../nodelist.lis as the file containing the list of hostnames for `parallel_stereo` to use
    # You may wish to edit out the --nodes-list argument if running this script in a non-SLURM environment
    # See the ASP manual for information on running `parallel_stereo` with a node list argument that is suitable for your environment

    # We break parallel_stereo into 3 stages in order to optimize resource utilization. The first and third stages let parallel_stereo decide how to do this.
    # For the second stage, we specify an optimal number of processes and number of threads to use for multi-process and single-process portions of the code.
    # By default, we assume running on a machine with 16 cores. Users should tune this to suit their hardware.

    echo "Begin parallel_stereo on \"$i\" at $(date)"
    if [ -r ../nodelist.lis ]; then
        # stop parallel_stereo after correlation
        parallel_stereo --nodes-list=../nodelist.lis --stop-point 2 $L $R -s ${config} results_ba/${i}_ba --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 1 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 1 of $i at $(date)"
        fi
        # attempt to optimize parallel_stereo for running on 16-core machines for Steps 2 (refinement) and 3 (filtering)
        parallel_stereo --nodes-list=../nodelist.lis --processes ${num_procs_asp} --threads-multiprocess ${num_cores_asp} --threads-singleprocess ${num_threads_asp} --entry-point 2 --stop-point 4 $L $R -s ${config} results_ba/${i}_ba --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 2 & 3 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 2 & 3 of $i at $(date)"
        fi
        # finish parallel_stereo using default options for Stage 4 (Triangulation)
        parallel_stereo --nodes-list=../nodelist.lis --entry-point 4 $L $R -s ${config} results_ba/${i}_ba --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 4 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 4 of $i at $(date)"
        fi

    else
        parallel_stereo --processes ${num_procs_asp} --threads-multiprocess ${num_cores_asp} --threads-singleprocess ${num_threads_asp} --stop-point 4 $L $R -s ${config} results_ba/${i}_ba --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 1 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 1 of $i at $(date)"
        fi
        # Run Triangulation step will as many threads as possible as it can run n processes
        parallel_stereo  --entry-point 4 --processes ${num_threads_asp} --threads-multiprocess ${num_cores_asp} --threads-singleprocess ${num_threads_asp} $L $R -s ${config} results_ba/${i}_ba --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 2 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 2 of $i at $(date)"
        fi
    fi

    mkdir -p ./adjust/logs
    mv ./adjust/*-log-* ./adjust/logs/

    cd ../ || exit 1
    echo "Finished parallel_stereo on \"$i\" at $(date)"
done


# loop through the directories listed in stereodirs.lis and run point2dem, image footprint and hillshade generation
for i in $( cat stereodirs.lis ); do
    # cd into the directory containing the stereopair i
    cd $i || exit 1
    # extract the proj4 string from one of the image cubes using asap and store it in a variable (we'll need it later for point2dem)
    proj=$(awk '{print("asap get-srs-info "$1".lev1eo.cub")}' stereopair.lis | sh | sed 's/'\''//g') || exit 1

    # cd into the results directory for stereopair $i
    cd results_ba/ || exit 1
    # run point2dem with orthoimage and intersection error image outputs. no hole filling
    point2dem --threads ${num_threads_asp} --t_srs "${proj}" -r mars --nodata -32767 -s 24 -n --errorimage ${i}_ba-PC.tif --orthoimage ${i}_ba-L.tif -o dem/${i}_ba
    if [ $? -ne 0 ]
    then
        echo "Failure running point2dem at 24m/p for $i at $(date)"
        exit 1
    else
        echo "Success running point2dem at 24m/p for $i at $(date)"
    fi

    mkdir -p ./dem/logs
    mv ./dem/*-log-* ./dem/logs/

    # Generate hillshade (useful for getting feel for textural quality of the DEM)
    gdaldem hillshade ./dem/${i}_ba-DEM.tif ./dem/${i}_ba-hillshade.tif
    if [ $? -ne 0 ]
    then
        echo "Failure running gdaldem hillshade for $i at $(date)"
        exit 1
    else
        echo "Success running gdaldem hillshade for $i at $(date)"
    fi

    ## OPTIONAL ##
    # # Create a shapefile containing the footprint of the valid data area of the DEM
    # # This requires the `gdal_trace_outline` tool from the "Dan's GDAL Scripts" collection
    if hash gdal_trace_outline 2>/dev/null; then
        gdal_trace_outline dem/${i}_ba-DEM.tif -ndv -32767 -erosion -out-cs en -ogr-out dem/${i}_ba_footprint.shp
    fi

    #cleanup logs
    mkdir -p logs
    mv *-log-* ./logs/

    cd ../../ || exit 1
done
echo "End   asp_ctx_lev1eo2dem $(date)"
set +x
