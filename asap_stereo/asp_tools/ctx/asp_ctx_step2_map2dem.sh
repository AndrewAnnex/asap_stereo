#!/usr/bin/env bash
set -x
# Summary:
# For a series of stereopairs, create a low-resolution DEM (100 m/px with 50 px hole-filling) from a bundle_adjust'd point cloud created by asp_ctx_lev1eo2dem.sh and
#  mapproject the Level 1eo images onto it. Then run the map-projected images through parallel_stereo.
# The purpose of this is to combine the benefits of bundle_adjust with those of proving ASP with map-projected images. This 2-step approach is meant to help remove some of the
#  spurious jagged edges that can appear on steep slopes in DEMs made from un-projected images.

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
    # echo " The script will search for CTX Level 1eo products in the current directory before processing with ASP."
    echo "<stereo.default> is the name and absolute path to the stereo.default file to be used by the stereo command."
    echo ""
}

### Check for sane commandline arguments

if [[ $# = 0 ]] || [[ "$1" != "-"* ]]; then
    # print usage message and exit
    print_usage
    exit 0
elif  [[ "$1" = "-"* ]]; then # Else use getopts to parse flags that may have been set
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

# Check that ISIS has been initialized already

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

######
echo "Starting asp_ctx_step2_map2dem $(date)"
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
awk '{print $1" "$2 >$3"/stereopair.lis"}' stereopairs.lis

# If this script is run as part of a job on Midway, we write the nodelist to a file named "nodelist.lis" so parallel_stereo can use it
# This line is NOT portable to environments that are NOT running SLURM
if hash scontrol 2>/dev/null; then
    scontrol show hostname $SLURM_NODELIST | tr ' ' '\n' > nodelist.lis
fi
#######################################################

#######################################################
# figure out the number of cores/threads the pc has
# we assume we have hyperthreading so the cores should
# just be half the thread count
export num_threads_asp=`getconf _NPROCESSORS_ONLN`
export num_cores_asp=$((num_threads_asp / 2 ))
export num_procs_asp=$((num_cores_asp / 4 ))
#######################################################

# Create low-resolution DEMs from point clouds created during earlier run
# loop through the directories listed in stereodirs.lis and run point2dem, image footprint and hillshade generation
for i in $( cat stereodirs.lis ); do
    # cd into the directory containing the stereopair i
    cd $i || exit 1

    # extract the proj4 string from one of the image cubes using asap and store it in a variable (we'll need it later for point2dem)
    proj=$(awk '{print("asap get-srs-info "$1".lev1eo.cub")}' stereopair.lis | sh | sed 's/'\''//g') || exit 1

    # cd into the results directory for stereopair $i
    cd results_ba/ || exit 1
    # run point2dem to create 100 m/px DEM with 50 px hole-filling
    point2dem --threads ${num_threads_asp} --t_srs "${proj}" -r mars --nodata -32767 -s 100 --dem-hole-fill-len 50 ${i}_ba-PC.tif -o dem/${i}_ba_100_fill50
    if [ $? -ne 0 ]
    then
        echo "Failure running point2dem at 100m/p for $i at $(date)"
        exit 1
    else
        echo "Success running point2dem at 100m/p for $i at $(date)"
    fi

    # Generate hillshade (useful for getting feel for textural quality of the DEM)
    gdaldem hillshade ./dem/${i}_ba_100_fill50-DEM.tif ./dem/${i}_ba_100_fill50-hillshade.tif
    if [ $? -ne 0 ]
    then
        echo "Failure running gdaldem hillshade for $i at $(date)"
        exit 1
    else
        echo "Success running gdaldem hillshade for $i at $(date)"
    fi
    cd ../../ || exit 1
done

##   Start the big bad FOR loop to mapproject the bundle_adjust'd images onto the corresponding low-res DEM and pass to parallel_stereo
for i in $( cat stereodirs.lis ); do
    cd $i || exit 1
    # Store the complete path to the DEM we will use as the basis of the map projection step in a variable called $refdem
    refdem=${PWD}/results_ba/dem/${i}_ba_100_fill50-DEM.tif

    # If the specified DEM does not exist or does not have nonzero size, throw an error and exit.
    if [ ! -s "$refdem" ]; then
        echo "The specified DEM does not exist or has zero size"
        echo $refdem
        exit 1
    fi
    # Store the names of the Level1 EO cubes in variables
    Lcam=$(awk '{print($1".lev1eo.cub")}' stereopair.lis)
    Rcam=$(awk '{print($2".lev1eo.cub")}' stereopair.lis)

    # ## Mapproject the CTX images against a specific DTM using the adjusted camera information
    echo "Projecting \"$Lcam\" against $refdem"
    awk -v refdem=$refdem -v L=$Lcam '{print("mapproject -t isis "refdem" "L" "$1".ba.map.tif --mpp 6 --bundle-adjust-prefix adjust/ba")}' stereopair.lis | sh
    if [ $? -ne 0 ]
    then
        echo "Failure running mapproject of Left image in $i at $(date)"
        exit 1
    else
        echo "Success running mapproject of Left image in $i at $(date)"
    fi
    echo "Projecting \"$Rcam\" against $refdem"
    awk -v refdem=$refdem -v R=$Rcam '{print("mapproject -t isis "refdem" "R" "$2".ba.map.tif --mpp 6 --bundle-adjust-prefix adjust/ba")}' stereopair.lis | sh
    if [ $? -ne 0 ]
    then
        echo "Failure running mapproject of Right image of $i at $(date)"
        exit 1
    else
        echo "Success running mapproject of Right image of $i at $(date)"
    fi
    # Store the names of the map-projected cubes in variables
    Lmap=$(awk '{print($1".ba.map.tif")}' stereopair.lis)
    Rmap=$(awk '{print($2".ba.map.tif")}' stereopair.lis)
    # Note that we specify ../nodelist.lis as the file containing the list of hostnames for `parallel_stereo` to use
    # You may wish to edit out the --nodes-list argument if running this script in a non-SLURM environment
    # See the ASP manual for information on running `parallel_stereo` with a node list argument that is suitable for your environment

    echo "Begin parallel_stereo on $i at $(date)"
    if [ -r ../nodelist.lis ]; then
        # for SLURM environment
        # stop parallel_stereo after correlation
        parallel_stereo --nodes-list=../nodelist.lis -t isis --stop-point 2 $Lmap $Rmap $Lcam $Rcam -s ${config} results_map_ba/${i}_map_ba --bundle-adjust-prefix adjust/ba $refdem
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 1 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 1 of $i at $(date)"
        fi
        # attempt to optimize parallel_stereo for running on the sandyb nodes (16 cores each) for Steps 2 (refinement) and 3 (filtering)
        parallel_stereo --nodes-list=../nodelist.lis -t isis --processes 2 --threads-multiprocess 8 --threads-singleprocess 16 --entry-point 2 --stop-point 4 $Lmap $Rmap $Lcam $Rcam -s ${config} results_map_ba/${i}_map_ba --bundle-adjust-prefix adjust/ba $refdem
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 2 & 3 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 2 & 3 of $i at $(date)"
        fi
        # finish parallel_stereo using default options for Stage 4 (Triangulation)
        parallel_stereo --nodes-list=../nodelist.lis -t isis --entry-point 4 $Lmap $Rmap $Lcam $Rcam -s ${config} results_map_ba/${i}_map_ba --bundle-adjust-prefix adjust/ba $refdem
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 4 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 4 of $i at $(date)"
        fi
    else
        # for single node environment
        parallel_stereo -t isis --processes ${num_procs_asp} --threads-multiprocess ${num_cores_asp} --threads-singleprocess ${num_threads_asp} --stop-point 4 $Lmap $Rmap $Lcam $Rcam -s ${config} results_map_ba/${i}_map_ba --bundle-adjust-prefix adjust/ba $refdem
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Steps 1-3 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Steps 1-3 of $i at $(date)"
        fi
        # finish Stage 4 with a lot more processes for speed
        parallel_stereo -t isis --entry-point 4 --processes ${num_threads_asp} --threads-multiprocess ${num_cores_asp} --threads-singleprocess ${num_threads_asp} $Lmap $Rmap $Lcam $Rcam -s ${config} results_map_ba/${i}_map_ba --bundle-adjust-prefix adjust/ba $refdem
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 4 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 4 of $i at $(date)"
        fi
    fi

    mkdir -p ./adjust/logs
    mv ./adjust/*-log-* ./adjust/logs/

    cd ../ || exit 1
    echo "Finished parallel_stereo on $i at $(date)"
done


# loop through the directories listed in stereodirs.lis and run point2dem, image footprint and hillshade generation
for i in $( cat stereodirs.lis ); do
    # cd into the directory containing the stereopair i
    cd $i || exit 1
    # extract the proj4 string from one of the image cubes using asap and store it in a variable (we'll need it later for point2dem)
    proj=$(awk '{print("asap get-srs-info "$1".lev1eo.cub")}' stereopair.lis | sh | sed 's/'\''//g') || exit 1
    # cd into the results directory for stereopair $i
    cd results_map_ba/ || exit 1
    # run point2dem with orthoimage and intersection error image outputs. no hole filling
    point2dem --threads ${num_threads_asp} --t_srs "${proj}" -r mars --nodata -32767 -s 24 ${i}_map_ba-PC.tif --orthoimage -n --errorimage ${i}_map_ba-L.tif -o dem/${i}_map_ba
    if [ $? -ne 0 ]
    then
        echo "Failure running point2dem at 24m/p for $i at $(date)"
        exit 1
    else
        echo "Success running point2dem at 24m/p for $i at $(date)"
    fi
    # Generate hillshade (useful for getting feel for textural quality of the DEM)
    gdaldem hillshade ./dem/${i}_map_ba-DEM.tif ./dem/${i}_map_ba-hillshade.tif
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
        gdal_trace_outline ./dem/${i}_map_ba-DEM.tif -ndv -32767 -erosion -out-cs en -ogr-out dem/${i}_map_ba_footprint.shp
    fi

    # cleanup logs
    mkdir -p logs
    mv *-log-* ./logs

    cd ../../ || exit 1
done
echo "Finished asp_ctx_step2_map2dem $(date)"
set +x
