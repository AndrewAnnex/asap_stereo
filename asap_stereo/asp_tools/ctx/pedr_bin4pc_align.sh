#!/usr/bin/env bash
set -x
# Script to generate pc_align-compatible CSV from MOLA PEDR binary files
# Filters input CSV by lat/long bounds of a map-projected ISIS3 cube
# Also all_calls_args `proj` to convert lat/long values into Easting/Northing and output a corresponding CSV so that PEDR shots can easily be compared to a preliminary DEM in QGIS

# This script is meant to be called from the root directory of a stereoproject.
# It assumes that the root directory contains a file named "stereodirs.lis" which lists the names of the directories containing the actual stereo data

# DEPENDENCIES
#      USGS ISIS3
#      GDAL
#      GNU parallel
#      pedr2tab (compiled from the USGS-modified version available here:
#                   https://github.com/USGS-Astrogeology/socet_set/blob/master/SS4HiRISE/Software/ISIS3_MACHINE/SOURCE_CODE/pedr2tab.PCLINUX.f)
#
#      The user must also have a copy of at least a subset of the PEDR binary files whose locations are listed in a file passed as an argument to this script

##################################

# Just a simple function to print a usage message
print_usage (){
    echo ""
    echo "Usage: $0 <pedr_list.lis>"
    echo " Where <pedr_list.lis> is a file containing a list of the MOLA PEDR binary files to search through, including absolute path."
    echo " <pedr_list.lis> itself should should be specified using the absolute path to its location."
    echo "  This file is required by pedr2tab."
    echo ""
}

### Check for sane commandline arguments and verify that several essential files exist ###
if [[ $# != 1 ]] ; then
    # print usage message and exit
    print_usage
    exit 0
fi

# store the first argument in a variable called $pedr_list
pedr_list=$1

# Quick test to see if the file $pedr_list exists and is a regular file
if [ ! -f "$pedr_list" ]; then
    echo "$1 not found"
    print_usage
    exit 1
fi

# Test that pedr2tab exists and is executable
if ! hash pedr2tab 2>/dev/null; then
    echo "pedr2tab not found in PATH or is not executable" 1>&2
    echo "Make sure you have added pedr2tab to your PATH variable" 1>&2
    echo " and that the file is executable." 1>&2
    print_usage
    exit 1
fi

# Test that a file named "stereopairs.lis" exists in the current directory
if [ ! -e "stereopairs.lis" ]; then
    echo "stereopairs.lis not found"
    print_usage
    exit 1
fi

# If the input file exists, check that ISIS has been initialized by looking for pds2isis,
# if not, initialize it
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
# Force the ISIS binary dir to the front of $PATH
# This works around possible name collision with `getkey` on some systems
# https://stackoverflow.com/questions/7128542/how-to-set-an-environment-variable-only-for-the-duration-of-the-script
export PATH=$ISISROOT/bin:$PATH

# Define a function that can be used by GNU Parallel
parallel_pedr_bin4pc_align (){
    # change into the directory for stereopair $3
    cd $3 || exit 1
    # We copy the file containing the list of PEDR binary files to the stereopair directory
    #  because `pedr2tab` refuses to read it if it's located anywhere other than the working directory
    pedr_list=$4
    cp $pedr_list ./
    pedr_list=$(basename $pedr_list)
    # Get the name of the first map-projected cube listed in stereopairs.lis
    cube=${1}.lev1eo.cub
    # # Get the bounding coordinates for the cube named above
    minlong=$(asap get-map-info ${cube} MinimumLongitude)
    maxlat=$(asap get-map-info ${cube} MaximumLatitude)
    maxlong=$(asap get-map-info ${cube} MaximumLongitude)
    minlat=$(asap get-map-info ${cube} MinimumLatitude)
    ##########################################################
    # Build a PEDR2TAB.PRM file for pedr2tab
    ##########################################################
    ## Template used to build the PEDR2TAB.PRM file:
    echo "T # lhdr" > PEDR2TAB.PRM
    echo "T # 0: shot longitude, latitude, topo, range, planetary_radius,ichan,aflag" >> PEDR2TAB.PRM
    echo "F # 1: MGS_longitude, MGS_latitude, MGS_radius" >> PEDR2TAB.PRM
    echo "F # 2: offnadir_angle, EphemerisTime, areodetic_lat,areoid" >> PEDR2TAB.PRM
    echo "T # 3: ishot, iseq, irev, gravity model number" >> PEDR2TAB.PRM
    echo "F # 4: local_time, solar_phase, solar_incidence" >> PEDR2TAB.PRM
    echo "F # 5: emission_angle, Range_Correction,Pulse_Width_at_threshold,Sigma_optical,E_laser,E_recd,Refl*Trans" >> PEDR2TAB.PRM
    echo "F # 6: bkgrd,thrsh,ipact,ipwct" >> PEDR2TAB.PRM
    echo "F # 7: range_window, range_delay" >> PEDR2TAB.PRM
    echo "F #   All shots, regardless of shot_classification_code" >> PEDR2TAB.PRM
    echo "T # F = noise or clouds, T = ground returns" >> PEDR2TAB.PRM
    echo "T # do crossover correction" >> PEDR2TAB.PRM
    echo "T \"${3}_pedr.asc\" # Name of file to write output to (must be enclosed in quotes)." >> PEDR2TAB.PRM
    echo "" >> PEDR2TAB.PRM
    echo "$minlong     # ground_longitude_min" >> PEDR2TAB.PRM
    echo "$maxlong     # ground_longitude_max" >> PEDR2TAB.PRM
    echo "$minlat      # ground_latitude_min" >> PEDR2TAB.PRM
    echo "$maxlat      # ground_latitude_max" >> PEDR2TAB.PRM
    echo "" >> PEDR2TAB.PRM
    echo "192.    # flattening used for areographic latitude" >> PEDR2TAB.PRM

    ##########################################################
    # run pedr2tab and send STDOUT to a log file (so we don't clutter up the terminal)
    pedr2tab $pedr_list > ${3}_pedr2tab.log
    ## Now for some housekeeping!
    # "pedr2tab" is a misnomer because the output is actually formatted as space-delimited, fixed-length columns
    #  so the first thing we want to do is transform ${3}_pedr.asc to an actual tab-delimited file that we can feed to `proj` while simultaneously eliminating columns we don't need
    #  The remaining columns will be:
    #   Longitude, Latitude, Datum_Elevation, Longitude, Latitude, Orbit
    #  We print the Longitude and Latitude columns twice to make things easier later after running `proj`
    # Delete 1 or more spaces at the beginning of each line and convert groups of 1 or more remaining spaces to a single comma | Rearrange the columns and calculate datum elevation | convert comma delimiter to tab delimiter and direct to file
    sed -e 's/^ \+//' -e 's/ \+/,/g' ${3}_pedr.asc | awk -F, 'NR > 2 {print($1","$2","($5 - 3396190)","$1","$2","$10)}' | sed 's/,/\t/g' > ${3}_pedr.tab
    inputTAB=${3}_pedr.tab

    # extract the proj4 string from one of the image cubes using asap and store it in a variable (we'll need it later for proj)
    projstr=$(asap get-srs-info $cube | sed 's/'\''//g')
    echo $projstr
    echo "#Latitude,Longitude,Datum_Elevation,Easting,Northing,Orbit" > ${3}_pedr4align.csv
    proj $projstr $inputTAB | sed 's/\t/,/g' | awk -F, '{print($5","$4","$3","$1","$2","$6)}' >> ${3}_pedr4align.csv
    # # Clean up extraneous files
    rm ${3}_pedr.asc ${3}_pedr.tab
    # ## TODO: Add functionality to build VRT for CSV and then convert to shapefile using ogr2ogr
}

#############################################################################
# export the function so GNU Parallel can use it
export -f parallel_pedr_bin4pc_align
# Call the function
echo "Start pedr_bin4pc_align $(date)"
awk -v pedrlist=$pedr_list '{print($0" "pedrlist)}' stereopairs.lis | parallel --joblog parallel_pedr_bin4pc_align.log --colsep ' ' parallel_pedr_bin4pc_align
echo "End   pedr_bin4pc_align $(date)"
set +x
