#!/usr/bin/env bash
set -x
# Summary:
# Script to take downloaded HiRISE stereopairs, run them through NASA Ames Stereo Pipeline's hiedr2mosaic.py and cam2map4stereo.py scripts
#  in anticipation of passing the output to ASP's `stereo` or `parallel_stereo` commands using other wrappers that are part of the UChicago ASP Scripts workflow.
# This script is capable of processing many stereopairs in a single run and uses GNU parallel
#  to improve the efficiency of the processing and reduce total wall time.

# Dependencies:
#      NASA Ames Stereo Pipeline
#      USGS ISIS3
#      GDAL
#      GNU parallel

# INPUT NOTES:
#  A textfile named productIDs.lis that contains the HiRISE product IDs to be processed.
#  The Product IDs of the stereopairs MUST be listed sequentially, i.e.
#   Pair1_Left
#   Pair1_Right
#   Pair2_Left
#   Pair2_Right
#     etc.


# Just a simple function to print a usage message
print_usage (){
    echo ""
    echo "Usage: asp_hirise_prep.sh -p <productIDs.lis>"
    echo " Where <productIDs.lis> is a file containing a list of the IDs of the HiRISE products to be processed."
    echo " Product IDs belonging to a stereopair must be listed sequentially."
    echo " The script will search for HiRISE EDRs in subdirectories named for the items in <productIDs.lis> before processing with ASP."
    echo " "
}

### Check for sane commandline arguments
if [[ $# = 0 ]] || [[ "$1" != "-"* ]]; then
    # print usage message and exit
    echo "ERROR: Invalid number of arguments or first arg is not a flag"
    print_usage
    exit 1
elif  [[ "$1" = "-"* ]]; then # Else use getopts to parse flags that may have been set
    while getopts ":p:" opt; do
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
####

echo "Start asp_hirise_prep $(date)"

#######################################################
## Housekeeping and Creating Some Support Files for ASP
#######################################################

# Create a 3-column, space-delimited file containing list of HiRISE stereo product IDs and the name of the corresponding directory that will be created for each pair
awk '{printf "%s ", $0}!(NR % 2){printf "\n"}' $prods | awk '{print($1" "$2" "$1"_"$2)}' > stereopairs.lis
# Extract Column 3 (the soon-to-be- directory names) from stereopairs.lis and write it to a file called stereodirs.lis
awk '{print($3)}' stereopairs.lis > stereodirs.lis
# Make directories named according to the lines in stereodirs.lis
awk '{print("mkdir -p "$1)}' stereodirs.lis | sh
# Now extract each line from stereopairs.lis (created above) and write it to a textfile inside the corresponding subdirectory we created on the previous line
# These files are used to ensure that the input images are specified in the same order during every step of `stereo` in ASP
awk '{print $1" "$2 >$3"/stereopair.lis"}' stereopairs.lis

#####################################################

###
## Use GNU parallel to invoke many instances of hiedr2mosaic.py
# define a function that will be used by GNU parallel to mosaic the CCDs of many image products more computationally efficient
function parallel_hiedr2mosaic() {
    cd $1 || { echo "could not cd into $1" && exit 1; }
    hiedr2mosaic.py $1*.IMG
    cd ../ || { echo "could not cd ../" && exit 1; }
}
# Must export the function for it to work with GNU parallel
export -f parallel_hiedr2mosaic
# Run hiedr2mosaic.py by using GNU parallel to call parallel_hiedr2mosaic
echo "Running hiedr2mosaic.py using GNU Parallel"
parallel --joblog parallel_hiedr2mosaic.log parallel_hiedr2mosaic :::: $prods
if [ $? -ne 0 ]
then
    echo "Failure running parallel_hiedr2mosaic at $(date)"
    exit 1
else
    echo "Success running parallel_hiedr2mosaic at $(date)"
fi

echo "Finished hiedr2mosaic.py"
date

echo "Moving HiJitreged cubes into stereopair directories..."
# Move the NOPROJ cubes into the working directory named for the stereopair each belongs to
awk '{print("mv  "$1"/"$1"*.mos_hijitreged.norm.cub ./"$3"/")}' stereopairs.lis | sh
awk '{print("mv  "$2"/"$2"*.mos_hijitreged.norm.cub ./"$3"/")}' stereopairs.lis | sh

## Use GNU parallel to run many instances of cam2map4stereo.py at once
#   and project the images of each stereopair into a common projection
# Define a function that GNU parallel will call to run cam2map4stereo.py
function parallel_cam2map4stereo() {
    cd $3 || exit 1
    cam2map4stereo.py ${1}_RED.mos_hijitreged.norm.cub ${2}_RED.mos_hijitreged.norm.cub
}
export -f parallel_cam2map4stereo

echo "Running cam2map4stereo.py using GNU Parallel"
parallel --joblog parallel_cam2map4stereo.log --colsep ' ' parallel_cam2map4stereo :::: stereopairs.lis
if [ $? -ne 0 ]
then
    echo "Failure running parallel_cam2map4stereo at $(date)"
    exit 1
else
    echo "Success running parallel_cam2map4stereo at $(date)"
fi


echo "End   asp_hirise_prep $(date)"
set +x
