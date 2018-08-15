#!/usr/bin/env bash
set -x

# Summary:
# Script to take downloaded HiRISE stereopairs, run them through NASA Ames stereo Pipeline and produce preliminary, uncontrolled DEMs.
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
    echo "Usage: $0 -p <productIDs.lis>"
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
echo "Start asp_hirise_ba $(date)"
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
awk '{print $1" "$2 > $3"/stereopair.lis"}' stereopairs.lis

# If this script is run as part of a job on Midway, we write the nodelist to a file named "nodelist.lis"
if hash scontrol 2>/dev/null; then
    # This line is not portable to environments that are NOT running SLURM
    scontrol show hostname $SLURM_NODELIST | tr ' ' '\n' > nodelist.lis
fi
#####################################################
##  Run ALL Bundle adjustments
# TODO: since bundle adjust for isis3 images is single threaded, we could do parallel if there is enough mem?
for i in $( cat stereodirs.lis ); do
    echo "Begin parallel_stereo on $i at $(date)"
    cd $i || { echo "could not cd into $i"; exit 1; }
    # Store the names of the HiRISE cubes in variables
    Lmos=$(awk '{print($1".mos_hijitreged.norm.cub"' stereopair.lis)
    Rmos=$(awk '{print($2".mos_hijitreged.norm.cub"' stereopair.lis)
    Lmap=$(awk '{print($1"_RED.map.cub")}' stereopair.lis)
    Rmap=$(awk '{print($2"_RED.map.cub")}' stereopair.lis)
    Lmosmap=$(awk '{print($1".mos_hijitreged.norm.map.cub"' stereopair.lis)
    Rmosmap=$(awk '{print($2".mos_hijitreged.norm.map.cub"' stereopair.lis)
    Ladj=$(awk '{print($1"_RED.map.adjust")}' stereopair.lis)
    Radj=$(awk '{print($2"_RED.map.adjust")}' stereopair.lis)

    # Run ASP's bundle_adjust on the given stereopair
    echo "Begin bundle_adjust on \"$i\" at $(date)"
    bundle_adjust $Lmap $Rmap -o adjust/ba --threads 16
    if [[ $? -ne 0 ]] || [[ ! -e "adjust/ba-${Ladj}" ]] || [[ ! -e "adjust/ba-${Radj}" ]]; then
        echo "Failure running bundle_adjust of $i at $(date)"
        exit 1
    else
        echo "Success running bundle_adjust of $i at $(date)"
    fi
    echo "Finished bundle_adjust on \"$i\" at $(date)"
    cd ../ ||  { echo "could not cd ../"; exit 1; }
done
echo "End   asp_hirise_ba $(date)"
set +x
