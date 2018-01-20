#!/usr/bin/env bash
set -x
# Preprocessing script meant to be run as part of a CTX Ames Stereo Pipeline workflow.
# This script uses USGS ISIS3 routines to transform CTX EDRs into Level 1 products with the even/odd detector correction applied (hence "lev1eo"),
#   making them suitable for processing in ASP using the asp_ctx_map2dem.sh script

# Requires GNU parallel

# INPUT NOTES:
#  A textfile named productIDs.lis that contains the CTX product IDs to be processed.
#  The Product IDs of the stereopairs MUST be listed sequentially, i.e.
#   Pair1_Left
#   Pair1_Right
#   Pair2_Left
#   Pair2_Right
#     etc.


# Just a simple function to print a usage message
print_usage (){
    echo ""
    echo "Usage: $0 <productIDs.lis>"
    echo " Where <productIDs.lis> is a file containing a list of the IDs of the CTX products to be processed."
    echo " Product IDs belonging to a stereopair must be listed sequentially."
    echo " The script will search for CTX EDRs in the current directory before processing with ISIS."
    echo ""
}

### Check for sane commandline arguments

if [[ $# != 1 ]] || [[ "$1" = "-"* ]]; then
    # print usage message and exit
    print_usage
    exit 0
else
    p=$1
fi

echo "Start ctxedr2lev1eo.sh $(date)"

# Check that files corresponding to the ProductIDs in productIDs.lis exist and are of non-zero size
#   If a product is missing or empty, throw a warning and exit
prodarr=($(cat $p))
for i in "${prodarr[@]}"; do
    if ([[ ! -s $i.IMG ]]) || ([[ ! -s $i.IMG ]]); then
	     echo "Warning: \"$i\" EDR is Missing or Empty" 1>&2
	     exit 1
	fi
done

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

## ISIS Processing
#Ingest CTX EDRs into ISIS using mroctx2isis
awk '{print("ls "$1"*")}' $p | sh | awk '/.img$|.IMG$/' | parallel --joblog mroctx2isis.log mroctx2isis from={} to={.}.cub

#Add spice data using spiceinit
awk '{print("ls "$1".cub")}' $p | sh | parallel --joblog spiceinit.log spiceinit from={}

#Smooth SPICE using spicefit
awk '{print("ls "$1".cub")}' $p | sh | parallel --joblog spicefit.log spicefit from={}

#Apply photometric calibration using ctxcal
awk '{print("ls "$1".cub")}' $p | sh | parallel --joblog ctxcal.log ctxcal from={} to={.}.lev1.cub

#Apply CTX even/odd detector correction, ctxevenodd
awk '{print("ls "$1".lev1.cub")}' $p | sh | parallel --joblog ctxevenodd.log ctxevenodd from={} to={.}eo.cub

# Delete intermediate files
awk '{print("[ -f "$1".cub ] && rm "$1".cub")}' $p | sh
awk '{print("[ -f "$1".lev1.cub ] && rm "$1".lev1.cub")}' $p | sh

echo "End   ctxedr2lev1eo.sh $(date)"
set +x
