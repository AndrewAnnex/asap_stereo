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
    echo "Usage: $0 -s <stereo.default> -p <productIDs.lis>"
    echo " Where <productIDs.lis> is a file containing a list of the IDs of the HiRISE products to be processed."
    echo " Product IDs belonging to a stereopair must be listed sequentially."
    echo " The script will search for HiRISE EDRs in subdirectories named for the items in <productIDs.lis> before processing with ASP."
    echo " "
    echo "<stereo.default> is the name and absolute path to the stereo.default file to be used by the stereo command."
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
echo "Start asp_hirise_map2dem $(date)"
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

# If this script is run as part of a job on Midway, we write the nodelist to a file named "nodelist.lis" so parallel_stereo can use it
if hash scontrol 2>/dev/null; then
    # This line is not portable to environments that are NOT running SLURM
    scontrol show hostname $SLURM_NODELIST | tr ' ' '\n' > nodelist.lis
fi
#####################################################
##  Run ALL stereo in series for each stereopair using `parallel_stereo`
# This is not the most resource efficient way of doing this but it's a hell of a lot more efficient compared to using plain `stereo` in series
for i in $( cat stereodirs.lis ); do
    echo "Begin parallel_stereo on $i at $(date)"
    cd $i || { echo "could not cd into $i"; exit 1; }
    # Store the names of the HiRISE cubes in variables
    Lmos=$(awk '{print($1".mos_hijitreged.norm.cub")}' stereopair.lis)
    Rmos=$(awk '{print($2".mos_hijitreged.norm.cub")}' stereopair.lis)
    Lmap=$(awk '{print($1"_RED.map.cub")}' stereopair.lis)
    Rmap=$(awk '{print($2"_RED.map.cub")}' stereopair.lis)
    Lmosmap=$(awk '{print($1".mos_hijitreged.norm.map.cub")}' stereopair.lis)
    Rmosmap=$(awk '{print($2".mos_hijitreged.norm.map.cub")}' stereopair.lis)
    Ladj=$(awk '{print($1"_RED.map.adjust")}' stereopair.lis)
    Radj=$(awk '{print($2"_RED.map.adjust")}' stereopair.lis)

    if [[ $? -ne 0 ]] || [[ ! -e "adjust/ba-${Ladj}" ]] || [[ ! -e "adjust/ba-${Radj}" ]]; then
        echo "No bundle adjust output found! at $(date)"
        exit 1
    fi

    # Note that we specify ../nodelist.lis as the file containing the list of hostnames for `parallel_stereo` to use
    # You may wish to edit out the --nodes-list argument if running this script in a non-SLURM environment
    # See the ASP manual for information on running `parallel_stereo` with a node list argument that is suitable for your environment

    echo "Begin parallel_stereo on \"$i\" at $(date)"
    if [ -r ../nodelist.lis ]; then
        # for SLURM environment
        # stop parallel_stereo after correlation
        parallel_stereo --nodes-list=../nodelist.lis --stop-point 2 $Lmap $Rmap -s ${config} results/$i --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 1 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 1 of $i at $(date)"
        fi
        # attempt to optimize parallel_stereo for running on the sandyb nodes (16 cores each) for Steps 2 (refinement) and 3 (filtering)
        parallel_stereo --nodes-list=../nodelist.lis --processes 2 --threads-multiprocess 8 --threads-singleprocess 16 --entry-point 2 --stop-point 4 $Lmap $Rmap -s ${config} results/$i --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 2 & 3 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 2 & 3 of $i at $(date)"
        fi
        # finish parallel_stereo using default options for Stage 4 (Triangulation)
        parallel_stereo --nodes-list=../nodelist.lis --entry-point 4 $Lmap $Rmap -s ${config} results/$i --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 4 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 4 of $i at $(date)"
        fi
    else
        # for single node environment
        parallel_stereo --processes 2 --threads-multiprocess 8 --threads-singleprocess 16 --stop-point 4 $Lmap $Rmap -s ${config} results/$i --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Steps 1-3 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Steps 1-3 of $i at $(date)"
        fi
        parallel_stereo --processes 8  --threads-multiprocess 8 --threads-singleprocess 16 --entry-point 4 $Lmap $Rmap -s ${config} results/$i --bundle-adjust-prefix adjust/ba
        if [ $? -ne 0 ]
        then
            echo "Failure running parallel_stereo Step 4 of $i at $(date)"
            exit 1
        else
            echo "Success running parallel_stereo Step 4 of $i at $(date)"
        fi
    fi

    cd ../ ||  { echo "could not cd ../"; exit 1; }
    echo "Finished parallel_stereo on $i at $(date)"
done

## Transform point clouds to DEMs using `point2dem`. DEMs will have same map projection as parent images.
for i in $( cat stereodirs.lis ); do
    cd $i/results ||  { echo "could not cd into $i/results"; exit 1; }
    # extract the proj4 string from one of the map-projected image cubes and store it in a variable (we'll need it later for point2dem)
    proj=$(awk 'NR==1 {print("gdalsrsinfo -o proj4 ../"$1"_RED.map.cub")}' ../stereopair.lis | sh | sed 's/'\''//g')
    echo "Begin point2dem on $i at $(date)"
    point2dem --t_srs "${proj}" -r mars --nodata -32767 -s 2 -n --errorimage $i-PC.tif --orthoimage $i-L.tif -o dem/$i
    if [ $? -ne 0 ]
    then
        echo "Failure running point2dem for $i at $(date)"
        exit 1
    else
        echo "Success running point2dem for $i at $(date)"
    fi
    mkdir -p logs
    mv *-log-* ./logs
    cd ../../ || exit 1
    echo "Finished point2dem on $i at $(date)"
done
echo "End   asp_hirise_map2dem $(date)"
set +x
