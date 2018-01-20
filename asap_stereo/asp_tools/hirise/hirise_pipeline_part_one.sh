#!/usr/bin/env bash

function step1() {
    asp_hirise_prep.sh -p $1
    if [ $? -eq 0 ]
    then
        echo "Success asp_hirise_prep on $(date). (1/2)"
    else
        echo "Failure asp_hirise_prep on $(date). (1/2)" >&2
        exit 1
    fi
}

function step2() {
    asp_hirise_map2dem.sh -s $1 -p $2
    if [ $? -eq 0 ]
    then
        echo "Success asp_hirise_map2dem on $(date). (2/2)"
    else
        echo "Failure asp_hirise_map2dem on $(date). (2/2)" >&2
        exit 1
    fi

}

print_usage (){
    echo ""
    echo "Usage: $0 <stereo.default> <productIDs.lis> "
    echo " Where <stereo.default> is the name and absolute path to the stereo.default file to be used by the stereo command."
    echo " Where <productIDs.lis> is a file containing a list of the IDs of the HiRISE products to be processed."
    echo ""
}

##########################################
# begin parsing arguments

# 1 is stereo.default file
# 2 is productIDs file

### Check for the correct number of args and hope for the best ###
if [[ $# != 2 ]] ; then
    # print usage message and exit
    print_usage
    exit 1
fi
##########################################
# begin calling commands
echo "Start hirise_pipeline_part_one $(date)"

# introduce process locking (hint files) to not do repeats

step1 $2

step2 $1 $2

# done
##########################################
echo "End   hirise_pipeline_part_one $(date)"
