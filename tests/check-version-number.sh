#!/usr/bin/env bash

# This is a silly little bash script that checks the version number of
# the git repository against the CMake list.
#
# Most of the code is error handling setup, defined in the first
# section the rest is parsing tag numbers.

if [[ $- == *i* ]]; then
    echo "don't source me!" >&2
    return 1
fi

##################### Error Handling Setup ##########################

# Set the script to exit if anything returns an error
set -eu

# The ERROR variable indicates what went wrong
ERROR="something went wrong parsing version numbers"

# The HELP string suggests a way to fix it
HELP="Git tags should be of the form vX.Y[.Z] where X, Y, and Z are integers"

# This function is called if anything goes wrong
function print_bad_exit() {
    echo "ERROR: ${ERROR}. ${HELP}" >&2
}
trap print_bad_exit EXIT

################### Git tag parsing  #############################

# get the git tag
GIT_TAG=$(git describe --abbrev=0)

# parse out the number
ERROR="git tag ${GIT_TAG} contains no number"
GIT_TAG_NUMBER=$(egrep -o '[0-9.]+' <<< $GIT_TAG)

# check that there's only one number
ERROR="git tag ${GIT_TAG} contains too many numbers"
if (( $(wc -w <<< $GIT_TAG_NUMBER) != 1 )); then
    exit 1
fi

################### CMake tag parsing ################################

# No longer need to give help reguarding a bad git tag, change it to
# talk about cmake
HELP="The CMakeLists.txt file should contain 'project( lwtnn VERSION X.X)"

# get the cmake project
ERROR="Can't find CMake project name"
CMAKE_PROJECT=$(egrep "project *\(.*\)" CMakeLists.txt)

# Get the version number
ERROR="Can't parse CMake project name from '${CMAKE_PROJECT}'"
CMAKE_TAG=$(sed -n 's/.*VERSION * \([0-9\.]*\).*/\1/p' <<< $CMAKE_PROJECT)
if (( $(wc -w <<< $CMAKE_TAG) != 1 )); then
    exit 1
fi

##################### Tag comparisons ########################

HELP=""
# compare the tags
if [[ $GIT_TAG_NUMBER != $CMAKE_TAG ]]; then
    ERROR="git tag ${GIT_TAG_NUMBER} != cmake tag ${CMAKE_TAG}"
    exit 1
fi

trap - EXIT

echo "git tag matches cmake tag: ${CMAKE_TAG}"
