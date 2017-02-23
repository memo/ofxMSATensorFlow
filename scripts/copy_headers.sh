#!/bin/bash

DST='../libs/tensorflow/include'

if [[ $# -eq 0 ]] ; then
	echo
	echo 'This script copies files from various tensorflow source locations to '$DST
	echo '(Check the source of the script to see exactly what it copies)'
	echo
	echo '** Missing argument. I need the path to your tensorflow repo. **'
	echo
	echo 'Usage: copy_headers.sh path/to/tensorflow [cleanup: true or false]'
	echo 'e.g.'
	echo '       copy_headers.sh ~/lib/tensorflow'
	echo '       copy_headers.sh ~/lib/tensorflow true'
	echo
	echo 'cleanup option:'
	echo 'This script will copy ALL files from various folders in path/to/tensorflow to '$DST
	echo 'Set cleanup to ''true'' to delete unnecessary file types (e.g. cpp, c, cc, cxx, cmake, py etc) from the destination after copying'
	echo 
	echo 'Also note, the script copies the files to the relative path '$DST
	echo 'So make sure you are running this script from ofxMSATensorFlow/scripts'	
	echo

	exit 1
fi

SRC=$1  # command line argument is the location of tensorflow repo


if [[ $# -eq 2 ]] ; then
	DO_CLEANUP=$2
else
	DO_CLEANUP=false
fi

echo 'Copying files from '$SRC' to '$DST

# remove existing headers for a clean start
rm -rf $DST

mkdir -p $DST/tensorflow
cp -RL $SRC/tensorflow/core $DST/tensorflow
cp -RL $SRC/tensorflow/cc $DST/tensorflow

mkdir -p $DST/third_party
cp -RL $SRC/third_party/eigen3 $DST/third_party

#rm -rf $DST/third_party/eigen3/unsupported
cp -RLf $SRC/bazel-tensorflow/external/eigen_archive/unsupported $DST

cp -RL $SRC/bazel-genfiles/tensorflow/cc $DST/tensorflow
cp -RL $SRC/bazel-genfiles/tensorflow/core $DST/tensorflow

cp -RL $SRC/bazel-tensorflow/external/eigen_archive/Eigen $DST/Eigen


if $DO_CLEANUP ; then
	echo "Cleaning up. Deleting src files from "$DST 
	find $DST -name '*.cpp' -type f -delete
	find $DST -name '*.c' -type f -delete
	find $DST -name '*.cc' -type f -delete
	find $DST -name '*.cxx' -type f -delete
	find $DST -name '*.cmake' -type f -delete
	find $DST -name '*.py' -type f -delete
	find $DST -name '*.txt' -type f -delete
	find $DST -name '*.dat' -type f -delete
	find $DST -name '*.sh' -type f -delete
	find $DST -name '*.proto' -type f -delete
fi