#!/bin/bash

# ./copy_headers.sh ~/DEV/tensorflow

DST='../libs/tensorflow/include'

if [[ $# -eq 0 ]] ; then
	echo
	echo 'Missing argument. I need the path to your tensorflow repo'
	echo
	echo 'Usage: copy_headers.sh path/to/tensorflow'
	echo 'e.g.:  copy_headers.sh ~/DEV/tensorflow'
	echo
	echo 'Also note, it will copy the headers to '$DST
	echo 'so make sure you are running this script from ofxMSATensorFlow/scripts'	
	echo
	echo
	exit 1
fi

SRC=$1  # command line argument is the location of tensorflow repo


echo 'Copying files from '$SRC' to '$DST

# remove existing headers for a clean start
rm -rf $DST

mkdir -p $DST/tensorflow
cp -R $SRC/tensorflow/core/ $DST/tensorflow
cp -R $SRC/tensorflow/cc/ $DST/tensorflow

mkdir -p $DST/third_party
cp -R $SRC/third_party/eigen3/ $DST/third_party

cp -R $SRC/bazel-genfiles/tensorflow/cc/ $DST/tensorflow
cp -R $SRC/bazel-genfiles/tensorflow/core/ $DST/tensorflow

mkdir -p $DST/external/eigen_archive
cp -R $SRC/bazel-tensorflow/external/eigen_archive/eigen-eigen*/ $DST/external/eigen_archive



#find $DST -name '*.cpp' -type f -delete
#find $DST -name '*.c' -type f -delete
#find $DST -name '*.cc' -type f -delete
#find $DST -name '*.cxx' -type f -delete
#find $DST -name '*.cmake' -type f -delete
#find $DST -name '*.py' -type f -delete
#find $DST -name '*.txt' -type f -delete
#find $DST -name '*.dat' -type f -delete
#find $DST -name '*.' -type f -delete
#find $DST -name '*.sh' -type f -delete
