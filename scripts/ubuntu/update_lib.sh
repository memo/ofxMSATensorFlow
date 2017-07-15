#!/bin/bash

# this script copies the required dynamic lib (libtensorflow_cc.so) to desired folder

if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "*** You need to run this script as root (try with sudo) ***"
    echo 
	echo "This script":
	echo " 1. copies libtensorflow_cc.so"
	echo "      from src path (ofxMSATensorFlow/libs/tensorflow/lib/linux64 by default)"
	echo "      to destination folder (~/lib by defaut)"
	echo " 2. adds this path to /etc/ld.so.conf.d/libtensorflow_cc.so.conf"
	echo " 3. runs ldconfig"
	echo
	echo	
	echo "Usage (parameters optional):"
	echo
	echo "$ update_lib.sh [dst_folder] [src_folder]"
	echo
	exit
fi

# DEFAULTS
LIB_DST=$HOME'/lib' # override with arg 1
LIB_SRC='../../libs/tensorflow/lib/linux64/libtensorflow_cc.so' # override with arg2

DST_LD_CONF='/etc/ld.so.conf.d/libtensorflow_cc.so.conf'


if ! ([ -z "$1" ]); then LIB_DST=$1; fi
if ! ([ -z "$2" ]); then LIB_SRC=$2; fi


echo 'Copying' $LIB_SRC 'to' $LIB_DST
mkdir -p $LIB_DST
cp $LIB_SRC $LIB_DST

echo 'Writing path to' $DST_LD_CONF
echo $LIB_DST > $DST_LD_CONF

echo 'Running ldconfig...'
ldconfig
