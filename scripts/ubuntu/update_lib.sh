#!/bin/bash

# this script:
# 1. copies the libtensorflow_cc.so
#       from ofxMSATensorFlow/libs/tensorflow/lib/linux64/ to ~/lib (or user specified folder)
# 2. adds this path to /etc/ld.so.conf.d/lib/libtensorflow_cc.so.conf 
# 3. runs sudo ldconfig

if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "*** You need to run this script as root (try with sudo) ***"
    echo 
	echo "This script":
	echo " 1. copies libtensorflow_cc.so"
	echo "      from ofxMSATensorFlow/libs/tensorflow/lib/linux64"
	echo "      to a destination lib path (~/lib by defaut, or command line arg)"
	echo " 2. adds this path to /etc/ld.so.conf.d/libtensorflow_cc.so.conf"
	echo " 3. runs ldconfig"
	echo
    exit
fi


if [[ $# -eq 0 ]]
then
	LIB_DST=$HOME'/lib'
else
	LIB_DST=$1
fi

LIB_SRC='../../libs/tensorflow/lib/linux64/libtensorflow_cc.so'
DST_LD_CONF='/etc/ld.so.conf.d/libtensorflow_cc.so.conf'

echo 'Copying' $LIB_SRC 'to' $LIB_DST
mkdir $LIB_DST
cp $LIB_SRC $LIB_DST

echo 'Writing path to' $DST_LD_CONF
echo $LIB_DST > $DST_LD_CONF

echo 'Running ldconfig...'
ldconfig
