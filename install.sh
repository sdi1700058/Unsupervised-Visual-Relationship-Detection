#!/bin/bash
set -e

sudo apt install -y gnuplot parallel

sudo apt install -y mercurial g++ cmake make flex bison g++-multilib
#git submodule update --init --recursive
#(
#    hg clone http://hg.fast-downward.org downward
#    cd downward
#    ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release
#)

# https://github.com/roswell/roswell/wiki/1.-Installation
#sudo apt -y install build-essential automake libcurl4-openssl-dev
#git clone -b release https://github.com/roswell/roswell.git
#(
#    cd roswell
#    sh bootstrap
#   ./configure
#   make
#    sudo make install
#    ros setup
#)

#ros dynamic-space-size=8000 install numcl arrival eazy-gnuplot
#ros dynamic-space-size=8000 install guicho271828/magicffi guicho271828/dataloader

#make -j 1 -C lisp
conda env create -f environment.yml
# --no-deps, matching sh/sherlock_setup.sh. environment.yml has already resolved
# every dependency; without the flag pip re-resolves from setup.py and the two
# disagree, so a working environment is silently changed underneath itself:
# protobuf 3.20.3 becomes 3.19.6 and h5py <3.0 becomes ==2.10.0. environment.yml
# is the authority for what this project runs on.
conda run -n fosae pip install -e . --no-deps
conda activate fosae

mkdir -p ~/.keras
cp keras-tf.json ~/.keras/keras.json

