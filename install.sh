#!/bin/bash
set -e

# Fast Downward and Roswell, the planner's two external dependencies, install
# separately: see planner/install_fd.sh and
# planner/install_roswell.sh. Neither needs sudo.

conda env create -f environment.yml
# --no-deps, matching workbench/sh/sherlock_setup.sh. environment.yml has
# already resolved every dependency; without the flag pip re-resolves from
# setup.py and the two disagree, so a working environment is silently changed
# underneath itself: protobuf 3.20.3 becomes 3.19.6 and h5py <3.0 becomes
# ==2.10.0. environment.yml is the authority for what this project runs on.
conda run -n fosae pip install -e . --no-deps

# No `conda activate` here. In a non-interactive script it needs the shell hook
# and aborts under `set -e` without it. Activate yourself, once the script is
# done.

echo
echo "Done. Now activate the environment in your own shell:"
echo "    conda activate fosae"

