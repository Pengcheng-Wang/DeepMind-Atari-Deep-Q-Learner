#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################


TOPDIR=$PWD

# Prefix:
PREFIX=$PWD/torch
echo "Installing Torch into: $PREFIX"

# if [[ `uname` != 'Linux' ]]; then
#   echo 'Platform unsupported, only available for Linux'
#   exit
# fi
# if [[ `which brew` == '' ]]; then
#     echo 'brew not found, platform not supported'
#     exit
# fi

# # Install dependencies for Torch:
# brew update
# brew install -qqy build-essential
# brew install -qqy gcc g++
# brew install -qqy cmake
# install -qqy curl
# brew install -qqy libreadline-dev
# brew install -qqy git-core
# brew install -qqy libjpeg-dev
# brew install -qqy libpng-dev
# brew install -qqy ncurses-dev
# brew install -qqy imagemagick
# brew install -qqy unzip
brew install -qqy libqt4-dev
brew install -qqy liblua5.1-0-dev
brew install -qqy libgd-dev
# brew update

exit
echo "==> Torch7's dependencies have been installed"





# # Build and install Torch7
# cd /tmp
# rm -rf luajit-rocks
# git clone https://github.com/torch/luajit-rocks.git
# cd luajit-rocks
# mkdir -p build
# cd build
# git checkout master; git pull
# rm -f CMakeCache.txt
# cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
# RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
# make
# RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
# make install
# RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi


path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then
    cutorch=ok
    cunn=ok
fi

# Install base packages:
# luarocks install cwrap
# luarocks install paths
# luarocks install torch
# luarocks install nn

# [ -n "$cutorch" ] && \
# ($PREFIX/bin/luarocks install cutorch)
# [ -n "$cunn" ] && \
# ($PREFIX/bin/luarocks install cunn)

# luarocks install luafilesystem
# luarocks install penlight
# luarocks install sys
# luarocks install xlua
# luarocks install image
# luarocks install env
# luarocks install qtlua
# luarocks install qttorch

echo ""
echo "=> Torch7 has been installed successfully"
echo ""


# echo "Installing nngraph ... "
# $PREFIX/bin/luarocks install nngraph
# RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
# echo "nngraph installation completed"

# echo "Installing Xitari ... "
# cd /tmp
# rm -rf xitari
# git clone https://github.com/deepmind/xitari.git
# cd xitari
# $PREFIX/bin/luarocks make
# RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
# echo "Xitari installation completed"
#
# echo "Installing Alewrap ... "
# cd /tmp
# rm -rf alewrap
# git clone https://github.com/deepmind/alewrap.git
# cd alewrap
# $PREFIX/bin/luarocks make
# RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
# echo "Alewrap installation completed"

echo "Installing Lua-GD ... "
mkdir $PREFIX/src
cd $PREFIX/src
rm -rf lua-gd
git clone https://github.com/ittner/lua-gd.git
cd lua-gd
sed -i "s/LUABIN=lua5.1/LUABIN=..\/..\/bin\/luajit/" Makefile
$PREFIX/bin/luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Lua-GD installation completed"

echo
echo "You can run experiments by executing: "
echo
echo "   ./run_cpu game_name"
echo
echo "            or   "
echo
echo "   ./run_gpu game_name"
echo
echo "For this you need to provide the rom files of the respective games (game_name.bin) in the roms/ directory"
echo
