#!/bin/bash
BUILD_TYPE=$1
CMAKE_ARGS=

# Setup virtual environment
echo "Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Figure out the build type
echo "Detecting build type..."
unamestr=$(uname)
if [[ $BUILD_TYPE == "" ]]; then
  if [[ "$unamestr" == "Darwin" ]]; then
    BUILD_TYPE="metal"
  else
    amd=$(lspci | grep -i amd)
    nvidia=$(lspci | grep -i nvidia)

    if [[ "$nvidia" != "" ]]; then
      BUILD_TYPE="cublas"
    elif [[ "$amd" != "" ]]; then
      BUILD_TYPE="hipblas"
    else
      BUILD_TYPE="openblas"
    fi
  fi
fi

# Set up build type and CMake arguments
echo "Setting up build type..."
if [ "$BUILD_TYPE" = "cublas" ]; then
  CMAKE_ARGS+="-DGGML_CUDA=ON"
elif [ "$BUILD_TYPE" = "hipblas" ]; then
  CMAKE_ARGS+="-DGGML_HIPBLAS=on"
elif [ "$BUILD_TYPE" = "metal" ]; then
  CMAKE_ARGS+="-DGGML_METAL=ON"
else
  CMAKE_ARGS+="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt
