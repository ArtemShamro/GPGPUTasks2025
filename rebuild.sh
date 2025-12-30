rm -rf build
mkdir build && cd build
cmake -DGPU_CUDA_SUPPORT=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j8  