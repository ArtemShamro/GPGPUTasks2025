git clone https://github.com/google/shaderc.git
cd shaderc
git submodule update --init
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
