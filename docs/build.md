# CPU

On Windows, Ubuntu, and macOS
``` ssh 
$ cmake -B build . && cmake --build build
```
# Ubuntu
``` ssh
apt install libcurl4-gnutls-dev libssl-dev 
```

## Windows CMD
```
call "E:\software_vs\VC\Auxiliary\Build\vcvars64.bat"
```


# VULKAN
Install SDK on windows
+ https://vulkan.lunarg.com/sdk/home#windows

build
``` sh
enable GGML_VULKAN=ON
```
set Vulkan_SDK=C:\VulkanSDK\1.4.313.2


```
      - name: Dependencies
        id: depends
        run: |
          wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
          sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
          sudo apt-get update -y
          sudo apt-get install -y build-essential mesa-vulkan-drivers vulkan-sdk libcurl4-openssl-dev

      - name: Build
        id: cmake_build
        run: |
          cmake -B build \
            -DGGML_VULKAN=ON
          cmake --build build --config Release -j $(nproc)

```


Notes
- Nvidia GPU


# SYLC
- 

- install sycl  
call "E:\software_vs\VC\Auxiliary\Build\vcvars64.bat"
set VS2022INSTALLDIR=E:\software_vs\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat

- compile
```
cmake -G "Ninja" -B build -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_CPU=OFF -DGGML_SYCL=ON 
```


	- Support Intel buit-in GPU

## Reference
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html


# vcpkg
set CMAKE_TOOLCHAIN_FILE=C:/Users/harry/work/av_llm/vcpkg/scripts/buildsystems/vcpkg.cmake

# CUDA

# BLAS


# OpenBLAS


# BLIS


# Intel oneMLK



