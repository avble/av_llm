# CPU

On Windows, Ubuntu, and macOS
``` ssh 
$ cmake -B build . && cmake --build build
```

## Windows CMD
```
call "E:\software_vs\VC\Auxiliary\Build\vcvars64.bat"
```


# VULKAN
Install SDK
+ https://vulkan.lunarg.com/sdk/home#windows

build
``` sh
enable GGML_VULKAN=ON
```
set Vulkan_SDK=C:\VulkanSDK\1.4.313.2

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



