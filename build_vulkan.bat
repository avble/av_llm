set Vulkan_SDK=C:\VulkanSDK\1.4.313.2
cmake --preset x64-windows-vulkan-release -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build-x64-windows-vulkan-release
