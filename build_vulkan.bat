REM # edit to point to the VS++ x64 developer  
call "E:\software_vs\VC\Auxiliary\Build\vcvars64.bat"

REM Ninja Generator base
REM cmake --preset x64-windows-vulkan-release
REM cmake --build build-x64-windows-vulkan-release

REM # MS Generator base
cmake -B build-x64-windows-vulkan-release -A x64
cmake --build build-x64-windows-vulkan-release --config=Release

