call "E:\software_vs\VC\Auxiliary\Build\vcvars64.bat"
REM set VS2022INSTALLDIR=E:\software_vs
set VS2022INSTALLDIR=E:\software_vs\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat
cmake -B build -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx ^
-DCMAKE_BUILD_TYPE=Release ^
-DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON ^
-DGGML_SYCL=ON ^
-DGGML_CPU=ON ^
-DLLAMA_CURL=OFF -A x64
cmake --build build --config=Release
