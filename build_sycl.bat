cmake -B build_sycl -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx ^
-DCMAKE_BUILD_TYPE=Release ^
-DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON ^
-DGGML_SYCL=ON ^
-DGGML_CPU=ON ^
-DLLAMA_CURL=OFF -A x64
cmake --build build_sycl --config=Release
