call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cmake -G "Ninja" -B build -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx \ 
	-DCMAKE_BUILD_TYPE=Release \
	-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
	-DGGML_CPU=OFF -DGGML_SYCL=ON \
	-DLLAMA_CURL=OFF
cmake --build build --target ggml-sycl -j
