cp "${{ env.ONEAPI_ROOT }}/mkl/latest/bin/mkl_sycl_blas.5.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/mkl/latest/bin/mkl_core.2.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/mkl/latest/bin/mkl_tbb_thread.2.dll" ./build/bin

cp "${{ env.ONEAPI_ROOT }}/compiler/latest/bin/ur_adapter_level_zero.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/compiler/latest/bin/ur_adapter_opencl.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/compiler/latest/bin/ur_loader.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/compiler/latest/bin/ur_win_proxy_loader.dll" ./build/bin

cp "${{ env.ONEAPI_ROOT }}/compiler/latest/bin/sycl8.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/compiler/latest/bin/svml_dispmd.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/compiler/latest/bin/libmmd.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/compiler/latest/bin/libiomp5md.dll" ./build/bin

cp "${{ env.ONEAPI_ROOT }}/dnnl/latest/bin/dnnl.dll" ./build/bin
cp "${{ env.ONEAPI_ROOT }}/tbb/latest/bin/tbb12.dll" ./build/bin

echo "cp oneAPI running time dll files to ./build/bin done"
