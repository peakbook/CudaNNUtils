ARCH=compute_35
CODE=sm_35

kernels.ptx: kernels.cu quaternions.cu equiv.cu basics.cu actfuncs.cu
	nvcc -ptx -arch=${ARCH} -code=${CODE} kernels.cu

clean:
	-rm kernels.ptx
