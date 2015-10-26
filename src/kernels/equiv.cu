__device__ void equiv(cuDoubleComplex *dst,const Quaternion *src, const int w, const int h);
__device__ void equiv(Quaternion *dst,const cuDoubleComplex *src, const int w, const int h);
__device__ void equiv(cuFloatComplex *dst,const Quaternionf *src, const int w, const int h);
__device__ void equiv(Quaternionf *dst,const cuFloatComplex *src, const int w, const int h);
__device__ void transpose_pre(cuFloatComplex *dst, const int w, const int h);
__device__ void transpose_pre(cuDoubleComplex *dst, const int w, const int h);
template<typename T> __device__ void transpose(T *dst, const T *src, const int w, const int h);
template<typename T> __device__ void ctranspose(T *dst, const T *src, const int w, const int h);

__device__ void equiv(cuDoubleComplex *dst, const Quaternion *src, const int w, const int h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idxc1 = j*2*w+i;
    int idxc2 = j*2*w+i+w;
    int idxc3 = (j+h)*2*w+i;
    int idxc4 = (j+h)*2*w+i+w;
    int idxq = j*w+i;

    if(i < w && j< h)
    {
        Quaternion val = src[idxq];
        dst[idxc1] = make_double2(val.x, val.y);
        dst[idxc2] = make_double2(-val.z, val.w);
        dst[idxc3] = make_double2(val.z, val.w);
        dst[idxc4] = make_double2(val.x, -val.y);
    }
}

__device__ void equiv(cuFloatComplex *dst, const Quaternionf *src, const int w, const int h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idxc1 = j*2*w+i;
    int idxc2 = j*2*w+i+w;
    int idxc3 = (j+h)*2*w+i;
    int idxc4 = (j+h)*2*w+i+w;
    int idxq = j*w+i;

    if(i < w && j< h)
    {
        Quaternionf val = src[idxq];
        dst[idxc1] = make_float2(val.x, val.y);
        dst[idxc2] = make_float2(-val.z, val.w);
        dst[idxc3] = make_float2(val.z, val.w);
        dst[idxc4] = make_float2(val.x, -val.y);
    }
}

__device__ void equiv(Quaternion *dst, const cuDoubleComplex *src, const int w, const int h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idxc1 = j*2*w+i;
    int idxc3 = (j+h)*2*w+i;
    int idxq = j*w+i;

    if(i < w && j< h)
    {
        cuDoubleComplex a = src[idxc1];
        cuDoubleComplex b = src[idxc3];
        dst[idxq] = quaternion(cuCreal(a), cuCimag(a), cuCreal(b), cuCimag(b));
    }
}

__device__ void equiv(Quaternionf *dst, const cuFloatComplex *src, const int w, const int h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idxc1 = j*2*w+i;
    int idxc3 = (j+h)*2*w+i;
    int idxq = j*w+i;

    if(i < w && j< h)
    {
        cuFloatComplex a = src[idxc1];
        cuFloatComplex b = src[idxc3];
        dst[idxq] = quaternion(cuCrealf(a), cuCimagf(a), cuCrealf(b), cuCimagf(b));
    }
}

__device__ void transpose_pre(cuDoubleComplex *dst, const int w, const int h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int hw = w>>1;
    int hh = h>>1;
    int idx = j*w+i;

    if(i < w && j< h)
    {
        if (( i >= hw && j < hh) || ( i < hw && j >= hh)){
            cuDoubleComplex val = dst[idx];
            dst[idx] = make_double2(-val.x, val.y);
        }
    }
}

__device__ void transpose_pre(cuFloatComplex *dst, const int w, const int h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int hw = w>>1;
    int hh = h>>1;
    int idx = j*w+i;

    if(i < w && j< h)
    {
        if (( i >= hw && j < hh) || ( i < hw && j >= hh)){
            cuFloatComplex val = dst[idx];
            dst[idx] = make_float2(-val.x, val.y);
        }
    }
}

template<typename T>
__device__ void transpose(T *dst, const T *src, const int w, const int h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_dst = j*w+i;
    int idx_src = i*h+j;

    if(i < w && j< h)
    {
        dst[idx_dst] = src[idx_src];
    }
}

template<typename T>
__device__ void ctranspose(T *dst, const T *src, const int w, const int h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_dst = j*w+i;
    int idx_src = i*h+j;

    if(i < w && j< h)
    {
        dst[idx_dst] = qconj(src[idx_src]);
    }
}

extern "C"{
    void __global__ q2c_float(cuFloatComplex *dst, const Quaternionf *src, const int w, const int h) { equiv(dst,src,w,h);}
    void __global__ q2c_double(cuDoubleComplex *dst, const Quaternion *src, const int w, const int h) { equiv(dst,src,w,h);}
    void __global__ c2q_float(Quaternionf *dst, const cuFloatComplex *src, const int w, const int h) { equiv(dst,src,w,h);}
    void __global__ c2q_double(Quaternion *dst, const cuDoubleComplex *src, const int w, const int h) { equiv(dst,src,w,h);}
    void __global__ transpose_pre_float(cuFloatComplex *dst, const int w, const int h) { transpose_pre(dst,w,h);}
    void __global__ transpose_pre_double(cuDoubleComplex *dst, const int w, const int h) { transpose_pre(dst,w,h);}
    void __global__ transpose_float(Quaternionf *dst, const Quaternionf *src, const int w, const int h) { transpose(dst,src,w,h);}
    void __global__ transpose_double(Quaternion *dst, const Quaternion *src, const int w, const int h) { transpose(dst,src,w,h);}
    void __global__ ctranspose_float(Quaternionf *dst, const Quaternionf *src, const int w, const int h) { ctranspose(dst,src,w,h);}
    void __global__ ctranspose_double(Quaternion *dst, const Quaternion *src, const int w, const int h) { ctranspose(dst,src,w,h);}
}
