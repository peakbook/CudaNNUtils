extern __device__ inline bool comp(double a, double b);
extern __device__ inline bool comp(Quaternion a, Quaternion b);
extern __device__ inline bool comp(float a, float b);
extern __device__ inline bool comp(Quaternionf a, Quaternionf b);

__device__ inline bool comp(int a, int b){
    return a==b;
}

template <typename T>
__device__ void fill(T *a, T val, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N) a[i] = val;
}

template <typename T> 
__device__ void compare_real(int *x, const T *a, const T *b, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N) x[i] = comp(a[i], b[i]) ? 0 : 1;
}

template<typename T>
__device__ void compare_comp(int *x, const T *a, const T *b, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    T val_a = a[i];
    T val_b = b[i];
    if(i<N) x[i] = comp(val_a.x, val_b.x) && comp(val_a.y, val_b.y) ? 0 : 1;
}

template<typename T>
__device__ void compare_quat(int *x, const T *a, const T *b, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    T val_a = a[i];
    T val_b = b[i];
    if(i<N) x[i] = comp(val_a.x, val_b.x) && comp(val_a.y, val_b.y) && comp(val_a.z, val_b.z) && comp(val_a.w, val_b.w) ? 0 : 1; 
}

extern "C"
{
    void __global__ fill_real_float(float *a, float val, const int N) { fill(a, val, N);}
    void __global__ fill_real_double(double *a, double val, const int N) { fill(a, val, N);}
    void __global__ fill_comp_float(cuFloatComplex *a, cuFloatComplex val, const int N) { fill(a, val, N);}
    void __global__ fill_comp_double(cuDoubleComplex *a, cuDoubleComplex val, const int N) { fill(a, val, N);}
    void __global__ fill_quat_float(Quaternionf *a, Quaternionf val, const int N) { fill(a, val, N);}
    void __global__ fill_quat_double(Quaternion *a, Quaternion val, const int N) { fill(a, val, N);}

    void __global__ compare_real_int(int *x, const int *a, const int *b, const int N) { compare_real(x,a,b,N);}
    void __global__ compare_real_float(int *x, const float *a, const float *b, const int N) { compare_real(x,a,b,N);}
    void __global__ compare_real_double(int *x, const double *a, const double *b, const int N) { compare_real(x,a,b,N);}
    void __global__ compare_comp_float(int *x, const cuFloatComplex *a, const cuFloatComplex *b, const int N) { compare_comp(x,a,b,N);}
    void __global__ compare_comp_double(int *x, const cuDoubleComplex *a, const cuDoubleComplex *b, const int N) { compare_comp(x,a,b,N);}
    void __global__ compare_quat_float(int *x, const Quaternionf *a, const Quaternionf *b, const int N) { compare_quat(x,a,b,N);}
    void __global__ compare_quat_double(int *x, const Quaternion *a, const Quaternion *b, const int N) { compare_quat(x,a,b,N);}
}

