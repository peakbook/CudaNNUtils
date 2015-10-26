#include <math.h>
#include <cuComplex.h>
#include "quaternions.cu"

#define SIGN(x) (x>=0.0?1.0:-1.0)

template <typename T>
__device__ void actfunc_real(T *a, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N) a[i] = SIGN(a[i]);
}

__device__ void actfunc_comp(cuFloatComplex *a, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N) a[i] = make_cuFloatComplex(SIGN(cuCrealf(a[i])),SIGN(cuCimagf(a[i])));
}

__device__ void actfunc_comp(cuDoubleComplex *a, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N) a[i] = make_cuDoubleComplex(SIGN(cuCreal(a[i])),SIGN(cuCimag(a[i])));
}

__device__ void actfunc_quat(Quaternionf *a, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N) a[i] = make_float4(SIGN(a[i].x),SIGN(a[i].y),SIGN(a[i].z),SIGN(a[i].w));
}

__device__ void actfunc_quat(Quaternion *a, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N) a[i] = make_double4(SIGN(a[i].x),SIGN(a[i].y),SIGN(a[i].z),SIGN(a[i].w));
}

__device__ void actfunc_comp_multi(cuFloatComplex *a, const int K, const int N)
{
    cuFloatComplex t;
    float hphi, phi, theta, w;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i<N){
        hphi = M_PI/K;
        phi = 2.0f * hphi;
        t = a[i];
        theta = atan2f(cuCimagf(t),cuCrealf(t))+hphi;
        w = floorf(theta / phi) * phi;

        a[i] = make_cuFloatComplex(cosf(w),sinf(w));
    }
}

__device__ void actfunc_comp_multi(cuDoubleComplex *a, const int K, const int N)
{
    cuDoubleComplex t;
    double hphi, phi, theta, w;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i<N){
        hphi = M_PI/K;
        phi = 2.0 * hphi;
        t = a[i];
        theta = atan2(cuCimag(t),cuCreal(t))+hphi;
        w = floor(theta / phi) * phi;

        a[i] = make_cuDoubleComplex(cos(w),sin(w));
    }
}

__device__ float qsign(float phase, float coef, const int K)
{
    float dphase, phase0, k, p;

    dphase = 2*M_PI/K*coef;
    phase0 = M_PI*coef;
    k = roundf((phase-0.5f*dphase + phase0)/dphase);
    p = k*dphase - phase0;

    return p+0.5f*dphase;
}

__device__ void actfunc_quat_multi(Quaternionf *a, const int A, const int B, const int C, const int N)
{
    float phi, theta, psi;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N){
        QuaternionArgf q = quaternionarg(a[i]);

        phi = qsign(q.y, 1.0f, A);
        theta = qsign(q.z, 0.5f, B);
        psi = qsign(q.w, 0.25f, C);

        a[i] = quaternion(quaternionarg(1.0f, phi, theta, psi));
    }
}

__device__ double qsign(double phase, double coef, int K)
{
    double dphase, phase0, k, p;

    dphase = 2*M_PI/K*coef;
    phase0 = M_PI*coef;
    k = round((phase-0.5*dphase + phase0)/dphase);
    p = k*dphase - phase0;

    return p+0.5*dphase;
}

__device__ void actfunc_quat_multi(Quaternion *a, const int A, const int B, const int C, const int N)
{
    double phi, theta, psi;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N){
        QuaternionArg q = quaternionarg(a[i]);

        phi = qsign(q.y, 1.0, A);
        theta = qsign(q.z, 0.5, B);
        psi = qsign(q.w, 0.25, C);

        a[i] = quaternion(quaternionarg(1.0, phi, theta, psi));
    }
}

extern "C"
{
    void __global__ actfunc_real_float(float *a, const int N) { actfunc_real(a, N);}
    void __global__ actfunc_real_double(double *a, const int N) { actfunc_real(a, N);}

    void __global__ actfunc_comp_float(cuFloatComplex *a, const int N) { actfunc_comp(a, N);}
    void __global__ actfunc_comp_double(cuDoubleComplex *a, const int N) { actfunc_comp(a, N);}

    void __global__ actfunc_quat_float(Quaternionf *a, const int N) { actfunc_quat(a, N);}
    void __global__ actfunc_quat_double(Quaternion *a, const int N) { actfunc_quat(a, N);}

    void __global__ actfunc_comp_multi_float(cuFloatComplex *a, const int K, const int N) { actfunc_comp_multi(a, K, N);}
    void __global__ actfunc_comp_multi_double(cuDoubleComplex *a, const int K, const int N) { actfunc_comp_multi(a, K, N);}

    void __global__ actfunc_quat_multi_float(Quaternionf *a, const int A, const int B, const int C, const int N) { actfunc_quat_multi(a, A, B, C, N);}
    void __global__ actfunc_quat_multi_double(Quaternion *a, const int A, const int B, const int C, const int N) { actfunc_quat_multi(a, A, B, C, N);}
}
