#ifndef __CUDA_QUATERNIONS__
#define __CUDA_QUATERNIONS__

#include <float.h>
#define COMPARE_THRESHOLD (DBL_EPSILON*1e4)
#define COMPARE_THRESHOLD_F (FLT_EPSILON*1e2)

typedef float4 Quaternionf;
typedef double4 Quaternion;
typedef float4 QuaternionArgf;
typedef double4 QuaternionArg;

__device__ inline Quaternion quaternion(double r,double i, double j, double k);
__device__ inline Quaternion qadd(Quaternion a, Quaternion b);
__device__ inline Quaternion qsub(Quaternion a, Quaternion b);
__device__ inline Quaternion qmul(Quaternion a, Quaternion b);
__device__ inline Quaternion qmul(Quaternion a, double b);
__device__ inline Quaternion qdiv(Quaternion a, Quaternion b);
__device__ inline Quaternion qdiv(Quaternion a, double b);
__device__ inline double qnorm(Quaternion a);
__device__ inline double qnorm2(Quaternion a);
__device__ inline Quaternion qconj(Quaternion q);
__device__ inline Quaternion qinv(Quaternion q);

__device__ inline Quaternion quaternion(QuaternionArg q);
__device__ inline QuaternionArg quaternionarg(double q,double phi, double theta, double psi);
__device__ inline QuaternionArg quaternionarg(Quaternion q);
__device__ inline bool comp(double a, double b);
__device__ inline bool comp(Quaternion a, Quaternion b);


__device__ inline Quaternionf quaternion(float r,float i, float j, float k);
__device__ inline Quaternionf qadd(Quaternionf a, Quaternionf b);
__device__ inline Quaternionf qsub(Quaternionf a, Quaternionf b);
__device__ inline Quaternionf qmul(Quaternionf a, Quaternionf b);
__device__ inline Quaternionf qmul(Quaternionf a, float b);
__device__ inline Quaternionf qdiv(Quaternionf a, Quaternionf b);
__device__ inline Quaternionf qdiv(Quaternionf a, float b);
__device__ inline float qnorm(Quaternionf a);
__device__ inline float qnorm2(Quaternionf a);
__device__ inline Quaternionf qconj(Quaternionf q);
__device__ inline Quaternionf qinv(Quaternionf q);

__device__ inline Quaternionf quaternion(QuaternionArgf q);
__device__ inline QuaternionArgf quaternionarg(float q,float phi, float theta, float psi);
__device__ inline QuaternionArgf quaternionarg(Quaternionf q);
__device__ inline bool comp(float a, float b);
__device__ inline bool comp(Quaternionf a, Quaternionf b);

__device__
inline Quaternion quaternion(double r, double i, double j, double k)
{
    return make_double4(r, i, j, k);
}

__device__
inline Quaternion quaternion(QuaternionArg q)
{
    double q0, q1, q2, q3;
    q0=q.x*(cos(q.y)*cos(q.z)*cos(q.w) + sin(q.y)*sin(q.z)*sin(q.w));
    q1=q.x*(sin(q.y)*cos(q.z)*cos(q.w) - cos(q.y)*sin(q.z)*sin(q.w));
    q2=q.x*(cos(q.y)*sin(q.z)*cos(q.w) - sin(q.y)*cos(q.z)*sin(q.w));
    q3=q.x*(cos(q.y)*cos(q.z)*sin(q.w) + sin(q.y)*sin(q.z)*cos(q.w));
    return make_double4(q0,q1,q2,q3);
}

__device__
inline QuaternionArg quaternionarg(double q,double phi, double theta, double psi)
{
    return make_double4(q, phi, theta, psi);
}

__device__
inline QuaternionArg quaternionarg(Quaternion q)
{
    Quaternion qc;
    double qn,val,psi,phi,theta;
    qn = qnorm(q);
    if (qn==0.0)
    {
        return make_double4(0,0,0,0);
    }
    q = qdiv(q, qn);

    val = 2.0*(q.y*q.z - q.x*q.w);

    if (val > 1.0)
    {
        val = 1.0;
    } else if( val < -1.0)
    {
        val = -1.0;
    }
    psi = -0.5*asin(val);

    if ((psi != M_PI/4) && (psi != -M_PI/4)){
        qc = make_double4(q.x, q.y, -q.z, q.w); /* beta(conj(q)) */
        qc = qmul(q,qc);
        phi  = 0.5*atan2(qc.y, qc.x);

        qc = make_double4(q.x, -q.y, q.z, q.w); /* alpha(conj(q)) */
        qc = qmul(qc,q);
        theta= 0.5*atan2(qc.z, qc.x);
    }
    else {
        phi = 0;
        qc = make_double4(q.x, q.y, q.z, -q.w); /* gamma(conj(q)) */
        qc = qmul(qc,q);
        theta= 0.5*atan2(qc.z, qc.x);
    }

    qc = quaternion(quaternionarg(1.0,phi,theta,psi));
    if (comp(qc, qmul(q,-1.0))){
        phi = phi - (phi>=0.0?1.0:-1.0)*M_PI;
    }

    return quaternionarg(qn, phi, theta, psi);
}

__device__
inline bool comp(double a, double b)
{
    return fabs(a-b) < (fmax(fabs(a),fabs(b))*COMPARE_THRESHOLD);
}

__device__
inline bool comp(Quaternion a, Quaternion b)
{
    return (comp(a.x, b.x) && comp(a.y, b.y) && comp(a.z, b.z) && comp(a.w, b.w));
}

__device__
inline Quaternion qadd(Quaternion a, Quaternion b)
{
    Quaternion ans;
    ans = make_double4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
    return ans;
}

__device__
inline Quaternion qsub(Quaternion a, Quaternion b)
{
    Quaternion ans;
    ans = make_double4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
    return ans;
}

__device__
inline Quaternion qmul(Quaternion a, Quaternion b)
{
    Quaternion ans;
    ans = make_double4(
            a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
            a.y*b.x + a.x*b.y - a.w*b.z + a.z*b.w,
            a.z*b.x + a.w*b.y + a.x*b.z - a.y*b.w,
            a.w*b.x - a.z*b.y + a.y*b.z + a.x*b.w);
    return ans;
}

__device__
inline Quaternion qmul(Quaternion a, double b)
{
    Quaternion ans;
    ans = make_double4(a.x*b, a.y*b, a.z*b, a.w*b);
    return ans;
}

__device__
inline double qnorm2(Quaternion a)
{
    double ans;
    ans = a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w;
    return ans;
}

__device__
inline double qnorm(Quaternion a)
{
    double ans;
    ans = sqrt(qnorm2(a));
    return ans;
}

__device__
inline Quaternion qconj(Quaternion a)
{
    Quaternion ans;
    ans = make_double4(a.x, -a.y, -a.z, -a.w);
    return ans;
}

__device__
inline Quaternion qinv(Quaternion a)
{
    Quaternion ans;
    ans = qdiv(a, qnorm2(a));
    return ans;
}

__device__
inline Quaternion qdiv(Quaternion a, Quaternion b)
{
    Quaternion ans;
    ans = qmul(a, qinv(b));
    return ans;
}

__device__
inline Quaternion qdiv(Quaternion a, double b)
{
    Quaternion ans;
    ans = make_double4(a.x/b, a.y/b, a.z/b, a.w/b);
    return ans;
}





__device__
inline Quaternionf quaternion(float r, float i, float j, float k)
{
    return make_float4(r, i, j, k);
}

__device__
inline Quaternionf quaternion(QuaternionArgf q)
{
    float q0, q1, q2, q3;
    q0=q.x*(cosf(q.y)*cosf(q.z)*cosf(q.w) + sinf(q.y)*sinf(q.z)*sinf(q.w));
    q1=q.x*(sinf(q.y)*cosf(q.z)*cosf(q.w) - cosf(q.y)*sinf(q.z)*sinf(q.w));
    q2=q.x*(cosf(q.y)*sinf(q.z)*cosf(q.w) - sinf(q.y)*cosf(q.z)*sinf(q.w));
    q3=q.x*(cosf(q.y)*cosf(q.z)*sinf(q.w) + sinf(q.y)*sinf(q.z)*cosf(q.w));
    return make_float4(q0,q1,q2,q3);
}

__device__
inline QuaternionArgf quaternionarg(float q, float phi, float theta, float psi)
{
    return make_float4(q, phi, theta, psi);
}

__device__
inline QuaternionArgf quaternionarg(Quaternionf q)
{
    Quaternionf qc;
    float qn,val,psi,phi,theta;
    qn = qnorm(q);
    if (qn==0.0f)
    {
        return make_float4(0.0f,0.0f,0.0f,0.0f);
    }
    q = qdiv(q, qn);

    val = 2.0f*(q.y*q.z - q.x*q.w);

    if (val > 1.0f)
    {
        val = 1.0f;
    } else if( val < -1.0f)
    {
        val = -1.0f;
    }
    psi = -0.5f*asinf(val);

    if ((psi != M_PI/4) && (psi != -M_PI/4)){
        qc = make_float4(q.x, q.y, -q.z, q.w);
        qc = qmul(q,qc);
        phi  = 0.5f*atan2f(qc.y, qc.x);

        qc = make_float4(q.x, -q.y, q.z, q.w);
        qc = qmul(qc,q);
        theta= 0.5f*atan2f(qc.z, qc.x);
    }
    else {
        phi = 0.0f;
        qc = make_float4(q.x, q.y, q.z, -q.w);
        qc = qmul(qc,q);
        theta= 0.5f*atan2f(qc.z, qc.x);
    }

    qc = quaternion(quaternionarg(1.0f,phi,theta,psi));
    if (comp(qc, qmul(q,-1.0f))){
        phi = phi - (phi>=0.0f?1.0f:-1.0f)*M_PI;
    }

    return quaternionarg(qn, phi, theta, psi);
}

__device__
inline bool comp(float a, float b)
{
    return fabsf(a-b) < (fmaxf(fabsf(a),fabsf(b))*COMPARE_THRESHOLD_F);
}

__device__
inline bool comp(Quaternionf a, Quaternionf b)
{
    return (comp(a.x, b.x) && comp(a.y, b.y) && comp(a.z, b.z) && comp(a.w, b.w));
}

__device__
inline Quaternionf qadd(Quaternionf a, Quaternionf b)
{
    Quaternionf ans;
    ans = make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
    return ans;
}

__device__
inline Quaternionf qsub(Quaternionf a, Quaternionf b)
{
    Quaternionf ans;
    ans = make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
    return ans;
}

__device__
inline Quaternionf qmul(Quaternionf a, Quaternionf b)
{
    Quaternionf ans;
    ans = make_float4(
            a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
            a.y*b.x + a.x*b.y - a.w*b.z + a.z*b.w,
            a.z*b.x + a.w*b.y + a.x*b.z - a.y*b.w,
            a.w*b.x - a.z*b.y + a.y*b.z + a.x*b.w);
    return ans;
}

__device__
inline Quaternionf qmul(Quaternionf a, float b)
{
    Quaternionf ans;
    ans = make_float4(a.x*b, a.y*b, a.z*b, a.w*b);
    return ans;
}

__device__
inline float qnorm2(Quaternionf a)
{
    float ans;
    ans = a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w;
    return ans;
}

__device__
inline float qnorm(Quaternionf a)
{
    float ans;
    ans = sqrt(qnorm2(a));
    return ans;
}

__device__
inline Quaternionf qconj(Quaternionf a)
{
    Quaternionf ans;
    ans = make_float4(a.x, -a.y, -a.z, -a.w);
    return ans;
}

__device__
inline Quaternionf qinv(Quaternionf a)
{
    Quaternionf ans;
    ans = qdiv(a, qnorm2(a));
    return ans;
}

__device__
inline Quaternionf qdiv(Quaternionf a, Quaternionf b)
{
    Quaternionf ans;
    ans = qmul(a, qinv(b));
    return ans;
}

__device__
inline Quaternionf qdiv(Quaternionf a, float b)
{
    Quaternionf ans;
    ans = make_float4(a.x/b, a.y/b, a.z/b, a.w/b);
    return ans;
}

#endif

