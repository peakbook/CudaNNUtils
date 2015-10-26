using CUDArt
using CUBLAS
using CudaNNUtils
using Quaternions
using QuaternionArgs
using Base.Test


function test_equiv(n,m)
    for typ in [Float32, Float64]
        A = rand(Quaternion{typ}, n,m)

        d_A = CudaArray(A)
        d_B = CudaArray(Complex{typ}, 2n,2m)

        CudaNNUtils.equiv!(d_B, d_A)
        B = Quaternions.equiv(A)
        h_B = to_host(d_B)
        @test h_B == B

        CudaNNUtils.equiv!(d_A, d_B)
        h_A = to_host(d_A)
        @test h_A == A

        CUDArt.free(d_A)
        CUDArt.free(d_B)
    end
end

function test_equiv_mul(n)
    # parameters
    typ = Float64
    alpha = one(Complex{typ})
    beta = one(Complex{typ})

    # generate quaternion matrices
    A = rand(Quaternion{typ},n,n)
    B = rand(Quaternion{typ},n)
    C = zeros(Quaternion{typ},n)

    # generate equivalent complex matrices
    Ac = Quaternions.equiv(A)
    Bc = Quaternions.equiv(B)
    Cc = Quaternions.equiv(C)

    # allocate device memory
    d_A = CudaArray(Ac)
    d_B = CudaArray(Bc)
    d_C = CudaArray(Cc)

    # calc
    CUBLAS.gemm!('N','N',alpha,d_A,d_B,beta,d_C)
    C = (alpha*A)*B + beta*C

    # compare
    h_C = to_host(d_C)
    @test_approx_eq Quaternions.equiv(C) h_C

    for ptr in [d_A, d_B, d_C]
        CUDArt.free(ptr)
    end
end

function test_fill(n,m)
    for typ in [Float32, Float64]
        A = rand(typ, n,m)

        d_A = CudaArray(A)

        CudaNNUtils.fill!(d_A, zero(typ))
        s = sum(to_host(d_A))
        @test s == zero(typ)

        CUDArt.free(d_A)
    end
end

function test_transpose(n,m)
    for typ in [Quaternion{Float32}, Quaternion{Float64}]
        A = rand(typ, n,m)

        d_A = CudaArray(A)
        d_tA = CudaArray(eltype(A), size(A,2), size(A,1))

        CudaNNUtils.transpose!(d_tA, d_A)
        h_tA = to_host(d_tA)
        @test h_tA == transpose(A)

        CUDArt.free(d_A)
        CUDArt.free(d_tA)
    end
end

function test_ctranspose(n,m)
    for typ in [Quaternion{Float32}, Quaternion{Float64}]
        A = rand(typ, n,m)

        d_A = CudaArray(A)
        d_tA = CudaArray(eltype(A), size(A,2), size(A,1))

        CudaNNUtils.ctranspose!(d_tA, d_A)
        h_tA = to_host(d_tA)
        @test h_tA == A'

        CUDArt.free(d_A)
        CUDArt.free(d_tA)
    end
end

devlist = devices(dev->capability(dev)[1]>=1.3, nmax=1)
if length(devlist) != 1
    warn("NO DEVICE FOR TESTING")
    exit()
end

device_synchronize()
CudaNNUtils.init(devlist)

N = 100
M = 200
test_equiv(N, M)
test_equiv_mul(N)
test_fill(N, M)
test_transpose(N, M)
test_ctranspose(N, M)

CudaNNUtils.close()

