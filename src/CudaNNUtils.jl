module CudaNNUtils

using CUDArt
using Quaternions
using QuaternionArgs
import Base: transpose, ctranspose

const ptxdict = Dict()
const mdlist = Array(CuModule, 0)
const block_default = 128

function init(devlist)
    global ptxdict
    global mdlist
    isempty(mdlist) || return
    for dev in devlist
        device(dev)
        mod_dir = joinpath(dirname(@__FILE__), "kernels")
        mod_path = joinpath(mod_dir, "kernels.ptx")
        md = CuModule(mod_path, false) 
        ptxdict[(dev, "equiv", Complex{Float32}, Quaternion{Float32})] = CuFunction(md, "q2c_float")
        ptxdict[(dev, "equiv", Complex{Float64}, Quaternion{Float64})] = CuFunction(md, "q2c_double")
        ptxdict[(dev, "equiv", Quaternion{Float32}, Complex{Float32})] = CuFunction(md, "c2q_float")
        ptxdict[(dev, "equiv", Quaternion{Float64}, Complex{Float64})] = CuFunction(md, "c2q_double")
        ptxdict[(dev, "actfunc", Float32)] = CuFunction(md, "actfunc_real_float")
        ptxdict[(dev, "actfunc", Float64)] = CuFunction(md, "actfunc_real_double")
        ptxdict[(dev, "actfunc", Complex{Float32})] = CuFunction(md, "actfunc_comp_float")
        ptxdict[(dev, "actfunc", Complex{Float64})] = CuFunction(md, "actfunc_comp_double")
        ptxdict[(dev, "actfunc", Quaternion{Float32})] = CuFunction(md, "actfunc_quat_float")
        ptxdict[(dev, "actfunc", Quaternion{Float64})] = CuFunction(md, "actfunc_quat_double")
        ptxdict[(dev, "actfunc_multi", Complex{Float32})] = CuFunction(md, "actfunc_comp_multi_float")
        ptxdict[(dev, "actfunc_multi", Complex{Float64})] = CuFunction(md, "actfunc_comp_multi_double")
        ptxdict[(dev, "actfunc_multi", Quaternion{Float32})] = CuFunction(md, "actfunc_quat_multi_float")
        ptxdict[(dev, "actfunc_multi", Quaternion{Float64})] = CuFunction(md, "actfunc_quat_multi_double")
        ptxdict[(dev, "fill", Float32)] = CuFunction(md, "fill_real_float")
        ptxdict[(dev, "fill", Float64)] = CuFunction(md, "fill_real_double")
        ptxdict[(dev, "fill", Complex{Float32})] = CuFunction(md, "fill_comp_float")
        ptxdict[(dev, "fill", Complex{Float64})] = CuFunction(md, "fill_comp_double")
        ptxdict[(dev, "fill", Quaternion{Float32})] = CuFunction(md, "fill_quat_float")
        ptxdict[(dev, "fill", Quaternion{Float64})] = CuFunction(md, "fill_quat_double")
        ptxdict[(dev, "compare", Int32)] = CuFunction(md, "compare_real_int")
        ptxdict[(dev, "compare", Float32)] = CuFunction(md, "compare_real_float")
        ptxdict[(dev, "compare", Float64)] = CuFunction(md, "compare_real_double")
        ptxdict[(dev, "compare", Complex{Float32})] = CuFunction(md, "compare_comp_float")
        ptxdict[(dev, "compare", Complex{Float64})] = CuFunction(md, "compare_comp_double")
        ptxdict[(dev, "compare", Quaternion{Float32})] = CuFunction(md, "compare_quat_float")
        ptxdict[(dev, "compare", Quaternion{Float64})] = CuFunction(md, "compare_quat_double")
        ptxdict[(dev, "transpose_pre", Complex{Float32})] = CuFunction(md, "transpose_pre_float")
        ptxdict[(dev, "transpose_pre", Complex{Float64})] = CuFunction(md, "transpose_pre_double")
        ptxdict[(dev, "transpose", Quaternion{Float32})] = CuFunction(md, "transpose_float")
        ptxdict[(dev, "transpose", Quaternion{Float64})] = CuFunction(md, "transpose_double")
        ptxdict[(dev, "ctranspose", Quaternion{Float32})] = CuFunction(md, "ctranspose_float")
        ptxdict[(dev, "ctranspose", Quaternion{Float64})] = CuFunction(md, "ctranspose_double")
        push!(mdlist, md)
    end
end

function close()
    for md in mdlist
        unload(md)
    end
    empty!(mdlist)
    empty!(ptxdict)
end

function equiv!{T}(dst::CudaArray{Complex{T}}, src::CudaArray{Quaternion{T}})
    @assert(length(dst) == (4*length(src)))
    dev = device(dst)
    w = size(src,1)
    h = size(src,2)
    block = (16,16)
    grid = (ceil(Int,w/16),ceil(Int,h/16))
    func = ptxdict[(dev, "equiv", Complex{T}, Quaternion{T})]
    CUDArt.launch(func, grid, block, (dst, src, w, h))
end

function equiv!{T}(dst::CudaArray{Quaternion{T}}, src::CudaArray{Complex{T}})
    @assert((4*length(dst)) == length(src))
    dev = device(dst)
    w = size(dst,1)
    h = size(dst,2)
    block = (16,16)
    grid = (ceil(Int,w/16),ceil(Int,h/16))
    func = ptxdict[(dev, "equiv", Quaternion{T}, Complex{T})]
    CUDArt.launch(func, grid, block, (dst, src, w, h))
end

function equiv{T}(src::CudaArray{Quaternion{T}})
    w = size(src,1)
    h = size(src,2)
    dst = CudaArray(Complex{T}, w<<1, h<<1)
    equiv!(dst, src)
    return dst
end

function equiv{T}(src::CudaArray{Complex{T}})
    w = size(src,1)
    h = size(src,2)
    dst = CudaArray(Quaternion{T}, w>>1, h>>1)
    equiv!(dst, src)
    return dst
end

function actfunc!{T}(data::CudaArray{T}; block=block_default)
    dev = device(data)
    N = length(data)
    grid = ceil(Int,N/block)
    func = ptxdict[(dev, "actfunc", T)]
    CUDArt.launch(func, grid, block, (data,N))
end

function actfunc_multi!{T<:Complex}(data::CudaArray{T}, K::Integer; block=block_default)
    dev = device(data)
    N = length(data)
    grid = ceil(Int,N/block)
    func = ptxdict[(dev, "actfunc_multi", T)]
    CUDArt.launch(func, grid, block, (data,K,N))
end

function actfunc_multi!{T<:Quaternion}(data::CudaArray{T}, A::Integer, B::Integer, C::Integer; block=block_default)
    dev = device(data)
    N = length(data)
    grid = ceil(Int,N/block)
    func = ptxdict[(dev, "actfunc_multi", T)]
    CUDArt.launch(func, grid, block, (data,A,B,C,N))
end

function fill!{T}(data::CudaArray{T}, val::T; block=block_default)
    dev = device(data)
    N = length(data)
    grid = ceil(Int,N/block)
    func = ptxdict[(dev, "fill", T)]
    CUDArt.launch(func, grid, block, (data,val,N))
end

function fill!{T}(data::CudaArray{T}, val::T; block=block_default)
    dev = device(data)
    N = length(data)
    grid = ceil(Int,N/block)
    func = ptxdict[(dev, "fill", T)]
    CUDArt.launch(func, grid, block, (data,val,N))
end

function transpose_pre!{T}(dst::CudaArray{T}; block=block_default)
    dev = device(dst)
    w = size(dst,1)
    h = size(dst,2)
    block = (16,16)
    grid = (ceil(Int,w/16),ceil(Int,h/16))
    func = ptxdict[(dev, "transpose_pre", T)]
    CUDArt.launch(func, grid, block, (dst, w, h))
end

function transpose!{T<:Quaternion}(dst::CudaArray{T}, src::CudaArray{T}; block=block_default)
    dev = device(dst)
    w = size(dst,1)
    h = size(dst,2)
    block = (16,16)
    grid = (ceil(Int,w/16),ceil(Int,h/16))
    func = ptxdict[(dev, "transpose", T)]
    CUDArt.launch(func, grid, block, (dst, src, w, h))
end

function transpose{T<:Quaternion}(src::CudaArray{T}; block=block_default)
    w = size(src,1)
    h = size(src,2)
    dst = CudaArray(T, h,w)
    transpose!(dst, src)
    return dst
end

function ctranspose!{T<:Quaternion}(dst::CudaArray{T}, src::CudaArray{T}; block=block_default)
    dev = device(dst)
    w = size(dst,1)
    h = size(dst,2)
    block = (16,16)
    grid = (ceil(Int,w/16),ceil(Int,h/16))
    func = ptxdict[(dev, "ctranspose", T)]
    CUDArt.launch(func, grid, block, (dst, src, w, h))
end

function ctranspose{T<:Quaternion}(src::CudaArray{T}; block=block_default)
    w = size(src,1)
    h = size(src,2)
    dst = CudaArray(T, h,w)
    ctranspose!(dst, src)
    return dst
end
end # module
