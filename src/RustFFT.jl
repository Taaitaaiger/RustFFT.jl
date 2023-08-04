module RustFFT
using Reexport

include("Internal.jl")
using .Internal

@reexport using AbstractFFTs

import Base: *, size
import AbstractFFTs: Plan, ScaledPlan, plan_fft, fft, plan_fft!, fft!, plan_bfft, plan_bfft!,
    ifft, ifft!, fftdims, plan_inv
import LinearAlgebra: mul!

export GcSafe, GcUnsafe, AllArrayChecks, IgnoreArrayTracking, IgnoreArrayChecks, Forward, Backward, new_planner

"""
    GcSafety

Whether or not a planned FFT is executed in a GC-safe state.
"""
abstract type GcSafety end

"""
    GcSafe()

A GC-safe plan doesn't block the GC.
"""
struct GcSafe <: GcSafety end

"""
    GcUnsafe()

A GC-unsafe plan can block the GC. This is the default.
"""
struct GcUnsafe <: GcSafety end

"""
    Direction

The direction of the plan.
"""
abstract type Direction end

"""
    Forward()

The plan computes a forward FFT.
"""
struct Forward <: Direction end

"""
    Backward()

The plan computes a backward FFT.
"""
struct Backward <: Direction end

"""
    ArrayChecks

Safety checks that can be disabled.
"""
abstract type ArrayChecks end

"""
    AllArrayChecks()

Track the array and check if the length is compatible with the plan. This is the default.
"""
struct AllArrayChecks <: ArrayChecks end

"""
    IgnoreArrayTracking()

Only check if the length is compatible with the plan.
"""
struct IgnoreArrayTracking <: ArrayChecks end

"""
    IgnoreArrayChecks()

Perform no safety checks.
"""
struct IgnoreArrayChecks <: ArrayChecks end

# RustFFT only supports contiguous, one-dimensional arrays whose elements are either `ComplexF64`
# or `ComplexF32`.
const RustFFTNumber = Union{Complex{Float64},Complex{Float32}}

mutable struct RustFFTPlan{T<:RustFFTNumber,inplace,direction<:Direction,gcsafety<:GcSafety,arraychecks<:ArrayChecks} <: Plan{T}
    plan::FftInstance{T}
    pinv::ScaledPlan

    function RustFFTPlan{T,inplace,direction,gcsafety,arraychecks}(plan::FftInstance{T}) where
    {T<:RustFFTNumber,inplace,direction<:Direction,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

        new(plan)
    end
end

"""
    new_planner(::Type{ComplexF32})
    new_planner(::Type{ComplexF64})

Returns a new planner. By default each plan creates a new planner but one can also be provided to
the `plan_*`-functions as a keyword argument: `rustfft_planner`. By reusing the same planner,
internal data is reused across different plans saving memory and setup time.
"""
function new_planner end

@inline function new_planner(::Type{ComplexF32})::FftPlanner{ComplexF32}
    return FftPlanner32()
end

@inline function new_planner(::Type{ComplexF64})::FftPlanner{ComplexF64}
    return FftPlanner64()
end

function plan_fft(x::Vector{T}, region;
    rustfft_checks::arraychecks=IgnoreArrayTracking(),
    rustfft_gc_safe::gcsafety=GcUnsafe(),
    rustfft_planner::Union{FftPlanner{T},Nothing}=nothing,
    kws...) where
{T<:RustFFTNumber,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    instance = if isnothing(rustfft_planner)
        planner = new_planner(T)
        rustfft_plan_fft_forward_untracked!(planner, UInt(length(x)))
    else
        rustfft_plan_fft_forward!(rustfft_planner, UInt(length(x)))
    end

    RustFFTPlan{T,false,Forward,gcsafety,arraychecks}(instance)
end

function plan_fft!(x::Vector{T}, region;
    rustfft_checks::arraychecks=AllArrayChecks(),
    rustfft_gc_safe::gcsafety=GcUnsafe(),
    rustfft_planner::Union{FftPlanner{T},Nothing}=nothing,
    kws...) where
{T<:RustFFTNumber,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    instance = if isnothing(rustfft_planner)
        planner = new_planner(T)
        rustfft_plan_fft_forward_untracked!(planner, UInt(length(x)))
    else
        rustfft_plan_fft_forward!(rustfft_planner, UInt(length(x)))
    end

    RustFFTPlan{T,true,Forward,gcsafety,arraychecks}(instance)
end

function plan_bfft(x::Vector{T}, region;
    rustfft_checks::arraychecks=IgnoreArrayTracking(),
    rustfft_gc_safe::gcsafety=GcUnsafe(),
    rustfft_planner::Union{FftPlanner{T},Nothing}=nothing,
    kws...) where
{T<:RustFFTNumber,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    instance = if isnothing(rustfft_planner)
        planner = new_planner(T)
        rustfft_plan_fft_inverse_untracked!(planner, UInt(length(x)))
    else
        rustfft_plan_fft_inverse!(rustfft_planner, UInt(length(x)))
    end

    RustFFTPlan{T,false,Backward,gcsafety,arraychecks}(instance)
end

function plan_bfft!(x::Vector{T}, region;
    rustfft_checks::arraychecks=AllArrayChecks(),
    rustfft_gc_safe::gcsafety=GcUnsafe(),
    rustfft_planner::Union{FftPlanner{T},Nothing}=nothing,
    kws...) where
{T<:RustFFTNumber,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    instance = if isnothing(rustfft_planner)
        planner = new_planner(T)
        rustfft_plan_fft_inverse_untracked!(planner, UInt(length(x)))
    else
        rustfft_plan_fft_inverse!(rustfft_planner, UInt(length(x)))
    end

    RustFFTPlan{T,true,Backward,gcsafety,arraychecks}(instance)
end

function plan_inv(p::RustFFTPlan{T,inplace,Forward,gcsafety,arraychecks}) where
{T<:RustFFTNumber,inplace,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    planner = new_planner(T)
    len = size(p)[1]
    instance = rustfft_plan_fft_inverse_untracked!(planner, UInt(len))
    ScaledPlan(RustFFTPlan{T,inplace,Backward,gcsafety,arraychecks}(instance), 1 / len)
end

function plan_inv(p::RustFFTPlan{T,inplace,Forward,gcsafety,arraychecks}, planner::FftPlanner{T}) where
{T<:RustFFTNumber,inplace,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    len = size(p)[1]
    instance = rustfft_plan_fft_inverse!(planner, UInt(len))
    ScaledPlan(RustFFTPlan{T,inplace,Backward,gcsafety,arraychecks}(instance), 1 / len)
end

function plan_inv(p::RustFFTPlan{T,inplace,Backward,gcsafety,arraychecks}) where
{T<:RustFFTNumber,inplace,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    planner = new_planner(T)
    len = size(p)[1]
    instance = rustfft_plan_fft_forward_untracked!(planner, UInt(len))
    ScaledPlan(RustFFTPlan{T,inplace,Forward,gcsafety,arraychecks}(instance), 1 / len)
end

function plan_inv(p::RustFFTPlan{T,inplace,Backward,gcsafety,arraychecks}, planner::FftPlanner{T}) where
{T<:RustFFTNumber,inplace,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    len = size(p)[1]
    instance = rustfft_plan_fft!(planner, UInt(len))
    ScaledPlan(RustFFTPlan{T,inplace,Forward,gcsafety,arraychecks}(instance), 1 / len)
end

function size(p::RustFFTPlan{T,inplace,direction,gcsafety,arraychecks}) where
{T<:RustFFTNumber,inplace,direction<:Direction,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    (Int(rustfft_plan_size(p.plan)),)
end

fftdims(::RustFFTPlan{T,inplace,direction,gcsafety,arraychecks}) where
{T<:RustFFTNumber,inplace,direction<:Direction,gcsafety<:GcSafety,arraychecks<:ArrayChecks} = 1

@inline function apply_plan!(y::Vector{T}, p::RustFFTPlan{T,inplace,direction,GcSafe,AllArrayChecks}) where
{T<:RustFFTNumber,inplace,direction<:Direction}

    rustfft_fft_gcsafe!(p.plan, y)
end

@inline function apply_plan!(y::Vector{T}, p::RustFFTPlan{T,inplace,direction,GcSafe,IgnoreArrayTracking}) where
{T<:RustFFTNumber,inplace,direction<:Direction}

    rustfft_fft_untracked_gcsafe!(p.plan, y)
end

@inline function apply_plan!(y::Vector{T}, p::RustFFTPlan{T,inplace,direction,GcSafe,IgnoreArrayChecks}) where
{T<:RustFFTNumber,inplace,direction<:Direction}

    rustfft_fft_unchecked_gcsafe!(p.plan, y)
end

@inline function apply_plan!(y::Vector{T}, p::RustFFTPlan{T,inplace,direction,GcUnsafe,AllArrayChecks}) where
{T<:RustFFTNumber,inplace,direction<:Direction}

    rustfft_fft!(p.plan, y)
end

@inline function apply_plan!(y::Vector{T}, p::RustFFTPlan{T,inplace,direction,GcUnsafe,IgnoreArrayTracking}) where
{T<:RustFFTNumber,inplace,direction<:Direction}

    rustfft_fft_untracked!(p.plan, y)
end

@inline function apply_plan!(y::Vector{T}, p::RustFFTPlan{T,inplace,direction,GcUnsafe,IgnoreArrayChecks}) where
{T<:RustFFTNumber,inplace,direction<:Direction}

    rustfft_fft_unchecked!(p.plan, y)
end

function mul!(y::Vector{T}, p::RustFFTPlan{T,false,direction,gcsafety,arraychecks}, x::Vector{T}) where
{T<:RustFFTNumber,direction<:Direction,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    copy!(y, x)
    apply_plan!(y, p)
    y
end

function mul!(::Vector{T}, p::RustFFTPlan{T,true,direction,gcsafety,arraychecks}, x::Vector{T}) where
{T<:RustFFTNumber,direction<:Direction,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    apply_plan!(x, p)
    x
end

function *(p::RustFFTPlan{T,false,direction,gcsafety,arraychecks}, x::Vector{T}) where
{T<:RustFFTNumber,direction<:Direction,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    y = copy(x)
    apply_plan!(y, p)
    y
end

function *(p::RustFFTPlan{T,true,direction,gcsafety,arraychecks}, x::Vector{T}) where
{T<:RustFFTNumber,direction<:Direction,gcsafety<:GcSafety,arraychecks<:ArrayChecks}

    apply_plan!(x, p)
    x
end
end
