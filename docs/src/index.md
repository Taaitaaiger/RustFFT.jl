# RustFFT Documentation

Compute FFTs in Julia using RustFFT. Some parts of this documentation have been quoted from the [RustFFT docs](https://docs.rs/rustfft/latest/rustfft/).

> RustFFT is a high-performance, SIMD-accelerated FFT library written in pure Rust. It can compute FFTs of any size, including prime-number sizes, in O(nlogn) time.

## Usage

RustFFT.jl implements the generic FFT interface of [AbstractFFTs.jl](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#Public-Interface-1) but only supports one-dimensional, contiguous, complex-valued arrays: `Vector{ComplexF64}` and `Vector{ComplexF32}`.

Forward and inverse FFT:

```julia
using RustFFT

data = ones(ComplexF64, 1)
fft!(instance, data)
```

```julia
using RustFFT

data = ones(ComplexF64, 1)
ifft!(instance, data)
```

You can set several options by planning the FFT:

```julia
using RustFFT

planner = new_planner(ComplexF64)
data = ones(ComplexF64, 1)
plan = plan_fft!(data;
    rustfft_checks=IgnoreArrayChecks(), # Don't track the array or check if its length is compatible
    rustfft_gcsafe=GcSafe(), # Don't block the GC while an FFT is computed with this plan
    rustfft_planner=planner) # Reuse an existing planner
plan * data
```

It's currently not possible to choose the specific algorithm that will be used to compute the transform.

## RustFFT-specific API

```@autodocs
Modules = [RustFFT]
```

## Internals

```@autodocs
Modules = [RustFFT.Internal]
```
