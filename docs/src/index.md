# RustFFT Documentation

Compute FFTs in Julia using RustFFT. Some parts of this documentation have been quoted from the [RustFFT docs](https://docs.rs/rustfft/latest/rustfft/).

> RustFFT is a high-performance, SIMD-accelerated FFT library written in pure Rust. It can compute FFTs of any size, including prime-number sizes, in O(nlogn) time.

## Usage

Forward FFT:

```julia
using RustFFT

planner64 = RustFFT.FftPlanner64()
instance = RustFFT.plan_fft_forward(planner64, UInt(1))
data = complex([1.0])
RustFFT.fft!(instance, data)
@assert data[1] ≈ 1.0
```

Inverse FFT:

```julia
using RustFFT

planner64 = RustFFT.FftPlanner64()
instance = RustFFT.plan_fft_inverse(planner64, UInt(1))
data = complex([1.0])
RustFFT.fft!(instance, data)
@assert data[1] ≈ 1.0
```

Note that RustFFT does not normalize outputs:

> Callers must manually normalize the results by scaling each element by `1/len().sqrt()`. Multiple normalization steps can be merged into one via pairwise multiplication, so when doing a forward FFT followed by an inverse callers can normalize once by scaling each element by `1/len()`

It's currently not possible to choose the specific algorithm that will be used to compute the transform.

```@autodocs
Modules = [RustFFT]
```
