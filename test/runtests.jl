using Test
using RustFFT
using JlrsCore: JlrsError

using AbstractFFTs

@testset "Forward FFT" begin
    @test begin
        data = ones(ComplexF64, 1)
        fft!(data)
        data[1] ≈ 1.0
    end

    @test begin
        data = ones(ComplexF64, 2)
        fft!(data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end

    @test begin
        data = ones(ComplexF32, 1)
        fft!(data)
        data[1] ≈ 1.0
    end

    @test begin
        data = ones(ComplexF32, 2)
        fft!(data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end
end

@testset "Inverse FFT" begin
    @test begin
        data = ones(ComplexF64, 1)
        bfft!(data)
        data[1] ≈ 1.0
    end

    @test begin
        data = [one(ComplexF64), zero(ComplexF64)]
        bfft!(data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end

    @test begin
        data = ones(ComplexF32, 1)
        bfft!(data)
        data[1] ≈ 1.0
    end

    @test begin
        data = [one(ComplexF32), zero(ComplexF32)]
        bfft!(data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end
end

@testset "Exceptions" begin
    @test_throws JlrsError begin
        longer = ones(ComplexF64, 2)
        plan = plan_fft(longer)
        plan * ones(ComplexF64, 1)
    end
end

# Fails because RustFFT only supports 1D data.
# AbstractFFTs.TestUtils.test_complex_ffts(Vector{ComplexF32}; test_adjoint=false)
