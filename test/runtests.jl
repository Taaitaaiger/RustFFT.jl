using Test
using RustFFT

@testset "Forward FFT" begin
    planner64 = RustFFT.FftPlanner64()
    planner32 = RustFFT.FftPlanner32()

    @test begin
        instance = RustFFT.plan_fft_forward(planner64, UInt(1))
        data = [1.0 + 0.0im]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_forward(planner64, UInt(2))
        data = [1.0 + 0.0im; 1.0 + 0.0im]
        RustFFT.fft!(instance, data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end

    @test begin
        instance = RustFFT.plan_fft_forward(planner32, UInt(1))
        data = [ComplexF32(1.0)]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_forward(planner32, UInt(2))
        data = [ComplexF32(1.0); ComplexF32(1.0)]
        RustFFT.fft!(instance, data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner64, :forward, UInt(1))
        data = [1.0 + 0.0im]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner64, :forward, UInt(2))
        data = [1.0 + 0.0im; 1.0 + 0.0im]
        RustFFT.fft!(instance, data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner32, :forward, UInt(1))
        data = [ComplexF32(1.0)]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner32, :forward, UInt(2))
        data = [ComplexF32(1.0); ComplexF32(1.0)]
        RustFFT.fft!(instance, data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end
end

@testset "Inverse FFT" begin
    planner64 = RustFFT.FftPlanner64()
    planner32 = RustFFT.FftPlanner32()

    @test begin
        instance = RustFFT.plan_fft_inverse(planner64, UInt(1))
        data = [1.0 + 0.0im]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_inverse(planner64, UInt(2))
        data = [1.0 + 0.0im; 0.0 + 0.0im]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_inverse(planner32, UInt(1))
        data = [ComplexF32(1.0)]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_inverse(planner32, UInt(2))
        data = [ComplexF32(1.0); ComplexF32(0.0)]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner64, :inverse, UInt(1))
        data = [1.0 + 0.0im]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner64, :inverse, UInt(2))
        data = [1.0 + 0.0im; 0.0 + 0.0im]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner32, :inverse, UInt(1))
        data = [ComplexF32(1.0)]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner32, :inverse, UInt(2))
        data = [ComplexF32(1.0); ComplexF32(0.0)]
        RustFFT.fft!(instance, data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end
end

@testset "Async FFT" begin
    planner64 = RustFFT.FftPlanner64()
    planner32 = RustFFT.FftPlanner32()

    @test begin
        instance = RustFFT.plan_fft_forward(planner64, UInt(1))
        data = [1.0 + 0.0im]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_forward(planner64, UInt(2))
        data = [1.0 + 0.0im; 1.0 + 0.0im]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end

    @test begin
        instance = RustFFT.plan_fft_forward(planner32, UInt(1))
        data = [ComplexF32(1.0)]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_forward(planner32, UInt(2))
        data = [ComplexF32(1.0); ComplexF32(1.0)]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner64, :forward, UInt(1))
        data = [1.0 + 0.0im]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner64, :forward, UInt(2))
        data = [1.0 + 0.0im; 1.0 + 0.0im]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner32, :forward, UInt(1))
        data = [ComplexF32(1.0)]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner32, :forward, UInt(2))
        data = [ComplexF32(1.0); ComplexF32(1.0)]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 2.0 && data[2] ≈ 0.0
    end

    @test begin
        instance = RustFFT.plan_fft_inverse(planner64, UInt(1))
        data = [1.0 + 0.0im]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_inverse(planner64, UInt(2))
        data = [1.0 + 0.0im; 0.0 + 0.0im]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_inverse(planner32, UInt(1))
        data = [ComplexF32(1.0)]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft_inverse(planner32, UInt(2))
        data = [ComplexF32(1.0); ComplexF32(0.0)]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner64, :inverse, UInt(1))
        data = [1.0 + 0.0im]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner64, :inverse, UInt(2))
        data = [1.0 + 0.0im; 0.0 + 0.0im]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner32, :inverse, UInt(1))
        data = [ComplexF32(1.0)]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0
    end

    @test begin
        instance = RustFFT.plan_fft(planner32, :inverse, UInt(2))
        data = [ComplexF32(1.0); ComplexF32(0.0)]
        RustFFT.fft_async!(instance, data)
        data[1] ≈ 1.0 && data[2] ≈ 1.0
    end
end

@testset "Exceptions" begin
    planner64 = RustFFT.FftPlanner64()

    @test_throws JlrsError begin
        instance = RustFFT.plan_fft_forward(planner64, UInt(2))
        data = [1.0 + 0.0im]
        RustFFT.fft!(instance, data)
    end

    @test_throws JlrsError begin
        RustFFT.plan_fft(planner64, :iversse, UInt(2))
    end
end