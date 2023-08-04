module Internal
using JlrsCore.Wrap
import rustfft_jll: librustfft_path

export FftInstance, FftPlanner, FftPlanner32, FftPlanner64, rustfft_plan_fft_forward!,
    rustfft_plan_fft_forward_untracked!, rustfft_plan_fft_forward!,
    rustfft_plan_fft_inverse_untracked!, rustfft_plan_direction, rustfft_plan_size,
    rustfft_fft!, rustfft_fft_gcsafe!,
    rustfft_fft_untracked!, rustfft_fft_untracked_gcsafe!,
    rustfft_fft_unchecked!, rustfft_fft_unchecked_gcsafe!

@wrapmodule(librustfft_path, :rustfft_jl_init)

function __init__()
    @initjlrs
end
end