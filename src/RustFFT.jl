module RustFFT
using JlrsCore
import rustfft_jll: librustfft_path
using JlrsCore.Wrap

@wrapmodule(librustfft_path, :rustfft_jl_init)

function __init__()
    @initjlrs
end
end
