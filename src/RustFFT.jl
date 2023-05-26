module RustFFT
using JlrsCore
if !isdefined(Main, :JlrsCore)
    ccall(:jl_set_global, Cvoid, (Any, Any, Any), Main, :JlrsCore, JlrsCore)
end

import rustfft_jll: librustfft_path
using JlrsCore.Wrap

@wrapmodule(librustfft_path, :rustfft_jl_init)

function __init__()
    @initjlrs
end
end
