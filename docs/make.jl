push!(LOAD_PATH,"../src/")

using Documenter, RustFFT
makedocs(
    sitename="RustFFT",
    modules = [RustFFT]
)

deploydocs(
    repo = "github.com/Taaitaaiger/RustFFT.jl.git",
)
