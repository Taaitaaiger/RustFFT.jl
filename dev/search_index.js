var documenterSearchIndex = {"docs":
[{"location":"#RustFFT-Documentation","page":"RustFFT Documentation","title":"RustFFT Documentation","text":"","category":"section"},{"location":"","page":"RustFFT Documentation","title":"RustFFT Documentation","text":"Modules = [RustFFT]","category":"page"},{"location":"#RustFFT.FftInstance32","page":"RustFFT Documentation","title":"RustFFT.FftInstance32","text":"FftInstance32\n\nA planned FFT instance that can compute either forward or inverse FFTs of Vector{Complex{Float32}} data whose length is an integer multiple of the planned length.\n\n\n\n\n\n","category":"type"},{"location":"#RustFFT.FftInstance64","page":"RustFFT Documentation","title":"RustFFT.FftInstance64","text":"FftInstance64\n\nA planned FFT instance that can compute either forward or inverse FFTs of  Vector{Complex{Float64}} data whose length is an integer multiple of the planned length.\n\n\n\n\n\n","category":"type"},{"location":"#RustFFT.FftPlanner32","page":"RustFFT Documentation","title":"RustFFT.FftPlanner32","text":"FftPlanner32\n\nA planner for forward and inverse FFTs of Vector{Complex{Float32}} data. A new planner can be created by calling the zero-argument constructor.\n\n\n\n\n\n","category":"type"},{"location":"#RustFFT.FftPlanner64","page":"RustFFT Documentation","title":"RustFFT.FftPlanner64","text":"FftPlanner64\n\nA planner for forward and inverse FFTs of Vector{Complex{Float64}} data. A new planner can be created by calling the zero-argument constructor.\n\n\n\n\n\n","category":"type"},{"location":"#RustFFT.fft!","page":"RustFFT Documentation","title":"RustFFT.fft!","text":"fft!(instance, data)\n\nComputes the forward or inverse FFT of the data in-place. instance must be either a FftInstance32 or a FftInstance64. data must be either a Vector{Complex{Float32}} or a Vector{Complex{Float64}}, the width must match the that of the provided instance, its length must be an integer multiple the length of the instance.\n\n\n\n\n\n","category":"function"},{"location":"#RustFFT.fft_async!","page":"RustFFT Documentation","title":"RustFFT.fft_async!","text":"fft_async!(instance, data)\n\nComputes the forward or inverse FFT of the data in-place. The transform is computed by a background thread, this function waits for that computation to be finished before returning. See RustFFT.fft! for more info.\n\n\n\n\n\n","category":"function"},{"location":"#RustFFT.plan_fft","page":"RustFFT Documentation","title":"RustFFT.plan_fft","text":"plan_fft(planner, direction::Symbol, len::UInt)\n\nPlan either a forward or an inverse FFT of length len. Returns either an FftInstance32 or FftInstance64 depending on the provided planner, which must be either an FftPlanner32 or FftPlanner64. The direction must be either :forward or :inverse\n\n\n\n\n\n","category":"function"},{"location":"#RustFFT.plan_fft_forward","page":"RustFFT Documentation","title":"RustFFT.plan_fft_forward","text":"plan_fft_forward(planner, len::UInt)\n\nPlan a forward FFT of length len. Returns either an FftInstance32 or FftInstance64 depending on the provided planner, which must be either an FftPlanner32 or  FftPlanner64.\n\n\n\n\n\n","category":"function"},{"location":"#RustFFT.plan_fft_inverse","page":"RustFFT Documentation","title":"RustFFT.plan_fft_inverse","text":"plan_fft_inverse(planner, len::UInt)\n\nPlan an inverse FFT of length len. Returns either an FftInstance32 or FftInstance64 depending on the provided planner, which must be either an FftPlanner32 or  FftPlanner64.\n\n\n\n\n\n","category":"function"}]
}
