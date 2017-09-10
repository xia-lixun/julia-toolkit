using WAV
using DataFrames

include("./audio-features.jl")
x,fs = wavread("standard-16000.wav")
println(fs)

p = Frame1D{Int64}(fs,floor(0.025*fs),floor(0.01*fs),0)
fbe = filter_bank_energy(x[:,1], p, 512, zero_append=true, fl=100, fh=6800)

writetable("./julia-output.csv",DataFrame(fbe))