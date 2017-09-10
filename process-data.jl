using WAV
using DataFrames
using HDF5


# iterate over the dataset tree and for each .wav/.ogg/.mp3/.aac/.ac3 etc. do:
# [1] cp a.wav /tmp/a.wav || ffmpeg -i a.!wav /tmp/a.wav
# [2] sox /tmp/a.wav -r 16000 /tmp/a-16000.wav
# [3] x,fs=wavread("/tmp/a-16000.wav")
#     parse("a-16000.wav") for classid info/folder info
#     x = stand(x)
#     fbe = filter_bank_energies(x)
# [4] h5write(fbe) to test/train folders

include("./audio-features.jl")
x,fs = wavread("standard-16000.wav")
println(fs)

p = Frame1D{Int64}(fs,floor(0.025*fs),floor(0.01*fs),0)
fbe = filter_bank_energy(x[:,1], p, 512, zero_append=true, fl=100, fh=6800)

writetable("./julia-output.csv",DataFrame(fbe))