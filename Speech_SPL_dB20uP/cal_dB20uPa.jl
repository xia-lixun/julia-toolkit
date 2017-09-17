using WAV
using Plots
include("../audio-features.jl")





struct Source{T <: AbstractFloat, U <: Integer}
    rate::U
    path::AbstractString
    start::T
    finish::T
    Source{T,U}(x,y,p,q) where {T <: AbstractFloat, U <: Integer} = p >= q ? error("start time must be less than finishing time") : new(x,y,p,q)
end




# provide source information for speech extraction from the recordings
# s = Source{Int64,Float64}(target_fs, "PreparationRCV_100_8000_BPF.wav", 0.35, 10.425)
function extract_clip_and_merge(recording::AbstractString, repeats::U, ρ::Source{T,U}) where {T <: AbstractFloat, U <: Integer}
    
    # data preparation
    r, fs = wavread(recording)
    assert(U(fs) == ρ.rate)
    r = r[:,1]
    n = length(r) 

    s, fs =  wavread(ρ.path)
    assert(U(fs) == ρ.rate)
    assert(size(s,1) >= U(floor(ρ.rate * ρ.finish)))
    s = mean(s,2)[:,1]

    # if there is any boundary definitions
    t0 = 1 + U(round(ρ.rate * ρ.start))
    t1 = U(round(ρ.rate * ρ.finish))
    s = s[t0:t1]
    m = length(s)

    
    ℝ = xcorr(s, r)
    #display(plot(ℝ))
    𝓡 = sort(ℝ[local_maxima(ℝ)], rev = true)
    
    y = zeros(typeof(r[1]), repeats * m)
    peaks = zeros(Int64, repeats)

    for i = 1:repeats

        ploc = find(x->x==𝓡[i],ℝ)[1]
        peaks[i] = ploc
        info("peak location in correlation: $ploc")
        lb = n - ploc + 1
        rb = lb + m - 1
        #display(plot(r[lb:rb]))
        y[1+(i-1)*m:i*m] = r[lb:rb]
    end
    peaks = sort(peaks)
    info("diff of peak locations: $(diff(peaks)./ρ.rate) seconds")
    #display(plot(y))
    y
end



# p = Frame1D{Int64}(target_fs, block, div(block,4), 0)
# s = Source{Int64,Float64}(target_fs, "PreparationRCV_100_8000_BPF.wav", 0.35, 10.425)
function cal_dB20uPa( calib_recording::AbstractString, p::Frame1D{U}, recording::AbstractString, repeats::U, ρ::Source{T,U}; fl = 100, fh = 12000, piston_dbspl = 114.0) where {T <: AbstractFloat, U <: Integer}

    # extracting calibration recording and measurement recording
    r, fs = wavread(calib_recording)
    assert(U(fs) == p.rate)
    assert(p.rate == ρ.rate)
    rp = power_spectrum(r[:,1], p, p.block, window=hann)

    x = extract_clip_and_merge(recording, repeats, ρ)
    xp = power_spectrum(x, p, p.block, window=hann)
    
    fl = U(floor(fl/p.rate * p.block))
    fh = U(floor(fh/p.rate * p.block))
    
    offset = 10*log10(sum_kbn(rp[fl:fh]))
    return 10*log10(sum_kbn(xp[fl:fh])) + (piston_dbspl - offset)
end






function cal_(recording::AbstractString)
    
    #recording = "Vol70-CotRcvEq-AplayBPF-46AN.wav"

    p = Frame1D{Int64}(48000, 16384, div(16384,4), 0)
    ρ = Source{Float64, Int64}(48000, "PreparationRCV_100_8000_BPF.wav", 0.25, 10.425)
    dBSPL = cal_dB20uPa("1000hz-piston-114dBSPL-46AN.wav", p, recording, 3, ρ)
    println("SPL = $dBSPL dB")       
end