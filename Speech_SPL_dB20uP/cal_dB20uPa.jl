using WAV
using Plots
include("../audio-features.jl")





struct Source{T <: Real, U <: Integer}
    rate::U
    path::AbstractString
    start::T
    finish::T
    Source{T,U}(x,y,p,q) where {T <: Real, U <: Integer} = p >= q ? error("start time must be less than finishing time") : new(x,y,p,q)
end




function local_maxima(x::Array{T,1}) where {T <: Real}
    
    gtl = [false; x[2:end] .> x[1:end-1]]
    gtu = [x[1:end-1] .>= x[2:end]; false]
    imax = gtl .& gtu
end


# s = Source{Int64,Float64}(target_fs, "PreparationRCV_100_8000_BPF.wav", 0.35, 10.425)
function extract_clip_and_merge(recording, src::Source{T,U}, repeats) where {T <: Real, U <: Integer}
    
    # data preparation
    r, fs = wavread(recording)
    assert(Int64(fs) == src.rate)
    r = r[:,1]

    s, fs =  wavread(src.path)
    assert(Int64(fs) == src.rate)
    assert(size(s,1) >= Int64(floor(src.rate * src.finish)))
    s = mean(s,2)[:,1]

    # if there is any boundary definitions
    t0 = 1 + U(round(src.rate * src.start))
    t1 = U(round(src.rate * src.finish))
    s = s[t0:t1]

    rxx = xcorr(s, r)
    #display(plot(rxx))
    maxima_sorted = sort(rxx[local_maxima(rxx)], rev = true)

    sig = Array{typeof(r[1]),1}()
    peaks = Array{Int64,1}()
    for i = 1:repeats

        peak = find(x->x==maxima_sorted[i],rxx)[1]
        peaks = [peaks; peak]
        info("peak location in correlation: $peak")
        lb = length(r) - peak + 1
        rb = lb + length(s) - 1
        #display(plot(r[lb:rb]))
        sig = [sig; r[lb:rb]]
    end
    peaks = sort(peaks)
    info("diff of peak locations: $(diff(peaks)./src.rate) seconds")
    #display(plot(sig))
    sig
end


# target_fs = 48000
# block = 16384
# p = Frame1D{Int64}(target_fs, block, div(block,4), 0)
# s = Source{Int64,Float64}(target_fs, "PreparationRCV_100_8000_BPF.wav", 0.35, 10.425)
function cal_dB20uPa(calib_recording, p::Frame1D{U}, recording, src::Source{T,U}, repeats; fl = 100, fh = 12000) where {T <: Real, U <: Integer}

    # extracting calibration recording and measurement recording
    ref, fs = wavread(calib_recording)
    assert(Int64(fs) == p.rate)
    sig = extract_clip_and_merge(recording, src, repeats)

    win = hann(Float64, p.block)
    
    sig_p = get_frames(sig, p)
    sig_p = sig_p .* repmat(win,1,size(sig_p,2))
    sig_p = (abs.(fft(sig_p,1))).^2
    sig_p = mean(sig_p[1:div(p.block,2)+1,:],2)
    
    ref_p = get_frames(ref[:,1], p)
    ref_p = ref_p .* repmat(win,1,size(ref_p,2))
    ref_p = (abs.(fft(ref_p,1))).^2
    ref_p = mean(ref_p[1:div(p.block,2)+1,:],2)
    
    
    #xl = 0:length(sig_p)-1;
    #figure; plot(xl/blk*fs,sig_p); grid on;
    #figure; plot(xl/blk*fs,ref_p); grid on;
    
    fl = U(floor(fl/p.rate * p.block))
    fh = U(floor(fh/p.rate * p.block))
    
    offset = 10*log10(sum_kbn(ref_p[fl:fh]))
    return 10*log10(sum_kbn(sig_p[fl:fh])) + (114 - offset)
end


function cal_(recording::AbstractString)
    target_fs = 48000
    calib_recording = "1000hz-piston-114dBSPL-46AN.wav"
    
    #recording = "Vol70-CotRcvEq-AplayBPF-46AN.wav"
    recording_repeat = 3

    source = "PreparationRCV_100_8000_BPF.wav"
    source_start = 0.25
    source_end = 10.425

    block = 16384
    update = div(block,4)
    p = Frame1D{Int64}(target_fs, block, update, 0)
    s = Source{Float64, Int64}(target_fs, source, 0.35, 10.425)

    return cal_dB20uPa(calib_recording, p, recording, s, recording_repeat)        
end