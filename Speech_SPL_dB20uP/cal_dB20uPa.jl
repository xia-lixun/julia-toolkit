using WAV
using Plots
include("../audio-features.jl")





struct Source{T <: AbstractFloat, U <: Integer}
    rate::U
    path::AbstractString
    use_file_size::Bool
    start::T
    stop::T
    Source{T,U}(x,y,s,p,q) where {T <: AbstractFloat, U <: Integer} = p >= q ? error("start time must be less than stop time") : new(x,y,s,p,q)
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
    s = mean(s,2)[:,1]

    # if there is any boundary definitions
    if ρ.use_file_size == false
        assert(size(s,1) >= U(floor(ρ.rate * ρ.stop)))
        t0 = 1 + U(floor(ρ.rate * ρ.start)) 
        t1 = U(floor(ρ.rate * ρ.stop))
        s = s[t0:t1]        
    end
    m = length(s)

    
    ℝ = xcorr(s, r)
    #display(plot(ℝ))
    𝓡 = sort(ℝ[local_maxima(ℝ)], rev = true)
    
    y = zeros(typeof(r[1]), repeats * m)
    peaks = zeros(Int64, repeats)

    # find the anchor point
    ploc = find(x->x==𝓡[1],ℝ)[1]
    peaks[1] = ploc
    info("peak anchor-[1] in correlation: $ploc")
    lb = n - ploc + 1
    rb = lb + m - 1
    y[1:m] = r[lb:rb]
    ip = 1
    if repeats > 1
        for i = 2:length(𝓡)
            ploc = find(x->x==𝓡[i],ℝ)[1]
            if sum(abs.(peaks[1:ip] - ploc) .> m) == ip
                ip += 1
                peaks[ip] = ploc
                info("peak anchor-[$ip] in correlation: $ploc")
                lb = n - ploc + 1
                rb = lb + m - 1
                #display(plot(r[lb:rb]))
                y[1+(ip-1)*m : ip*m] = r[lb:rb]
                if ip == repeats
                    break
                end
            end
        end
        peaks = sort(peaks)
        info("diff of peak locations: $(diff(peaks)./ρ.rate) seconds")
    end
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
    rp = mean(rp, 2)

    x = extract_clip_and_merge(recording, repeats, ρ)
    xp = power_spectrum(x, p, p.block, window=hann)
    xp = mean(xp, 2)
    
    fl = U(floor(fl/p.rate * p.block))
    fh = U(floor(fh/p.rate * p.block))
    
    offset = 10*log10(sum_kbn(rp[fl:fh]))
    return 10*log10(sum_kbn(xp[fl:fh])) + (piston_dbspl - offset)
end

function cal_dB20uPa( calib_recording::AbstractString, p::Frame1D{U}, recording::AbstractString; tl = 5, tr = 15, fl = 100, fh = 12000, piston_dbspl = 114.0) where {U <: Integer}
    
        # extracting calibration recording and measurement recording
        r, fs = wavread(calib_recording)
        assert(U(fs) == p.rate)
        rp = power_spectrum(r[:,1], p, p.block, window=hann)
        rp = mean(rp, 2)
    
        x, fs = wavread(recording)
        assert(U(fs) == p.rate)
        channels = size(x,2)
        dBSPL = zeros(typeof(x[1,1]), channels)

        start = 1+U(floor(tl * p.rate))
        stop = U(floor(tr * p.rate))

        for c = 1:channels
            xp = power_spectrum(x[start:stop,c], p, p.block, window=hann)
            xp = mean(xp, 2)
            fl = U(floor(fl/p.rate * p.block))
            fh = U(floor(fh/p.rate * p.block))
            offset = 10*log10(sum_kbn(rp[fl:fh]))
            dBSPL[c] = 10*log10(sum_kbn(xp[fl:fh])) + (piston_dbspl - offset)    
        end
        dBSPL 
    end







function dBSPL_46AN(recording::AbstractString, symbol::AbstractString, repeats; use_symbol_size=true, start=0.25, stop=10.45)
    
    p = Frame1D{Int64}(48000, 16384, div(16384,4), 0)
    ρ = Source{Float64, Int64}(48000, symbol, use_symbol_size, start, stop)
    dBSPL = cal_dB20uPa("1000hz-piston-114dBSPL-46AN.wav", p, recording, repeats, ρ)
    println("SPL = $dBSPL dB")       
end

function dBSPL_40AN(recording::AbstractString, symbol::AbstractString, repeats; use_symbol_size=true, start=0.25, stop=10.45)
    
    p = Frame1D{Int64}(48000, 16384, div(16384,4), 0)
    ρ = Source{Float64, Int64}(48000, symbol, use_symbol_size, start, stop)
    dBSPL = cal_dB20uPa("1000hz-piston-114dBSPL-40AN.wav", p, recording, repeats, ρ)
    println("SPL = $dBSPL dB")       
end

function dBSPL_46AN(recording::AbstractString; start=5, stop=15)
    
    p = Frame1D{Int64}(48000, 16384, div(16384,4), 0)
    dBSPL = cal_dB20uPa("1000hz-piston-114dBSPL-46AN.wav", p, recording, tl = start, tr = stop)
    println("SPL = $dBSPL dB")       
end