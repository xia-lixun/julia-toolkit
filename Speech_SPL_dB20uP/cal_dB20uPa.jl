using WAV
using Plots
include("../audio-features.jl")





function extract_symbol_and_merge(x::Array{T,1}, s::Array{T,1}, rep::U) where {T <: AbstractFloat, U <: Integer}
    
    n = length(x) 
    m = length(s)
    
    ℝ = xcorr(s, x)                                
    #display(plot(ℝ))
    𝓡 = sort(ℝ[local_maxima(ℝ)], rev = true)
    isempty(𝓡) && ( error("no local maxima found!") )

    y = zeros(T, rep * m)
    peaks = zeros(Int64, rep)

    # find the anchor point
    ploc = find(z->z==𝓡[1],ℝ)[1]
    peaks[1] = ploc
    info("peak anchor-[1] in correlation: $ploc")
    lb = n - ploc + 1
    rb = lb + m - 1
    y[1:m] = x[lb:rb]
    ip = 1

    if rep > 1
        for i = 2:length(𝓡)
            ploc = find(z->z==𝓡[i],ℝ)[1]
            if sum(abs.(peaks[1:ip] - ploc) .> m) == ip
                ip += 1
                peaks[ip] = ploc
                info("peak anchor-[$ip] in correlation: $ploc")
                lb = n - ploc + 1
                rb = lb + m - 1
                #display(plot(r[lb:rb]))
                y[1+(ip-1)*m : ip*m] = x[lb:rb]
                if ip == rep
                    break
                end
            end
        end
        peaks = sort(peaks)
        #info("diff of peak locations: $(diff(peaks)./ρ.rate) seconds")
    end
    #display(plot(y))
    (y, diff(peaks))
end


  
function cal_dB20uPa( 
    
    piston_recording::AbstractString,
    unknown_recording::AbstractString,
    symbol::AbstractString,
    repeat::U, 
    symbol_l,
    symbol_h,
    p::Frame1D{U}; 
    
    fl = 100, 
    fh = 12000, 
    piston_dbspl = 114.0
    
    ) where {U <: Integer}

    # extracting calibration recording and measurement recording
    r, fs = wavread(piston_recording)
    assert(U(fs) == p.rate)

    rp = power_spectrum(r[:,1], p, p.block, window=hann)
    rp = mean(rp, 2)

    x, fs = wavread(unknown_recording)
    assert(U(fs) == p.rate)
    channels = size(x,2)
    info("file channels $channels")
    dbspl = zeros(typeof(x[1,1]), channels)

    s, fs =  wavread(symbol)
    assert(U(fs) == p.rate)
    s = mean(s,2)[:,1]

    # to use whole symbol, set symbol_l >= symbol_h
    if symbol_l < symbol_h
        assert(size(s,1) >= U(floor(p.rate * symbol_h))) 
        s = s[1 + U(floor(p.rate * symbol_l)) : U(floor(p.rate * symbol_h))]        
    end

    for c = 1:channels
        xp, interval = extract_symbol_and_merge(x[:,c], s, repeat)
        info("diff of peak locations: $(interval./p.rate) seconds")
        xp = power_spectrum(xp, p, p.block, window=hann)
        xp = mean(xp, 2)
        
        fl = U(floor(fl/p.rate * p.block))
        fh = U(floor(fh/p.rate * p.block))
        
        offset = 10*log10(sum_kbn(rp[fl:fh]))
        dbspl[c] = 10*log10(sum_kbn(xp[fl:fh])) + (piston_dbspl - offset)
        info("channel $c: SPL = $(dbspl[c]) dB")           
    end
    dbspl
end




function dBSPL_46AN( recording, symbol; 
    repeat = 3,
    symbol_start=0.25,
    symbol_stop=10.45,
    fl = 100, 
    fh = 12000, 
    piston_dbspl = 114.0
    )
    
    p = Frame1D{Int64}(48000, 16384, div(16384,4), 0)
    dBSPL = cal_dB20uPa(
        "1000hz-piston-114dBSPL-46AN.wav", 
        recording, 
        symbol, 
        repeat, 
        symbol_start, 
        symbol_stop, 
        p,
        fl = fl,
        fh = fh,
        piston_dbspl = piston_dbspl)    
end