using WAV
using Plots
include("../audio-features.jl")





function extract_symbol_and_merge(x::Array{T,1}, s::Array{T,1}, rep::U) where {T <: AbstractFloat, U <: Integer}
    
    n = length(x) 
    m = length(s)
    y = zeros(T, rep * m)
    peaks = zeros(Int64, rep)

    ℝ = xcorr(s, x)                                
    #display(plot(ℝ))
    𝓡 = sort(ℝ[local_maxima(ℝ)], rev = true)
    isempty(𝓡) && ( return (y, diff(peaks)) )


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



# r : referece calibrator recording
# x : multi-track to be measured
# s : symbol signal
# return dbspl of all channels in x
function cal_dB20uPa( r::Array{T,1}, x::Array{T,2}, s::Array{T,1}, repeat::U, symbol_l, symbol_h, p::Frame1D{U}; 

    fl = 100, 
    fh = 12000, 
    piston_dbspl = 114.0

    ) where {T <: AbstractFloat, U <: Integer}

    # calibration
    rp = power_spectrum(r, p, p.block, window=hann)
    rp = mean(rp, 2)

    # recording
    channels = size(x,2)
    info("file channels $channels")
    dbspl = zeros(typeof(x[1,1]), channels)


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
        
    # calibration 
    r, fs = wavread("Calibration\\46an_1000hz_114dbspl_201709191531.wav")
    assert(Int64(fs) == p.rate)
    
    # recording
    x, fs = wavread(recording)
    assert(Int64(fs) == p.rate)
    
    # symbol
    s, fs =  wavread(symbol)
    assert(Int64(fs) == p.rate)
    s = mean(s,2)[:,1]


    dbspl = cal_dB20uPa(r[:,1], x, s, repeat, symbol_start, symbol_stop, p,
        fl = fl,
        fh = fh,
        piston_dbspl = piston_dbspl)    
end





function symbol_group()
    
    hz = [71, 90, 112, 141, 179, 224, 280, 355, 450, 560, 710, 900, 1120, 1410, 1790, 2240, 2800, 3550, 4500]
    t = 3
    fs = 48000
      
    m = Int64(floor(t*fs))
    n = length(hz)
    y = ones(m, n)
    for i = 1:n
        y[:,i] = sin.(2*π*hz[i]/fs*(0:m-1))
    end
    (y, fs)    
end


# single file multiple symbols
function dBSPL_46AN_SG( recording, sg; 
    repeat = 1,
    symbol_start=0,
    symbol_stop=0,
    fl = 100, 
    fh = 12000, 
    piston_dbspl = 114.0
    )
    
    p = Frame1D{Int64}(48000, 16384, div(16384,4), 0)
    
    # calibration 
    r, fs = wavread("Calibration\\46an_1000hz_114dbspl_201709191531.wav")
    assert(Int64(fs) == p.rate)
        
    # recording
    x, fs = wavread(recording)
    assert(Int64(fs) == p.rate)
    
    # symbols
    s, fs = sg()
    assert(Int64(fs) == p.rate)

    y = zeros(size(x,2), size(s,2))
    for i = 1:size(s,2)
        dbspl = cal_dB20uPa(r[:,1], x, s[:,i], repeat, symbol_start, symbol_stop, p,
            fl = fl,
            fh = fh,
            piston_dbspl = piston_dbspl)
        y[:,i] = dbspl
    end
    y
end