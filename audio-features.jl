


# n_update  (shift samples)
# n_overlap (overlapping samples)
# n_block   (block size)

function hamming(T, n)
    ω = Array{T,1}(n)
    α = T(0.54)
    β = 1 - α
    for i = 0:n-1
        ω[i+1] = α - β * T(cos(2π * i / (n-1)))
    end
    ω
end




# immutable type definition
# note that BlockProcessing{Int16}(1024.0, 256.0, 0) is perfectly legal as new() will convert every parameter to T
# but BlockProcessing{Int16}(1024.0, 256.3, 0) would not work as it raises InexactError()
# also note that there is not white space between BlockProcessing and {T <: Integer}
struct Frame1D{T <: Integer}
    rate::T
    block::T
    update::T
    overlap::T
    Frame1D{T}(r, x, y, z) where {T <: Integer} = x < y ? error("block size must ≥ update size!") : new(r, x, y, x-y)
end
# we define an outer constructor as the inner constructor infers the overlap parameter
# again the block_len and update_len accepts Integers as well as AbstructFloat with no fractions
#
# example type outer constructors: 
# FrameInSample(fs, block, update) = Frame1D{Int64}(fs, block, update, 0)
# FrameInSecond(fs, block, update) = Frame1D{Int64}(fs, floor(block * fs), floor(update * fs), 0)




function tile(x::Array{T,1}, p::Frame1D{U}; zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    
    zero_init && (x = [zeros(T, p.overlap); x])                                     # zero padding to the front for defined init state
    length(x) < p.block && error("signal length must be at least one block!")       # detect if length of x is less than block size
    n = div(size(x,1) - p.block, p.update) + 1                                      # total number of frames to be processed
    
    if zero_append
        m = rem(size(x,1) - p.block, p.update)
        if m != 0
            x = [x; zeros(T, p.update-m)]
            n += 1
        end
    end
    (x,n)
end



# function    : get_frames
# x           : array of AbstractFloat {Float64, Float32, Float16, BigFloat}
# p           : frame size immutable struct
# zero_init   : simulate the case when block buffer is init to zero and the first update comes in
# zero_append : simulate the case when remaining samples of x doesn't make up an update length
# 
# example:
# x = collect(1.0:100.0)
# p = Frame1D{Int64}(8000, 17, 7.0, 0)
# y = get_frames(x, p)
function get_frames(x::Array{T,1}, p::Frame1D{U}; window=ones, zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    
    x, n = tile(x, p, zero_init = zero_init, zero_append = zero_append)
    
    ω = window(T, p.block)
    y = zeros(T, p.block, n)
    h = 0
    for i = 1:n
        y[:,i] = ω .* x[h+1:h+p.block]
        h += p.update
    end
    y
end

# example:
# x = collect(1.0:100.0)
# p = Frame1D{Int64}(8000, 17, 7.0, 0)
# y = spectrogram(x, p, nfft, window=hamming, zero_init=true, zero_append=true)
function spectrogram(x::Array{T,1}, p::Frame1D{U}, nfft::U; window=ones, zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    
    nfft < p.block && error("nfft length must be greater than or equal to block/frame length")
    x, n = tile(x, p, zero_init = zero_init, zero_append = zero_append)

    ω = window(T, nfft)
    P = plan_fft(ω)
    𝕏 = zeros(Complex{T}, nfft, n)
    h = 0
    for i = 1:n
        𝕏[:,i] = P * ( ω .* [x[h+1:h+p.block]; zeros(T,nfft-p.block)] )
        h += p.update
    end
    𝕏
end


# v: indicates vector <: AbstractFloat
energy(v) = x.^2
intensity(v) = abs.(v)
zero_crossing_rate(v) = floor.((abs.(diff(sign.(v)))) ./ 2)

function short_term(f, x::Array{T,1}, p::Frame1D{U}; zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    frames = get_frames(x, p, zero_init=zero_init, zero_append=zero_append)
    n = size(frames,2)
    ste = zeros(T, n)
    for i = 1:n
        ste[i] = sum_kbn(f(frames[:,i])) 
    end
    ste
end

pp_norm(v) = (v - minimum(v)) ./ (maximum(v) - minimum(v))
stand(v) = (v - mean(v)) ./ std(v)
hz_to_mel(hz) = 2595 * log10.(1 + hz * 1.0 / 700)
mel_to_hz(mel) = 700 * (10 .^ (mel * 1.0 / 2595) - 1)


# calculate power spectrum of 1-D array on a frame basis
# note that T=Float16 may not be well supported by FFTW backend
function power_spectrum(x::Array{T,1}, p::Frame1D{U}, nfft::U; window=ones, zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    
    nfft < p.block && error("nfft length must be greater than or equal to block/frame length")
    x, n = tile(x, p, zero_init = zero_init, zero_append = zero_append)

    ω = window(T, nfft)
    f = plan_fft(ω)
    m = div(nfft,2)+1
    ℙ = zeros(T, m, n)
    ρ = T(1 / nfft)

    h = 0
    for i = 1:n
        ξ = f * (ω .* [x[h+1:h+p.block]; zeros(T,nfft-p.block)]) # typeof(ξ) == Array{Complex{T},1} 
        ℙ[:,i] = ρ * ((abs.(ξ[1:m])).^2)
        h += p.update
    end
    ℙ
end

# calculate filter banks
function filter_banks(T, rate::U, nfft::U; filt_num=26, fl=0, fh=div(rate,2)) where {U <: Integer}
    
    fh > div(rate,2) && error("high frequency must be less than or equal to nyquist frequency!")
    
    ml = hz_to_mel(fl)
    mh = hz_to_mel(fh)
    mel_points = linspace(ml, mh, filt_num+2)
    hz_points = mel_to_hz(mel_points)

    # round frequencies to nearest fft bins
    𝕓 = U.(floor.((hz_points/rate) * (nfft+1)))
    #print(𝕓)

    # first filterbank will start at the first point, reach its peak at the second point
    # then return to zero at the 3rd point. The second filterbank will start at the 2nd
    # point, reach its max at the 3rd, then be zero at the 4th etc.
    𝔽 = zeros(T, filt_num, div(nfft,2)+1)

    for i = 1:filt_num
        for j = 𝕓[i]:𝕓[i+1]
            𝔽[i,j+1] = T((j - 𝕓[i]) / (𝕓[i+1] - 𝕓[i]))
        end
        for j = 𝕓[i+1]:𝕓[i+2]
            𝔽[i,j+1] = T((𝕓[i+2] - j) / (𝕓[i+2] - 𝕓[i+1]))
        end
    end
    𝔽
end


function filter_bank_energy(x::Array{T,1}, p::Frame1D{U}, nfft::U; window=ones, zero_init=false, zero_append=false, filt_num=26, fl=0, fh=div(p.rate,2), use_log=false) where {T <: AbstractFloat, U <: Integer}

    ℙ = power_spectrum(x, p, nfft, window=window, zero_init=zero_init, zero_append=zero_append)
    𝔽 = filter_banks(T, p.rate, nfft, filt_num=filt_num, fl=fl, fh=fh)
    ℙ = 𝔽 * ℙ
    use_log && (log.(ℙ + eps()))
    ℙ
end