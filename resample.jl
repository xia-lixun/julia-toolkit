#!/full/path/to/julia/exe

#simple sample rate converter based on FIR filters generated from resample() function of Matlab
function resample(data::Array{Float32,1}, fs_new::Float32, fs_old::Float32, filter::Array{Float32,1})

    const FILTER_LEN::Int64 = length(filter)

    #determine the interpolate/decimate rates
    fs_old_int = convert(Int64, fs_old)
    fs_new_int = convert(Int64, fs_new)

    g = gcd(fs_new_int, fs_old_int)
    interpolate = div(fs_new_int, g)
    decimate = div(fs_old_int, g)

    #allocate target data buffer
    result = zeros(Float32, div(length(data) * interpolate, decimate) + 1)
    
    #apply filtering to the original data and store the result to target data buffer
    count::Int64 = 1
    offset::Int64 = 0
    const interpolated::Int64 = length(data) * interpolate
    nn::Int64 = -1
    dp::Float32 = -1.f0

    while offset < interpolated

        nn = div(offset, interpolate)
        dp = 0.f0
        for i = nn:-1:0
            dp += (offset - i * interpolate < FILTER_LEN) ? data[i+1] * filter[offset - i * interpolate + 1] : 0.f0
        end
        result[count] = dp
        count += 1
        offset += decimate

    end
    return result        
end
