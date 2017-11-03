# management of wav files for machine learning projects
# lixun.xia@outlook.com
# 2017-10-16
using SHA
using WAV
using JSON
using HDF5
include("audio-features.jl")




#filst(path) will list all busfolders under path, without root!
#flist(path, t=".wav") will list all wav files under path and uid 
function flist(path; t="")

    x = Array{String,1}()
    for (root, dirs, files) in walkdir(path)
        for dir in dirs
            isempty(t) && push!(x, dir)
        end
        for file in files
            !isempty(t) && lowercase(file[end-length(t)+1:end]) == lowercase(t) && push!(x, joinpath(root, file))
        end
    end
    x
end
function fsha256(list)

    d = zeros(UInt8,32)
    n = length(list)
    pz = -1
    for (i,j) in enumerate(list)
        d += open(j) do f
            sha256(f)
            end
        p = Int64(round((i/n)*100))
        if in(p, 0:10:100) && (p != pz)
            pz = p
            println("%$p")
        end
    end
    d
end

# api
function updatesha256(path)

    p = joinpath(path,"index.sha256")
    writedlm(p, fsha256(flist(path, t=".wav")))
    info("checksum updated in $p")
end
function checksha256(path)

    p = readdlm(joinpath(path,"index.sha256"), UInt8)[:,1]
    q = fsha256(flist(path, t="wav"))
    ok = (0x0 == sum(p-q))
end




# 1. walk through path_in folder for all wav files recursively
# 2. convert to target fs
# 3. put result to path_out foler linearly
function resample(path_in, path_out, target_fs)

    a = flist(path_in, t=".wav")
    n = length(a)
    u = Array{Int64,1}(n)

    for (i,j) in enumerate(a)
        run(`ffmpeg -y -i $j D:\\temp.wav`)
        p = joinpath(path_out, relpath(dirname(j), path_in))
        mkpath(p)
        p = joinpath(p, basename(j))
        run(`sox D:\\temp.wav -r 16000 $p`)
            
        x,fs = wavread(p)
        assert(fs == 16000.0f0)
        u[i] = size(x,1)
        
        info("$i/$n complete")
    end
    info("max: $(maximum(u)/16000) seconds")
    info("min: $(minimum(u)/16000) seconds")
    rm("D:\\temp.wav", force=true)
end



#generate template JSON file based on folder contents
function specgen()
    
    x = Array{Dict{String,String},1}()
    a = Dict(
        "samplerate" => "16000",
        "samplespace"=>"17",
        "mixsnr" => ["20", "15", "10", "5", "0", "-5"],
        "speechlevel" => ["-22", "-32", "-42"],
        "mixbase" => "speech",
        "mixrange" => ["0.1", "0.6"],
        "mixoutput" => "D:\\mix-utility\\mixed",
        "speech_rootpath" => "D:\\VoiceBox\\TIMIT-16k\\train",
        "noise_rootpath" => "D:\\NoiseBox\\104Nonspeech-16k",
        "build_speechlevel_index" => "true",
        "build_noiselevel_index" => "true",
        "noise_categories" => x,

        "feature" => Dict("frame_size"=>"512","step_size"=>"128","window_function"=>"Hamming","frame_radius"=>"6","frame_noisest"=>"7")
        )
    for i in flist(a["noise_rootpath"])
        push!(x, Dict("name"=>i,"type"=>"stationary|nonstationary|impulsive","percent"=>"0.0"))
    end

    #rm(a["mixoutput"], force=true, recursive=true)
    !isdir(a["mixoutput"]) && mkpath(a["mixoutput"])
    open(joinpath(a["mixoutput"],"specification-$(replace(replace("$(now())",":","-"),".","-")).json"),"w") do f
        write(f, JSON.json(a))
    end
end




rms(x,dim) = sqrt.(sum((x.-mean(x,dim)).^2,dim)/size(x,dim))
rms(x) = sqrt(sum((x-mean(x)).^2)/length(x))

# calculate index for noise samples:
# peak level: (a) as level of impulsive sounds (b) avoid level clipping in mixing
# rms level: level of stationary sounds
# median level: level of non-stationary sounds
#
# note: wav read errors will be skipped but warning pops up
#       corresponding features remain zero
function build_level_index(path, rate)

    a = flist(path, t=".wav")
    n = length(a)
    lpeak = zeros(n)
    lrms = zeros(n)
    lmedian = zeros(n)
    len = zeros(Int64, n)
    
    #uid = Array{String,1}(n)
    m = length(path)

    # wav must be monochannel and fs==16000
    pz = -1
    for (i,j) in enumerate(a)
        try
            x, fs = wavread(j)
            assert(Int64(fs) == Int64(rate))
            assert(size(x,2) == 1)
            x = x[:,1]
            lpeak[i] = maximum(abs.(x))
            lrms[i] = rms(x)
            lmedian[i] = median(abs.(x))
            len[i] = length(x)
            #uid[i] = replace(j[m+1:end-length(".wav")], "/", "+")
        catch
            warn(j)
        end

        p = Int64(round((i/n)*100))
        if in(p, 0:10:100) && (p != pz)
            pz = p
            println("%$p")
        end
    end

    # save level index to csv
    index = joinpath(path, "index.level")
    writedlm(index, [a lpeak lrms lmedian len], ',')
    info("index build to $index")
end



# mix procedure implements the specification
function mix(specification)

    # read the specification for mixing task
    s = JSON.parsefile(specification)          
    fs = parse(Int64,s["samplerate"])          #16000
    n = parse(Int64,s["samplespace"])          #17
    snr = parse.(Float64,s["mixsnr"])          #[-20.0, -10.0, 0.0, 10.0, 20.0]
    spl = parse.(Float64,s["speechlevel"])     #[-22.0, -32.0, -42.0]

    mr = parse.(Float64,s["mixrange"])         #[0.1, 0.6]
    mo = s["mixoutput"]
    clean = flist(mo, t=".wav")
    for i in clean
        rm(i, force=true)
    end

    #Part I. Noise treatment
    !isdir(s["noise_rootpath"]) && error("noise root path doesn't exist")
    if parse(Bool, s["build_noiselevel_index"])
        for i in s["noise_categories"]
            build_level_index(joinpath(s["noise_rootpath"],i["name"]), fs)
            updatesha256(joinpath(s["noise_rootpath"],i["name"]))
        end
    else
        for i in s["noise_categories"]
            !checksha256(joinpath(s["noise_rootpath"],i["name"])) && error("checksum $(i["name"])")
            info("noise checksum ok")
        end
    end
    #index format: path-to-wav-file, peak-level, rms-level, median, length-in-samples
    ni = Dict( x["name"] => readdlm(joinpath(s["noise_rootpath"], x["name"],"index.level"), ',') for x in s["noise_categories"])


    #Part II. Speech treatment
    !isdir(s["speech_rootpath"]) && error("speech root path doesn't exist")
    if parse(Bool, s["build_speechlevel_index"])
        #build_level_index(s["speech_rootpath"])
        #assume index ready by Matlab:activlevg() and provided as csv in format: path-to-wavfile, speech-level, length-in-samples
        updatesha256(s["speech_rootpath"])
    else
        !checksha256(s["speech_rootpath"]) && error("speech data checksum error")
        info("speech checksum ok")
    end
    #index format: path-to-wav-file, peak-level, speech-level(dB), length-in-samples
    si = readdlm(joinpath(s["speech_rootpath"],"index.level"), ',', header=false, skipstart=3)
    #(si, ni)


    # Part III. Mixing them up
    label = Dict{String, Array{Tuple{Int64, Int64},1}}()
    srand(1234)
    for i in s["noise_categories"]
        for j = 1:Int64(round(parse(Float64, i["percent"]) * 0.01 * n)) # items in each noise category
            
            # 3.0 preparation of randomness
            sp = spl[rand(1:length(spl))]
            sn = snr[rand(1:length(snr))]
            rs = rand(1:size(si,1))
            rn = rand(1:size(ni[i["name"]],1))

            # 3.1: random speech, in x[:,1]
            x = Array{Float64,1}()
            try
                x = wavread(si[rs,1])[1][:,1]
            catch
                #todo: test if my wav class could handle the corner cases of WAV.jl
                #todo: wrap my wav class with c routines to wav.dll, then wrap with julia
                warn("missing $(si[rs,1])")
            end
            g = 10^((sp-si[rs,3])/20)
            g * si[rs,2] > 1 && (g = 1 / si[rs,2]; info("relax gain to avoid clipping $(si[rs,1]):$(si[rs,3])->$(sp)(dB)"))
            x *= g #level speech to target level
            
            # 3.2: random noise
            u = Array{Float64,1}()
            try
                u = wavread(ni[i["name"]][rn,1])[1][:,1]
            catch
                #todo: test if my wav class could handle the corner cases of WAV.jl
                #todo: wrap my wav class with c routines to wav.dll, then wrap with julia
                warn("missing $(ni[i["name"]][rn,1])")
            end

            #3.3: random snr, calculate noise level based on speech level and snr
            t = 10^((sp-sn)/20)
            if i["type"] == "impulsive" 
                g = t / ni[i["name"]][rn,2]
            elseif i["type"] == "stationary"
                g = t / ni[i["name"]][rn,3]
            elseif i["type"] == "nonstationary"
                g = t / ni[i["name"]][rn,4]
            else
                error("wrong type in $(i["name"])")
            end
            g * ni[i["name"]][rn,2] > 1 && (g = 1 / ni[i["name"]][rn,2]; info("relax gain to avoid clipping $(ni[i["name"]][rn,1])"))
            u *= g
            
            #3.4: portion check
            nid = replace(relpath(ni[i["name"]][rn,1],s["noise_rootpath"]), "\\", "+")[1:end-4]
            sid = replace(relpath(si[rs,1],s["speech_rootpath"]), "\\", "+")[1:end-4]
            
            p = si[rs,4]
            q = ni[i["name"]][rn,5]
            if lowercase(s["mixbase"]) == "speech" 
                (x, u) = (u, x)
                p = ni[i["name"]][rn,5]
                q = si[rs,4]
            end
            η = p/q
            # x,p is the shorter signal
            # u,q is the longer signal

            if mr[1] <= η <= mr[2]
                rd = rand(1:q-p)
                u[rd:rd+p-1] += x
                # clipping sample if over-range?
                path = joinpath(mo,"$(nid)+$(sid)+1+$(sp)+$(sn).wav")
                wavwrite(u, path, Fs=fs)
                label[path] = [(rd, rd+p-1)]

            # η > mr[2] or η < mr[1]    
            else 
                np = 1
                nq = 1
                while !(mr[1] <= η <= mr[2])
                    η > mr[2] && (nq += 1)
                    η < mr[1] && (np += 1)
                    η = (np*p)/(nq*q)                    
                end
                path = joinpath(mo,"$(nid)+$(sid)+$(nq)+$(sp)+$(sn).wav")
                stamp = Array{Tuple{Int64, Int64},1}()

                u = repeat(u, outer=nq)
                pp = Int64(floor((nq*q)/np)) 
                for k = 0:np-1
                    rd = k*pp+rand(1:pp-p)
                    u[rd:rd+p-1] += x
                    push!(stamp,(rd, rd+p-1))
                end
                wavwrite(u, path, Fs=fs)
                label[path] = stamp
            end
            info("η = $η")
        end
    end
    open(joinpath(mo,"label.json"),"w") do f
        write(f, JSON.json(label))
    end
    info("label written to $(joinpath(mo,"label.json"))")
end


# remove old feature.h5 and make new
function feature(specification)
    # read the specification for feature extraction
    s = JSON.parsefile(specification)          
    fs = parse(Int64,s["samplerate"])          #16000
    n = parse(Int64,s["samplespace"])          #17    
    mo = s["mixoutput"]

    # clean level info
    si = readdlm(joinpath(s["speech_rootpath"],"index.level"), ',', header=false, skipstart=3)
    si = Dict(si[i,1] => si[i,2:end] for i = 1:size(si,1))
    
    # feature info
    m = parse(Int64, s["feature"]["frame_size"])
    d = parse(Int64, s["feature"]["step_size"])
    param = Frame1D{Int64}(fs, m, d, 0)
    win = Dict("Hamming"=>hamming, "Hann"=>hann)

    # read all mixed
    a = flist(mo, t=".wav")
    ptz = -1
    for (j,i) in enumerate(a)
        rm(i[1:end-length(".wav")]*".h5", force=true)
        p = split(i[1:end-length(".wav")],"+")
        #[1]"impulsive"
        #[2]"n48"      
        #[3]"dr1"      
        #[4]"mklw0"    
        #[5]"sa1"
        #[6]"3"      
        #[7]"-22.0"    
        #[8]"20.0"
        ref = joinpath(s["speech_rootpath"],p[3],p[4],p[5]) * ".wav"
        dup = parse(Int64,p[6])
        tagspl = parse(Float64,p[7])
        refspl = si[ref][2]
        refpk = si[ref][1]

        x = wavread(i)[1][:,1]
        r = wavread(ref)[1][:,1]

        g = 10^((tagspl-refspl)/20)
        g * refpk > 1 && (g = 1 / refpk; info("relax gain to avoid clipping $(refpk):$(refspl)->$(tagspl)(dB)"))
        r = g * repeat(r, outer=dup) #level speech to target level
        
        #wavwrite([x r], i*"label",Fs=fs)
        pxx = power_spectrum(x, param, m, window=win[s["feature"]["window_function"]])
        prr = power_spectrum(r, param, m, window=win[s["feature"]["window_function"]])

        h5write(i[1:end-length(".wav")]*".h5", "noisy", transpose(log.(pxx.+eps())))
        h5write(i[1:end-length(".wav")]*".h5", "clean", transpose(log.(prr.+eps())))
        #??? x 257 matrix

        pt = Int64(round((j/length(a))*100))
        in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; println("%$pt"))
    end
    info("feature done.")
end


# global variance:
# remove old global.h5 and make new
function gv(specification)
    # read the specification for feature extraction
    s = JSON.parsefile(specification)          
    fs = parse(Int64,s["samplerate"])          #16000
    mo = s["mixoutput"]
    m2 = div(parse(Int64, s["feature"]["frame_size"]), 2) + 1 # m2 = 257
    n = zero(Int128) #global frames
    rm(joinpath(mo,"global.h5"), force=true)

    a = flist(mo, t=".h5")
    la = length(a)

    # get global frame count
    ptz = -1
    for (j,i) in enumerate(a)
        x = h5read(i, "noisy") #(frames x m2) matrix
        n += size(x,1)            
        pt = Int64(round(100 * (j/la)))
        in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; print("."))
    end
    info("global spectrum count: $n")

    # get global mean log power spectrum
    μx = zeros(Float64, m2)
    μr = zeros(Float64, m2)
    μxi = zeros(BigFloat, la, m2)
    μri = zeros(BigFloat, la, m2)

    ptz = -1
    for(j,i) in enumerate(a)
        x = h5read(i, "noisy") #(frames x m2) matrix
        r = h5read(i, "clean")
        for k = 1:m2
            μxi[j,k] = sum_kbn(x[:,k])
            μri[j,k] = sum_kbn(r[:,k])
        end
        pt = Int64(round(100 * (j/la)))        
        in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; print("."))        
    end
    for k = 1:m2
        μx[k] = Float64(sum_kbn(μxi[:,k])/n)
        μr[k] = Float64(sum_kbn(μri[:,k])/n)
    end
    info("global spectrum mean (dimentionless): noise=$(mean(μx)), clean=$(mean(μr))")

    # get global std for unit variance
    σx = zeros(Float64, m2)
    σr = zeros(Float64, m2)
    fill!(μxi, zero(BigFloat))
    fill!(μri, zero(BigFloat))
    ptz = -1
    for(j,i) in enumerate(a)
        x = h5read(i, "noisy") #(frames x m2) matrix
        r = h5read(i, "clean")
        for k = 1:m2
            μxi[j,k] = sum_kbn((x[:,k]-μx[k]).^2)
            μri[j,k] = sum_kbn((r[:,k]-μr[k]).^2)
        end
        pt = Int64(round(100 * (j/la)))        
        in(pt, 0:10:100) && (pt != ptz) && (ptz = pt; print("."))        
    end
    for k = 1:m2
        σx[k] = Float64(sqrt(sum_kbn(μxi[:,k])/(n-1)))
        σr[k] = Float64(sqrt(sum_kbn(μri[:,k])/(n-1)))
    end
    info("global spectrum std (dimentionless): noise=$(mean(σx)), clean=$(mean(σr))")

    h5write(joinpath(mo,"global.h5"), "noisy_mu", μx)
    h5write(joinpath(mo,"global.h5"), "clean_mu", μr)
    h5write(joinpath(mo,"global.h5"), "noisy_std", σx)
    h5write(joinpath(mo,"global.h5"), "clean_std", σr)
    h5write(joinpath(mo,"global.h5"), "frames", Int64(n))
    assert(n < typemax(Int64))
    info("global results written to global.h5")
end