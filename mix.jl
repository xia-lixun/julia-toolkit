# management of wav files for machine learning projects
# lixun.xia@outlook.com
# 2017-10-16
using SHA
using WAV
using JSON




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
        "mixsnr" => ["-20", "-10", "0", "10", "20"],
        "speechlevel" => ["-22", "-32", "-42"],
        "mixrange" => ["0.1", "0.6"],
        "mixoutput" => "D:\\mix-utility\\Mixed",
        "speech_rootpath" => "D:\\mix-utility\\VoiceBox\\16k-LP7",
        "noise_rootpath" => "D:\\mix-utility\\NoiseSet",
        "build_speechlevel_index" => "true",
        "build_noiselevel_index" => "true",
        "noise_categories" => x
        )
    for i in flist(a["noise_rootpath"])
        push!(x, Dict("name"=>i,"type"=>"stationary|nonstationary|impulse","percent"=>"0.0"))
    end

    rm(a["mixoutput"], force=true, recursive=true)
    mkpath(a["mixoutput"])
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
    m = length(s["noise_rootpath"])


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
    #index format: path-to-wav-file, peak-level, rms-level, median, length-in-samples, filename-uid
    ni = Dict( x["name"] => readdlm(joinpath(s["noise_rootpath"], x["name"],"index.level"),',') for x in s["noise_categories"])


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
    #index format: path-to-wav-file, peak-level, speech-level(dB), length-in-samples, filename-uid
    si = readdlm(joinpath(s["speech_rootpath"],"index.level"))
    #(si, ni)


    # Part III. Mixing them up
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
                x = wavread(joinpath(s["speech_rootpath"],si[rs,1]))[1][:,1]
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
                warn("missing $(si[rn,1])")
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
            nfid = replace(ni[i["name"]][rn,1][m+1:end-length(".wav")], "\\", "+")

            p = si[rs,4]
            q = ni[i["name"]][rn,5]
            η = p/q
            if mr[2] >= η >= mr[1]
                rd = rand(1:q-p)
                u[rd:rd+p-1] += x
                # clipping sample if over-range?
                # ni[i["name"]][rn,6] #noise file name
                # si[rs,5]            #speech file name
                # sp                  #speech level
                # snr[rr]             #snr
                path = joinpath(mo,"$(nfid)+$(si[rs,5])+$(sp)+$(sn)+$(rd)+$(rd+p-1)+")
                wavwrite(u, path*".wav", Fs=fs)
                info("η = $η")

            elseif η > mr[2]
                m = Int64(floor(q*mr[2]))
                rd = rand(1:q-m)
                u[rd:rd+m-1] += x[1:m]
                path = joinpath(mo,"$(nfid)+$(si[rs,5])+$(sp)+$(sn)+$(rd)+$(rd+m-1)+")
                wavwrite(u, path*".wav", Fs=fs)
                info("η > $(mr[2])")

            elseif η < mr[1]
                m = Int64(ceil(q*mr[1]/p))
                rd = Dict{Int64,Int64}()
                rd[0] = -p + 1
                path = joinpath(mo,"$(nfid)+$si[rs,5]+$(sp)+$(sn)+")
                for k = 1:m
                    rd[k] = rand( rd[k-1]+p : q-(m-(k-1))*p )
                    u[rd[k] : rd[k]+p-1] += x
                    path *= "$(rd[k])+$(rd[k]+p-1)+"
                end
                # rd[1] = rand(1:q-m*p)
                # rd[2] = rand(rd[1]+p:q-(m-1)*p)
                # rd[3] = rand(rd[2]+p:q-(m-2)*p)
                info("η < $(mr[1])")
                wavwrite(u, path*".wav", Fs=fs)
            end
        end
    end
end