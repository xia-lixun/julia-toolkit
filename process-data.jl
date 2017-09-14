using WAV
using DataFrames
using HDF5

include("./audio-features.jl")






function decode_to_linear_pcm(file::AbstractString, encoders, dst)
# file is w/ root folder
# dst = "/tmp/a.wav"
    ext = lowercase(file[end - 3:end])
    if ext == ".wav"
        run(`cp $file $dst`)
    elseif in(ext, encoders) 
        run(`ffmpeg -i $file $dst -y`)
    else
        error("encoder type not supported: $ext")
    end
    nothing
end




function process_data()

    # iterate over the dataset tree and for each .wav|.ogg|.mp3|.aac|.ac3|.mp4 etc. do:
    # [1] cp a.wav /tmp/a.wav || ffmpeg -i a.!wav /tmp/a.wav
    # [2] sox /tmp/a.wav -r 16000 /tmp/a-16000.wav
    # [3] x,fs=wavread("/tmp/a-16000.wav")  
    #     parse("a-16000.wav") for classid info/folder info
    #     x = stand(x)
    #     fbe = filter_bank_energies(x)
    # [4] h5write(fbe) to test/train folders
    encoders = Set([".mp3", ".ogg", ".ac3", ".aac", ".mp4"])

    path_train = "/home/lixun/Downloads/train"
    path_test = "/home/lixun/Downloads/test" 
    path_root = "/home/lixun/Downloads/UrbanSound8K-SmallSet/"
    path_branch = "audio/fold"

    path_pcm1 = "/tmp/a.wav"
    path_pcm2 = "/tmp/b.wav"

    frame_duration = 0.025
    frame_step_duration = 0.01
    test_fold = 7
    target_fs = 16000
    target_bits = 16

    rm(path_train, force=true, recursive=true)
    rm(path_test, force=true, recursive=true)
    mkpath(path_train)
    mkpath(path_test)


    for i = 1:10
    
        # for each data fold, find all the audio files within the sub folders
        path = joinpath(path_root, path_branch) * "$i"
        for (root, dirs, files) in walkdir(path)

            for j in files

                decode_to_linear_pcm( joinpath(root, j), encoders, path_pcm1 )
                run(`sox $path_pcm1 -r $target_fs -b $target_bits $path_pcm2`)
                
                id = split(j[1:end - length(".wav")], "-")
                # id[1] : freesound id
                # id[2] : class id
                # id[3] : occurrence id
                # id[4] : slice id
                
                # convert multichannel to mono and normalize to unit variance
                x, fs = wavread(path_pcm2)
                assert(Int64(fs) == Int64(target_fs))
                x = mean(x,2)
                
                #channel = findmax(std(x,1))[2]
                x = stand(x[:,1])
                
                # extend normalized time series to uniform length
                #if length(x) > target_clip_samples
                #    x = x[1:target_clip_samples]
                #elseif length(x) < target_clip_samples
                #    x = [x; zeros(typeof(x[1]), target_clip_samples-length(x))]
                #end

                #extract spectral features
                p = Frame1D{Int64}(target_fs, floor(frame_duration * fs), floor(frame_step_duration * fs), 0)
                fbe = filter_bank_energy(x, p, 512, zero_append=true, fl=100, fh=6800, use_log=true)
                #writetable("/tmp/c.csv",DataFrame(fbe), header=false)
                
                if i == test_fold
                    dstp = joinpath(path_test, "$(j[1:end-length(".wav")]).h5")
                else
                    dstp = joinpath(path_train, "$(j[1:end-length(".wav")]).h5")
                end
                h5write(dstp, "class_id", parse(Int64, id[2], 10))
                h5write(dstp, "audio", fbe)
                h5write(dstp, "sr", target_fs)
                info("$j processed")
            end
        end

    end
end
