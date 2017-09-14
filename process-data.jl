using WAV
using DataFrames
using HDF5

include("./audio-features.jl")






function decode_to_linear_pcm(file <: AbstractString, encoders)
# file is w/ root folder    
    ext = lowercase(file[end - 3:end])
    dst = "/tmp/a.wav"

    if ext == ".wav"
        run(`cp $file $dst`)
    elseif in(ext, encoders) 
        run(`ffmpeg -i $file $dst -y`)
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

                decode_to_linear_pcm( joinpath(root, j) )
                run(`sox /tmp/a.wav -r $target_fs -b $target_bits /tmp/b.wav`)
                
                id = split(j[1:end - length(".wav")], "-")
                # id[1] : freesound id
                # id[2] : class id
                # id[3] : occurrence id
                # id[4] : slice id
                
                x, fs = wavread("/tmp/b.wav")
                assert(Int64(fs) == Int64(target_fs))
                x = stand(x[:, 1])
                    
                p = Frame1D{Int64}(target_fs, floor(0.025 * fs), floor(0.01 * fs), 0)
                fbe = filter_bank_energy(x, p, 512, zero_append=true, fl=100, fh=6800, use_log=true)
                #writetable("/tmp/c.csv",DataFrame(fbe), header=false)
                            
                if i == test_fold
                    dstp = joinpath(data_test, "$(j[1:end-length(".wav")]).h5")
                else
                    dstp = joinpath(data_train, "$(j[1:end-length(".wav")]).h5")
                end
                h5write(dstp, "id_class", parse(Int64, id[2], 10))
                h5write(dstp, "features", fbe)
            end
        end

    end
end
