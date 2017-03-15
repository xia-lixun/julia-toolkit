# Speech Project Benchmark Platform
# Lixun Xia



using WAV
using DataFrames
include("utility.jl")
include("resample.jl")



type config
    analysis_time::Float32
    sample_rate::Float32
    root_path::String
end    


# put user parameters here
param = config(5 * 60 + 37, 16000, "C:\\Scientific\\Workspace")
( param.root_path[end] == '\\' )  ||  ( param.root_path = "$(param.root_path)\\" )



# <1> load the reference speech
CleanTalk5min,fs = wavread("C:\\Program Files (x86)\\Microsoft\\OEMScoreUtilityFarAndNear\\ReferenceFiles\\CleanTalk5min.wav")
assert(fs == param.sample_rate)
assert(size(CleanTalk5min,2) == 1)

Chirp,fs = wavread("C:\\Program Files (x86)\\Microsoft\\OEMScoreUtilityFarAndNear\\ReferenceFiles\\Chirp.wav")
assert(fs == param.sample_rate)
assert(size(Chirp,2) == 1)


# <2> traverse the folder tree for all wav files
WavFilesForEval = locate(root, ".wav")
info("files to be processed:")
for t in WavFilesForEval
    info(t)
end    

# <3> read channels of wav files for processing
for t in WavFilesForEval

    data,fs = wavread(t)
    info("processing $t")

    #iterate over all channels
    for k = 1:size(data,2)
        # <3.1> resample
        fs == 48000.0f0  &&  ( data16k = resample(data(:,k), param.sample_rate, fs, FIR48kTO16k) )
        fs == 44100.0f0  &&  ( data16k = resample(data(:,k), param.sample_rate, fs, FIR44kTO16k) )
        #use data16k for post processing
        
        # <3.2> correlate resampled dta with the chirp for start time, remove content before t-5.0
        Rxy = xcorr(data16k, Chirp)
        abs(maximum(Rxy) < 0.40f0  &&  error("Chirp not found in channel $k of $t")
        start = find(Rxy .== abs(maximum(Rxy))
        info("starting point @ $start ")
        data16k = data16k[start-1*convert(Int64,param.sample_rate):start+315*convert(Int64,param.sample_rate)]
        info("segmentation done")
        
        # <3.3> write to temp folder for scoring
        wavwrite(data16k, , param.sample_rate, 32, )

        # <3.4> invoke the score command and blocking for results
        # run(`score`)

        # <3.5> analyze the score results
        # save recogonition file along with the score number
        # diff the recogonition file and make highlights?
        # correlation coeffs of all speech segments? more statistics on that?
    end
end

# generate the final report as DataFrames and write to file
