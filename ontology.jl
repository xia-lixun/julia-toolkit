#fetch audio data from google audioset
bal_train = readdlm("AudioSet/balanced_train_segments.csv")
bal_eval = readdlm("AudioSet/eval_segments.csv")
unbal_train = readdlm("AudioSet/unbalanced_train_segments.csv")

const tag = "/m/09x0r"

for i = 1:size(bal_train,1)
    contains(bal_train[i,4], tag)  &&  run(`youtube-dl.exe -f m4a https://www.youtube.com/watch?v=$(bal_train[i,1][1:end-1])`)
end

for i = 1:size(bal_eval,1)
    contains(bal_eval[i,4], tag)  &&  run(`youtube-dl.exe -f m4a https://www.youtube.com/watch?v=$(bal_eval[i,1][1:end-1])`)
end

for i = 1:size(unbal_train,1)
    contains(unbal_train[i,4], tag)  &&  run(`youtube-dl.exe -f m4a https://www.youtube.com/watch?v=$(unbal_train[i,1][1:end-1])`)
end

