% source signal for Shenzhen
% apply 100Hz - 8000kHz bandpass to the original signal to simulate the
% Skype receive-path effect.
close all; clear all; clc;

[g,fs] =  audioread('.\PreparationRCV.wav');
assert(fs == 48000);

fn = fs/2;
[b,a] = butter(4,[100/fn,8000/fn]);
gf = filter(b,a,g,[],1);
sig = gf(:,2);

blk = 16384;
step = blk/4;
win = hann(blk);

sig = buffer(sig, blk, blk-step, 'nodelay');
sig = sig .* repmat(win,1,size(sig,2));
sig_p = (abs(fft(sig))).^2;
sig_p = mean(sig_p(1:blk/2+1,:),2);

xl = 0:length(sig_p)-1;
figure; plot(xl/blk*fs,sig_p); grid on;

audiowrite('./PreparationRCV_100_8000_BPF.wav', gf, fs, 'BitsPerSample',32);