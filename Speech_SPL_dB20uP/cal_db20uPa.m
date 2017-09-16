bound = [0.25 10.45];
% 
% Calibration:
% 
% @MRP, use 1/4 mic
% @DUT, use 1/2 mic
%%%%%%%%%%%%%%%
clear all
close all
clc

calib_recording = '.\1000hz-piston-114dBSPL-46AN.wav';
%calib_recording = '.\1000hz-piston-114dBSPL-40AN.wav';


%recording = './Vol69-46AN-Skype_RCV_EQ.wav';
%source = '.\PreparationRCV.wav';
%c = [1e6, 1.5e6, 2.5e6];
%d = [1.5e6, 2.5e6, 3.5e6];

%recording = '.\Vol50-40AN-SkypeRcvEQ.wav';
%source = '.\PreparationRCV.wav';
%c = [0.2e6, 1.5e6, 3e6];
%d = [1.5e6, 3e6, 4.5e6];

%recording = '.\Vol50-SkypeRcvEQ-AplayBPF-40AN.wav';
%source = '.\PreparationRCV_100_8000_BPF.wav';
%c = [0.4e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

%recording = '.\Vol100-CotRcvEQ-AplayBPF-40AN.wav';
%source = '.\PreparationRCV_100_8000_BPF.wav';
%c = [0.4e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

%recording = './100_8000_BPF/Vol69-46AN-Skype_RCV_EQ-Source-BPF.wav';
%source = '.\100_8000_BPF\PreparationRCV_100_8000_BPF.wav';
%c = [0.5e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

%recording = '.\vol50-skypeRcvEq-AcquaPlay-46AN.wav';
%source = '.\PreparationRCV.wav';
%c = [0.4e6, 1e6, 2e6];
%d = [1e6, 2e6, 3e6];

%recording = '.\vol50-skypeRcvEq-AplayBPF-46AN.wav';
%source = '.\PreparationRCV_100_8000_BPF.wav';
%c = [0.4e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

%recording = '.\Vol70-CotRcvEq-AplayBPF-46AN.wav';
%source = '.\PreparationRCV_100_8000_BPF.wav';
%c = [0.4e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

recording = '.\Vol100-CotRcvEq-AplayBPF-46AN.wav';
source = '.\PreparationRCV_100_8000_BPF.wav';
c = [0.4e6, 1e6, 1.5e6];
d = [1e6, 1.5e6, 2e6];



% data preparation
[gp, fs] = audioread(recording);
assert(fs == 48000);
gp = gp(:,1);

[g,fs] =  audioread(source);
assert(fs == 48000);
g = mean(g,2);

rxx = xcorr(g, gp);
plot(rxx);

sig = [];
for i = 1:3
    [a(i),b(i)] = max(rxx(c(i):d(i)));
    peak = c(i) + b(i) - 1;
    lb = length(gp) - peak + 1;
    rb = lb + length(g) - 1;
    figure; plot(gp(lb:rb))
    sig = [sig; gp(lb:rb)];
end
figure; plot(sig)




[ref, fs] = audioread(calib_recording);



blk = 16384;
step = blk/4;
win = hann(blk);

sig = buffer(sig, blk, blk-step, 'nodelay');
sig = sig .* repmat(win,1,size(sig,2));
sig_p = (abs(fft(sig))).^2;
sig_p = mean(sig_p(1:blk/2+1,:),2);

ref = buffer(ref(:,1), blk, blk-step, 'nodelay');
ref = ref .* repmat(win,1,size(ref,2));
ref_p = (abs(fft(ref))).^2;
ref_p = mean(ref_p(1:blk/2+1,:),2);


xl = 0:length(sig_p)-1;
figure; plot(xl/blk*fs,sig_p); grid on;
figure; plot(xl/blk*fs,ref_p); grid on;


fl = 100;
fh = 12000;

fl = floor(fl/fs*blk);
fh = floor(fh/fs*blk);

offset = 10*log10(sum(ref_p(fl:fh)));
offsetMic = 114 - offset;
10*log10(sum(sig_p(fl:fh))) + offsetMic
%




