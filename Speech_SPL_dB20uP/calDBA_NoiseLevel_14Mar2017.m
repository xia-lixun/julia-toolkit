
% 
% Calibration:
% 
% @MRP, use 1/4 mic
% @DUT, use 1/2 mic
%%%%%%%%%%%%%%%
clear all
close all
clc

use_dba = 0;
%calib_recording = '.\1000hz-piston-114dBSPL-40AN.wav';
calib_recording = '.\1000hz-piston-114dBSPL-46AN.wav';

%recording = './Vol69-46AN-Skype_RCV_EQ.wav';
%source = '.\PreparationRCV.wav';
%c = [1e6, 1.5e6, 2.5e6];
%d = [1.5e6, 2.5e6, 3.5e6];

%recording = './100_8000_BPF/Vol69-46AN-Skype_RCV_EQ-Source-BPF.wav';
%source = '.\100_8000_BPF\PreparationRCV_100_8000_BPF.wav';
%c = [0.5e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

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

%recording = '.\Vol100-CotRcvEq-AplayBPF-46AN.wav';
%source = '.\PreparationRCV_100_8000_BPF.wav';
%c = [0.4e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

%recording = '.\Vol60-CotRcvEq-AplayBPF-46AN.wav';
%source = '.\PreparationRCV_100_8000_BPF.wav';
%c = [0.4e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

%recording = '.\Vol100-CotRcvEq-AplayBPF-46AN-12dB.wav';
%source = '.\PreparationRCV_100_8000_BPF-Minus12dB.wav';
%c = [0.4e6, 1e6, 1.5e6];
%d = [1e6, 1.5e6, 2e6];

recording = '.\FactoryMode-CotRcvEq-AplayBPF-46AN-12dB.wav';
source = '.\PreparationRCV_100_8000_BPF-Minus12dB.wav';
c = [0.4e6, 1e6, 2e6];
d = [1e6, 2e6, 3e6];




% data preparation
[gresp, fs] = audioread(recording);
assert(fs == 48000);
gresp = gresp(:,1);

[g,fs] =  audioread(source);
assert(fs == 48000);
g = mean(g,2);

rxx = xcorr(g, gresp);
plot(rxx);

sig = [];
for i = 1:3
    [a(i),b(i)] = max(rxx(c(i):d(i)));
    peak = c(i) + b(i) - 1;
    lb = length(gresp) - peak + 1;
    rb = lb + length(g) - 1;
    figure; plot(gresp(lb:rb))
    sig = [sig; gresp(lb:rb)];
end
figure; plot(sig)

fn = fs/2;
[b,a] = butter(4,[100/fn,12000/fn]);



[refMic, fs] = audioread(calib_recording);
if use_dba
    refMicW = filterA(refMic(:,1), fs);
    sigW = filterA(sig(:,1), fs);
else
    refMicW = filter(b,a,refMic(:,1));
    sigW = filter(b,a,sig);
end
offset = 10*log10(var(refMicW));

if use_dba
    offsetMic = 105.4 - offset;
else
    offsetMic = 114 - offset;
end
10*log10(var(sigW)) + offsetMic
%




