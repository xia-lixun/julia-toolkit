
% 
% Calibration:
% 
% @MRP, use 1/4 mic
% @DUT, use 1/2 mic
%%%%%%%%%%%%%%%
clear all
close all
clc

use_dba = 1;


% data preparation
[gresp, fs] = audioread('./Skype-EQ/1m-distance/rcv_recording-vol69-1m-skypeRcvEq.wav');
assert(fs == 48000);
gresp = gresp(:,1);

[g,fs] =  audioread('.\PreparationRCV.wav');
assert(fs == 48000);
g = sum(g,2)/2;

rxx = xcorr(g, gresp);
plot(rxx);

sig = [];
c = [0.5e6, 1e6, 1.5e6];
d = [1e6, 1.5e6, 2e6];
for i = 1:3
    [a(i),b(i)] = max(rxx(c(i):d(i)));
    peak = c(i) + b(i) - 1;
    lb = length(gresp) - peak + 1;
    rb = lb + length(g) - 1;
    figure; plot(gresp(lb:rb))
    sig = [sig; gresp(lb:rb)];
end


fn = fs/2;
[b,a] = butter(4,[50/fn,12000/fn]);



[refMic, fs] = audioread('.\250hz-piston-114_15dB.wav');
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
    offsetMic = 114.15 - offset;
end
10*log10(var(sigW)) + offsetMic
%






