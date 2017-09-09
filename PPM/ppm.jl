#P6
#1920 1080
#255
#<frame RGB>
#
# ASCII dump:
# 0x50 0x36   0x0a
# 0x31 0x39 0x32 0x30   0x20   0x31 0x30 0x38 0x30   0x0a
# 0x32 0x35 0x35   0x0a
# 
# lixun 2017/08/09
p6 = [0x50, 0x36, 0x0a]
horizontal = "480"
vertical = "640"
fps = 30
duration = 3600.7
start = 1
depth = [0x32, 0x35, 0x35, 0x0a]

noise_floor = "alexa-f0.ppm"
digest = "alexa.digest"



function delta(a, b)
    a > b && (return a-b)
    b-a
end

h = tryparse(Int64, horizontal)
h = isnull(h) ? 0 : get(h)
v = tryparse(Int64, vertical)
v = isnull(v) ? 0 : get(v)

header = zeros(UInt8, length(p6) + length(horizontal) + 1 + length(vertical) + 1 + length(depth))
f0 = zeros(UInt8, h*v*3)
fx = zeros(UInt8, h*v*3)
rgb = zeros(Float64, convert(Int64, ceil(duration * fps)), 3)


# capture the inactive frame as basis
fn = convert(Int64, ceil(start * fps))
for i = 1:fn
    read!(STDIN, header)
    read!(STDIN, f0)
end
open(noise_floor, "w") do f
    write(f, header)
    write(f, f0)
end


# dealing with the rest frames
i = 1
while !eof(STDIN)
    read!(STDIN, header)
    read!(STDIN, fx)
    #version 1: signal[i] = sum(frame) - md
    for k = 1:3
        rgb[i,k] = sum(delta.(fx[k:3:end], f0[k:3:end])) / (h*v)
    end 
    #version 3: signal[i] = delta(sum(frame), md)
    i += 1
end
open(digest, "w") do f
    for j = 1:i
        write(f, "$(rgb[j,1]), $(rgb[j,2]), $(rgb[j,3])\n")
    end
end

