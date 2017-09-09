#P6
#1920 1080
#255
#<frame RGB>
header = zeros(UInt8, 128)
read!(STDIN, header)
println(header)