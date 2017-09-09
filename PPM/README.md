To inspect the header info: ffmpeg.exe -i alexa-highcontrast.mp4 -f image2pipe -vcodec ppm pipe:1 | julia ppm_head.jl
To generate the digest: ffmpeg.exe -i alexa-highcontrast.mp4 -f image2pipe -vcodec ppm pipe:1 | julia ppm.jl
