cmake_minimum_required(VERSION 3.9)
project(denoise)
add_executable(denoise main.c audioDenoise.c fft.c)
target_link_libraries(denoise -lm)