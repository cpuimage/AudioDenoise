cmake_minimum_required(VERSION 3.9)
project(denoise)
add_executable(denoise main.c stftDenoise.c fft.c)