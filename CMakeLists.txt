cmake_minimum_required(VERSION 3.9)
project(denoise)

set(CMAKE_CXX_STANDARD 11)
add_executable(denoise main.c stftDenoise.c
        kiss_fft.c kiss_fftr.c
        )