#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "timing.h"
#include "audioDenoise.h"

#ifdef _WIN32

#include <Windows.h>

#else
#include <unistd.h>
#endif

//采用https://github.com/mackron/dr_libs/blob/master/dr_wav.h 解码
#define DR_WAV_IMPLEMENTATION

#include "dr_wav.h"

#ifndef nullptr
#define nullptr 0
#endif

#ifndef MIN
#define MIN(A, B)        ((A) < (B) ? (A) : (B))
#endif

//写wav文件
void wavWrite_scalar(char *filename, float *buffer, size_t sampleRate, size_t totalSampleCount) {
    drwav_data_format format ;
    format.container = drwav_container_riff;     // <-- drwav_container_riff = normal WAV files, drwav_container_w64 = Sony Wave64.
    format.channels = 1;
    format.sampleRate = (drwav_uint32) sampleRate;
    format.bitsPerSample = sizeof(float) * 8;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;

    drwav *pWav = drwav_open_file_write(filename, &format);
    if (pWav) {
        drwav_uint64 samplesWritten = drwav_write(pWav, totalSampleCount, buffer);
        drwav_uninit(pWav);
        if (samplesWritten != totalSampleCount) {
            fprintf(stderr, "ERROR\n");
            exit(1);
        }
    }
}

//读取wav文件
float *wavRead_scalar(char *filename, uint32_t *sampleRate, uint64_t *totalSampleCount) {
    unsigned int channels;
    float *buffer = drwav_open_and_read_file_f32(filename, &channels, sampleRate,
                                                           totalSampleCount);
    if (buffer == nullptr) {
        printf("读取wav文件失败.");
    }
    //仅仅处理单通道音频
    if (channels != 1) {
        drwav_free(buffer);
        buffer = nullptr;
        *sampleRate = 0;
        *totalSampleCount = 0;
    }
    return buffer;
}

//分割路径函数
void splitpath(const char *path, char *drv, char *dir, char *name, char *ext) {
    const char *end;
    const char *p;
    const char *s;
    if (path[0] && path[1] == ':') {
        if (drv) {
            *drv++ = *path++;
            *drv++ = *path++;
            *drv = '\0';
        }
    } else if (drv)
        *drv = '\0';
    for (end = path; *end && *end != ':';)
        end++;
    for (p = end; p > path && *--p != '\\' && *p != '/';)
        if (*p == '.') {
            end = p;
            break;
        }
    if (ext)
        for (s = end; (*ext = *s++);)
            ext++;
    for (p = end; p > path;)
        if (*--p == '\\' || *p == '/') {
            p++;
            break;
        }
    if (name) {
        for (s = p; s < end;)
            *name++ = *s++;
        *name = '\0';
    }
    if (dir) {
        for (s = path; s < p;)
            *dir++ = *s++;
        *dir = '\0';
    }
}

int DenoiseProc(float *buffer, int sampleRate, size_t SampleCount, int32_t time_win, float sigma_noise) {
    int32_t ret = -1;
    audioDenoiseHandle *denoise_handle = NULL;
    denoise_handle = audioDenoise_init(time_win, sampleRate, &ret, sigma_noise);
    if (ret != OK || denoise_handle == NULL) {
        printf("audioDenoise_init fail.\n");
        return -1;
    }
    int32_t samples = audioDenoise_samples_per_time(denoise_handle);
    int32_t outbuf_len = audioDenoise_max_output(denoise_handle);

    float *input = buffer;
    float *output = buffer;
    size_t nTotal = SampleCount / samples;
	size_t i = 0;
    for (i = 0; i < nTotal; i++) {
        ret = audioDenoise_denoise_scalar(denoise_handle, input, samples);
        if (ret == ERROR_PARAMS) {
            break;
        } else if (ret == NEED_MORE_SAMPLES) {
            input += samples;
            continue;
        } else if (ret == CAN_OUTPUT) {
            int32_t len = audioDenoise_output_scalar(denoise_handle, output, outbuf_len);
            output += len;
        }
        input += samples;
    }
    audioDenoise_free(denoise_handle);

    return 0;
}

void audio_deNoise(char *in_file, char *out_file) {
    uint32_t sampleRate = 0; 
    uint64_t inSampleCount = 0;
    float *inBuffer = wavRead_scalar(in_file, &sampleRate, &inSampleCount);
 
    if (inBuffer != nullptr) {
        int32_t time_win = 50;
        float sigma_noise =0.047f;
        double startTime = now();
        DenoiseProc(inBuffer, sampleRate, inSampleCount, time_win, sigma_noise);
        double time_interval = calcElapsed(startTime, now());
        printf("time interval: %d ms\n ", (int) (time_interval * 1000));
        wavWrite_scalar(out_file, inBuffer, sampleRate, inSampleCount);
        free(inBuffer);
    }
}

int main(int argc, char *argv[]) {
    printf("Audio Denoise by Time-Frequency Block Thresholding\n");
    printf("blog:http://cpuimage.cnblogs.com/\n");
    if (argc < 2)
        return -1;
    char *in_file = argv[1];
    char drive[3];
    char dir[256];
    char fname[256];
    char ext[256];
    char out_file[1024];
    splitpath(in_file, drive, dir, fname, ext);
    sprintf(out_file, "%s%s%s_out%s", drive, dir, fname, ext);
    audio_deNoise(in_file, out_file);

    printf("press any key to exit.\n");
    getchar();
    return 0;
}


