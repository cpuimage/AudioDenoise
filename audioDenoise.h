#ifndef AUDIO_DENOISE_H_
#define AUDIO_DENOISE_H_

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "fft.h"

#define OK                 0x00
#define ERROR_MEMORY       0x01
#define ERROR_PARAMS       0x02
#define NEED_MORE_SAMPLES  0x10
#define CAN_OUTPUT         0x20

typedef struct audioDenoiseHandle {
    int32_t fs;
    int32_t win_size;    // window size--odd window
    int32_t half_win_size; // half window size
    float *win_hanning; // hanning window

    int32_t max_nblk_time;
    int32_t max_nblk_freq;
    int32_t nblk_time;  // the number of block in time dimension
    int32_t nblk_freq;  // the number of block in frequency dimension
    int32_t macro_size; // the number of sample in one macro block
    int32_t have_nblk_time;
    float **SURE_matrix;

    float sigma_noise;  // assumption the sigma of gaussian white noise
    float sigma_hanning_noise;
    float *inbuf;       // internal buffer for keep one window size input samples
    float *inbuf_win;
    float *outbuf;      // internal buffer for keep one macro block output samples

    fft_complex **audio_coef;
    fft_complex **audio_thre;
    fft_complex **audio_coef_block;
    fft_complex **audio_coef_block_norm;

} audioDenoiseHandle;

audioDenoiseHandle *audioDenoise_init(int32_t time_win, int32_t fs, int32_t *err, float sigma_noise);

int32_t audioDenoise_reset(audioDenoiseHandle *handle);

int32_t audioDenoise_denoise_scalar(audioDenoiseHandle *handle, float *in, int32_t in_len);

int32_t audioDenoise_output_scalar(audioDenoiseHandle *handle, float *out, int32_t out_len);

int32_t audioDenoise_flush_scalar(audioDenoiseHandle *handle, float *out, int32_t out_len);

void audioDenoise_free(audioDenoiseHandle *handle);

int32_t audioDenoise_max_output(const audioDenoiseHandle *handle);

int32_t audioDenoise_samples_per_time(const audioDenoiseHandle *handle);

#endif