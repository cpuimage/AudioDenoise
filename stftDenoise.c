#include "stftDenoise.h"

#define FORWARD_FFT 0
#define BACKWARD_FFT 1

#ifndef nullptr
#define nullptr 0
#endif

int32_t fastPow(int32_t in, int32_t e)      // POWER(float,signed)
{
    float b = (float) in;
    if (e < 0) b = 1.0f / b;      // for negative powers, invert base
    e = (e ^ (e >> 31)) - (e >> 31);            // and raise to positive power

    float acc = 1.0f;           // init accumulator

    while (e) {
        if (e & 1) acc *= b;      // if LSB of exponent set, mult. by base
        e >>= 1;                // shift out LSB of exponent
        b *= b;                 // square base
    }
    return (int32_t) acc;
}

static const float m_lambda[3][5] = {{1.5f, 1.8f, 2.f,  2.5f, 2.5f},
                                     {1.8f, 2.f,  2.5f, 3.5f, 3.5f},
                                     {2.f,  2.5f, 3.5f, 4.7f, 4.7f}};



static void make_hanning_window(float *win, int32_t win_size) {
    float pi2 = 2 * 3.14159265358979323846f;
    int32_t half_win = win_size / 2;
    for (int32_t i = 0; i < half_win; i++) {
        win[i] = 0.5f - 0.5f * cosf(pi2 * i / (win_size - 1));
        win[win_size - 1 - i] = win[i];
    }
}

stftDenoiseHandle *stftDenoise_init(int32_t time_win, int32_t fs, int32_t *err, float sigma_noise) {
    if (time_win <= 0 || fs <= 0) {
        *err = ERROR_PARAMS;
        return nullptr;
    }

    stftDenoiseHandle *handle = (stftDenoiseHandle *) malloc(sizeof(struct StftDenoiseHandle));
    if (handle == nullptr) return nullptr;
    // Compute hanning window
    handle->win_size = fs / 1000 * time_win;
    if (handle->win_size & 0x01) {
        handle->win_size += 1;// even window
    }
    handle->half_win_size = handle->win_size / 2;
    handle->win_hanning = (float *) malloc(sizeof(float) * (handle->win_size));
    if (!(handle->win_hanning)) {
        goto end;
    }
    make_hanning_window(handle->win_hanning, handle->win_size);

    //Compute block params
    handle->max_nblk_time = 8;
    handle->max_nblk_freq = 16;
    handle->nblk_time = 3;
    handle->nblk_freq = 5;
    handle->sigma_noise = sigma_noise;
    handle->sigma_hanning_noise = handle->sigma_noise * sqrtf(0.375f);
    handle->macro_size = handle->half_win_size * handle->max_nblk_time;
    handle->have_nblk_time = 0;

    handle->SURE_matrix = (float **) malloc(sizeof(float *) * (handle->nblk_time));
    if (!(handle->SURE_matrix)) {
        goto end;
    }
    for (int32_t i = 0; i < handle->nblk_time; i++) {
        handle->SURE_matrix[i] = (float *) malloc(sizeof(float) * (handle->nblk_freq));
        if (!(handle->SURE_matrix[i])) {
            goto end;
        }
        memset(handle->SURE_matrix[i], 0, sizeof(float) * (handle->nblk_freq));
    }

    handle->inbuf = (kiss_fft_scalar *) malloc(sizeof(kiss_fft_scalar) * handle->win_size);
    if (!(handle->inbuf)) {
        goto end;
    }
    memset(handle->inbuf, 0, sizeof(kiss_fft_scalar) * (handle->win_size));
    handle->inbuf_win = (kiss_fft_scalar *) malloc(sizeof(kiss_fft_scalar) * handle->win_size);
    if (!(handle->inbuf_win)) {
        goto end;
    }
    memset(handle->inbuf_win, 0, sizeof(kiss_fft_scalar) * (handle->win_size));

    handle->outbuf = (kiss_fft_scalar *) malloc(sizeof(kiss_fft_scalar) * (handle->macro_size + handle->half_win_size));
    if (!(handle->outbuf)) {
        goto end;
    }
    memset(handle->outbuf, 0, sizeof(kiss_fft_scalar) * (handle->macro_size + handle->half_win_size));

    handle->stft_coef = (kiss_fft_cpx **) malloc(sizeof(kiss_fft_cpx *) * (handle->max_nblk_time));
    if (!(handle->stft_coef)) {
        goto end;
    }
    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        handle->stft_coef[i] = (kiss_fft_cpx *) malloc(sizeof(kiss_fft_cpx) * (handle->win_size / 2 + 1));
        if (!(handle->stft_coef[i])) {
            goto end;
        }
    }
    handle->stft_thre = (kiss_fft_cpx **) malloc(sizeof(kiss_fft_cpx *) * (handle->max_nblk_time));
    if (!(handle->stft_thre)) {
        goto end;
    }
    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        handle->stft_thre[i] = (kiss_fft_cpx *) malloc(sizeof(kiss_fft_cpx) * (handle->win_size / 2 + 1));
        if (!(handle->stft_thre[i])) {
            goto end;
        }
    }
    handle->stft_coef_block = (kiss_fft_cpx **) malloc(sizeof(kiss_fft_cpx *) * (handle->max_nblk_time));
    if (!(handle->stft_coef_block)) {
        goto end;
    }
    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        handle->stft_coef_block[i] = (kiss_fft_cpx *) malloc(sizeof(kiss_fft_cpx) * (handle->max_nblk_freq));
        if (!(handle->stft_coef_block[i])) {
            goto end;
        }
    }
    handle->stft_coef_block_norm = (kiss_fft_cpx **) malloc(sizeof(kiss_fft_cpx *) * (handle->max_nblk_time));
    if (!(handle->stft_coef_block_norm)) {
        goto end;
    }
    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        handle->stft_coef_block_norm[i] = (kiss_fft_cpx *) malloc(sizeof(kiss_fft_cpx) * (handle->max_nblk_freq));
        if (!(handle->stft_coef_block_norm[i])) {
            goto end;
        }
    }

    handle->forward_fftr_cfg = kiss_fftr_alloc(handle->win_size, FORWARD_FFT, 0, 0);
    if (!(handle->forward_fftr_cfg)) {
        goto end;
    }
    handle->backward_fftr_cfg = kiss_fftr_alloc(handle->win_size, BACKWARD_FFT, 0, 0);
    if (!(handle->backward_fftr_cfg)) {
        goto end;
    }

    *err = OK;
    return handle;

    end:

    stftDenoise_free(handle);
    handle = nullptr;
    *err = ERROR_MEMORY;
    return handle;
}

int32_t stftDenoise_reset(stftDenoiseHandle *handle) {
    if (!handle) {
        return ERROR_PARAMS;
    }
    handle->have_nblk_time = 0;
    for (int32_t i = 0; i < handle->nblk_time; i++) {
        memset(handle->SURE_matrix[i], 0, sizeof(float) * (handle->nblk_freq));
    }
    memset(handle->inbuf, 0, sizeof(kiss_fft_scalar) * (handle->win_size));
    memset(handle->outbuf, 0, sizeof(kiss_fft_scalar) * (handle->macro_size + handle->half_win_size));

    return OK;
}


static void stftDenoise_STFT(stftDenoiseHandle *handle) {
    //filter with window
    for (int32_t i = 0; i < handle->win_size; i++) {
        (handle->inbuf_win)[i] = (handle->inbuf)[i] * (handle->win_hanning)[i];
    }

    kiss_fftr(handle->forward_fftr_cfg, handle->inbuf_win,
              handle->stft_coef[handle->have_nblk_time]);
}

static void stftDenoise_inverse_STFT(stftDenoiseHandle *handle) {
    int32_t half_win_size = handle->half_win_size;

    memcpy(handle->outbuf,
           handle->outbuf + handle->macro_size,
           sizeof(kiss_fft_scalar) * half_win_size);
    memset(handle->outbuf + half_win_size, 0,
           sizeof(kiss_fft_scalar) * (handle->macro_size));

    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        kiss_fftri(handle->backward_fftr_cfg, handle->stft_coef[i], handle->inbuf_win);
        for (int32_t j = 0; j < handle->win_size; j++) {
            handle->outbuf[half_win_size * i + j] += handle->inbuf_win[j] / (handle->win_size);
        }
    }
}

// calculate the power of STFT in block [row_start:row_end, col_start:col_end]
static float power_STFT(kiss_fft_cpx **data,
                        int32_t row_start, int32_t row_end,
                        int32_t col_start, int32_t col_end) {
    float sum = 0;

    for (int32_t row = row_start; row <= row_end; row++) {
        for (int32_t col = col_start; col <= col_end; col++) {
            //sum += pow(data[row][col].r, 2) + pow(data[row][col].i, 2);
            kiss_fft_scalar r = data[row][col].r;
            kiss_fft_scalar i = data[row][col].i;
            sum += (r * r) + (i * i);
        }
    }

    return sum;
}

// calculate the energy of STFT in real dimension
static float energy_real_STFT(kiss_fft_cpx **data,
                              int32_t row_start, int32_t row_end,
                              int32_t col_start, int32_t col_end) {
    float sum = 0;
    float r = 0;
    for (int32_t row = row_start; row <= row_end; row++) {
        for (int32_t col = col_start; col <= col_end; col++) {
            r = data[row][col].r;
            sum += (r * r);
        }
    }

    return sum;
}

// implement scalar multiply: dst_matrix = src_matrix * a
static void scalar_multiply(kiss_fft_cpx **dst_matrix, kiss_fft_cpx **src_matrix,
                            int32_t row_start, int32_t row_end,
                            int32_t col_start, int32_t col_end,
                            float a) {
    for (int32_t row = row_start; row <= row_end; row++) {
        for (int32_t col = col_start; col <= col_end; col++) {
            dst_matrix[row][col].r = src_matrix[row][col].r * a;
            dst_matrix[row][col].i = src_matrix[row][col].i * a;
        }
    }
}

static void stftDenoise_adaptive_block(stftDenoiseHandle *handle,
                                       int32_t ith_half_macroblk_frq,
                                       int32_t *seg_time, int32_t *seg_freq) {
    float SURE_real = 0;
    float energy_real = 0;
    float size_blk = 0;
    float min_SURE_real = 0;
    float lambda = 0;
    float temp = 0;
    int32_t TT, FF;
    float norm = sqrtf(2.0f) / (sqrtf((float) handle->win_size) * (handle->sigma_hanning_noise));

    //Get STFT coef macro block and block norm
    for (int32_t index_blk_time = 0; index_blk_time < handle->max_nblk_time; index_blk_time++) {
        int32_t index_blk_freq = 1 + ith_half_macroblk_frq * (handle->max_nblk_freq);
        for (int32_t i = 0; i < handle->max_nblk_freq; i++) {
            (handle->stft_coef_block)[index_blk_time][i].r =
                    (handle->stft_coef)[index_blk_time][index_blk_freq + i].r;
            (handle->stft_coef_block)[index_blk_time][i].i =
                    (handle->stft_coef)[index_blk_time][index_blk_freq + i].i;
            (handle->stft_coef_block_norm)[index_blk_time][i].r =
                    (handle->stft_coef_block)[index_blk_time][i].r * norm;
            (handle->stft_coef_block_norm)[index_blk_time][i].i =
                    (handle->stft_coef_block)[index_blk_time][i].i * norm;
        }
    }

    //Compute adaptive block
    for (int32_t T = 0; T < handle->nblk_time; T++) {//loop over time 
        TT = (handle->max_nblk_time) * fastPow(2, -T);
        for (int32_t F = 0; F < handle->nblk_freq; F++) {//loop over frequency
            FF = (handle->max_nblk_freq) * fastPow(2, -F);
            lambda = m_lambda[T][F];
            SURE_real = 0;
            size_blk = (float) TT * FF;
            temp = (lambda * lambda) * (size_blk * size_blk) - 2 * lambda * size_blk * (size_blk - 2);
            for (int32_t ii = 0; ii < fastPow(2, T); ii++) {
                for (int32_t jj = 0; jj < fastPow(2, F); jj++) {
                    energy_real = energy_real_STFT(handle->stft_coef_block_norm,
                                                   TT * ii, TT * (ii + 1) - 1,
                                                   FF * jj, FF * (jj + 1) - 1);
                    SURE_real += size_blk + temp / energy_real * (energy_real > lambda * size_blk)
                                 + (energy_real - 2 * size_blk) * (energy_real <= lambda * size_blk);
                }
            }

            handle->SURE_matrix[T][F] = SURE_real;
        }
    }

    // find mini SURE segmentation
    min_SURE_real = handle->SURE_matrix[0][0];
    *seg_time = 0;
    *seg_freq = 0;
    for (int32_t i = 0; i < handle->nblk_time; i++) {
        for (int32_t j = 0; j < handle->nblk_freq; j++) {
            if (handle->SURE_matrix[i][j] < min_SURE_real) {
                min_SURE_real = handle->SURE_matrix[i][j];
                *seg_time = i;
                *seg_freq = j;
            }
        }
    }
}

static void stftDenoise_compute_thre(stftDenoiseHandle *handle,
                                     int32_t ith_half_macro_freq,
                                     int32_t seg_time,
                                     int32_t seg_freq) {
    int32_t TT = (handle->max_nblk_time) * fastPow(2, -seg_time);
    int32_t FF = (handle->max_nblk_freq) * fastPow(2, -seg_freq);
    float a = 0;
    float lambda = m_lambda[seg_time][seg_freq];
    float L_sigma = (handle->sigma_hanning_noise * handle->sigma_hanning_noise) * (handle->win_size);
    float L_Weight = lambda * TT * FF * L_sigma;
    for (int32_t ii = 0; ii < fastPow(2, seg_time); ii++) {
        int32_t TT_ii = TT * ii;
        for (int32_t jj = 0; jj < fastPow(2, seg_freq); jj++) {
            int32_t FF_jj = FF * jj;
            a = 1.0f - L_Weight / power_STFT(handle->stft_coef_block,
                                             TT_ii, TT_ii + TT - 1,
                                             FF_jj, FF_jj + FF - 1);
            a = a * (a > 0);
            // udpate attenuation map
            int32_t idx_base = 1 + ith_half_macro_freq * (handle->max_nblk_freq);
            for (int32_t kk = 0; kk < TT; kk++) {
                int32_t idx_row = TT_ii + kk;
                for (int32_t ww = 0; ww < FF; ww++) {
                    int32_t idx_col = FF_jj + ww;
                    (handle->stft_thre)[idx_row][idx_base + idx_col].r =
                            (handle->stft_coef_block)[idx_row][idx_col].r * a;
                    (handle->stft_thre)[idx_row][idx_base + idx_col].i =
                            (handle->stft_coef_block)[idx_row][idx_col].i * a;
                }
            }
        }
    }
}

static void stftDenoise_wiener(stftDenoiseHandle *handle) {
    float wiener = 0;
    kiss_fft_scalar sigma = handle->sigma_hanning_noise;
    kiss_fft_scalar w_sigma = (handle->win_size) * (sigma * sigma);
    for (int32_t t = 0; t < handle->max_nblk_time; t++) {
        for (int32_t f = 0; f < (handle->win_size + 1) / 2; f++) {
            kiss_fft_scalar r = (handle->stft_thre)[t][f].r;
            kiss_fft_scalar i = (handle->stft_thre)[t][f].i;
            wiener = (r * r) + (i * i);
            wiener = wiener / (wiener + w_sigma);
            handle->stft_coef[t][f].r *= wiener;
            handle->stft_coef[t][f].i *= wiener;
        }
    }
}

static void stftDenoise_core(stftDenoiseHandle *handle) {
    float L_pi = 8.0;
    float Lambda_pi = 2.5;
    float a = 0;
    int32_t half_nb_macroblk_frq = (handle->win_size - 1) / 2 / (handle->max_nblk_freq);
    int32_t seg_time = 0;
    int32_t seg_freq = 0;
    int32_t idx_freq_last = 0;
    float L_sigma = (handle->sigma_hanning_noise * handle->sigma_hanning_noise) * (handle->win_size);

    // DC part
    //a = 1 - (Lambda_pi*L_pi*pow(handle->sigma_hanning_noise,2)*(handle->win_size)) 
    //        / power_STFT(handle->stft_coef, 0, handle->max_nblk_time-1, 0, 0);
    a = 1.0f - (Lambda_pi * L_pi * L_sigma)
               / power_STFT(handle->stft_coef, 0, handle->max_nblk_time - 1, 0, 0);
    if (a < 0) {
        a = 0;
    }
    scalar_multiply(handle->stft_thre, handle->stft_coef, 0, handle->max_nblk_time - 1, 0, 0, a);

    // negative frequency part
    for (int32_t i = 0; i < half_nb_macroblk_frq; i++) {
        //adaptive block
        stftDenoise_adaptive_block(handle, i, &seg_time, &seg_freq);

        //compute the attenuation map base on adaptive block segmentation
        stftDenoise_compute_thre(handle, i, seg_time, seg_freq);
    }

    // for last few frequency that do not match 2D MarcroBlock
    idx_freq_last = 1 + half_nb_macroblk_frq * (handle->max_nblk_freq);
    if (idx_freq_last < (handle->win_size / 2 + 1)) {
        for (int32_t i = idx_freq_last; i < (handle->win_size / 2 + 1); i++) {
            //a = Lambda_pi*L_pi*pow(handle->sigma_hanning_noise, 2)*(handle->win_size);
            a = Lambda_pi * L_pi * L_sigma;
            a = 1 - a / power_STFT(handle->stft_coef, 0, handle->max_nblk_time - 1, i, i);
            if (a < 0) {
                a = 0;
            }
            scalar_multiply(handle->stft_thre, handle->stft_coef,
                            0, handle->max_nblk_time - 1,
                            i, i, a);
        }
    }
    // wiener filter
    stftDenoise_wiener(handle);
}

int32_t stftDenoise_denoise_scalar(stftDenoiseHandle *handle,
                                  kiss_fft_scalar *in, int32_t in_len) {
    if ((in_len != handle->half_win_size) || (!in)) {
        return ERROR_PARAMS;
    }

    // update inbuf
    int32_t half_win_size = handle->half_win_size;
    memcpy(handle->inbuf, handle->inbuf + half_win_size, sizeof(kiss_fft_scalar) * half_win_size);
    memcpy(handle->inbuf + half_win_size, in, sizeof(kiss_fft_scalar) * half_win_size);

    // do STFT
    stftDenoise_STFT(handle);

    (handle->have_nblk_time)++;

    if (handle->have_nblk_time != handle->max_nblk_time) {
        return NEED_MORE_SAMPLES;
    }
 
    // block thresholding
    stftDenoise_core(handle);

    // do inverse STFT
    stftDenoise_inverse_STFT(handle);

    handle->have_nblk_time = 0;

    return CAN_OUTPUT;
}


int32_t stftDenoise_output_scalar(stftDenoiseHandle *handle,
                                 kiss_fft_scalar *out, int32_t out_len) {
    if (out_len < handle->macro_size) {
        return 0;
    }

    memcpy(out, handle->outbuf, handle->macro_size * sizeof(kiss_fft_scalar));

    return handle->macro_size;
}



int32_t stftDenoise_flush_scalar(stftDenoiseHandle *handle,
                                kiss_fft_scalar *out, int32_t out_len) {
    int32_t half_win_size = handle->half_win_size;
    int32_t out_size = (handle->have_nblk_time) * half_win_size;

    if (out_len < out_size) {
        return -1;
    }

    memcpy(handle->outbuf,
           handle->outbuf + handle->macro_size,
           sizeof(kiss_fft_scalar) * half_win_size);
    memset(handle->outbuf + half_win_size, 0,
           sizeof(kiss_fft_scalar) * (handle->macro_size));

    for (int32_t i = 0; i < handle->have_nblk_time; i++) {
        kiss_fftri(handle->backward_fftr_cfg, handle->stft_coef[i], handle->inbuf_win);
        for (int32_t j = 0; j < handle->win_size; j++) {
            handle->outbuf[half_win_size * i + j] += handle->inbuf_win[j] / (handle->win_size);
        }
    }

    memcpy(out, handle->outbuf, out_size * sizeof(kiss_fft_scalar));

    return out_size;
}

void stftDenoise_free(stftDenoiseHandle *handle) {
    if (handle) {
        if (handle->win_hanning)
            free(handle->win_hanning);
        if (handle->SURE_matrix) {
            for (int32_t i = 0; i < handle->nblk_time; i++) {
                if (handle->SURE_matrix[i])
                    free(handle->SURE_matrix[i]);
            }
            free(handle->SURE_matrix);
        }
        if (handle->inbuf)
            free(handle->inbuf);
        if (handle->outbuf)
            free(handle->outbuf);
        if (handle->inbuf_win)
            free(handle->inbuf_win);

        if (handle->stft_coef) {
            for (int32_t i = 0; i < handle->max_nblk_time; i++) {
                if (handle->stft_coef[i])
                    free(handle->stft_coef[i]);
            }
            free(handle->stft_coef);
        }
        if (handle->stft_thre) {
            for (int32_t i = 0; i < handle->max_nblk_time; i++) {
                if (handle->stft_thre[i])
                    free(handle->stft_thre[i]);
            }
            free(handle->stft_thre);
        }
        if (handle->stft_coef_block) {
            for (int32_t i = 0; i < handle->max_nblk_time; i++) {
                if (handle->stft_coef_block[i])
                    free(handle->stft_coef_block[i]);
            }
            free(handle->stft_coef_block);
        }
        if (handle->stft_coef_block_norm) {
            for (int32_t i = 0; i < handle->max_nblk_time; i++) {
                if (handle->stft_coef_block_norm[i])
                    free(handle->stft_coef_block_norm[i]);
            }
            free(handle->stft_coef_block_norm);
        }
        if (handle->forward_fftr_cfg)
            free(handle->forward_fftr_cfg);
        if (handle->backward_fftr_cfg)
            free(handle->backward_fftr_cfg);
        free(handle);
    }
}

int32_t stftDenoise_max_output(const stftDenoiseHandle *handle) {
    return handle->macro_size;
}

int32_t stftDenoise_samples_per_time(const stftDenoiseHandle *handle) {
    return handle->half_win_size;
}

