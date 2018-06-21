#include "audioDenoise.h"
#include "fft.h"


#ifndef nullptr
#define nullptr 0
#endif

int32_t fastPow(int32_t in, int32_t e)      // POWER(float,signed)
{
	float b = (float)in;
	if (e < 0) b = 1.0f / b;      // for negative powers, invert base
	e = (e ^ (e >> 31)) - (e >> 31);            // and raise to positive power

	float acc = 1.0f;           // init accumulator

	while (e) {
		if (e & 1) acc *= b;      // if LSB of exponent set, mult. by base
		e >>= 1;                // shift out LSB of exponent
		b *= b;                 // square base
	}
	return (int32_t)acc;
}

static const float m_lambda[3][5] = { {1.5f, 1.8f, 2.f,  2.5f, 2.5f},
									 {1.8f, 2.f,  2.5f, 3.5f, 3.5f},
									 {2.f,  2.5f, 3.5f, 4.7f, 4.7f} };


static void make_hanning_window(float *win, int32_t win_size) {
	float pi2 = 2 * 3.14159265358979323846f;
	int32_t half_win = win_size / 2;
	for (int32_t i = 0; i < half_win; i++) {
		win[i] = 0.5f - 0.5f * cosf(pi2 * i / (win_size - 1));
		win[win_size - 1 - i] = win[i];
	}
}

int roundup_pow_of_two(int x) {
	int r = 1;

	while (x) {
		x >>= 1;
		r <<= 1;
	}
	return r;
}

audioDenoiseHandle *audioDenoise_init(int32_t time_win, int32_t fs, int32_t *err, float sigma_noise) {
    if (time_win <= 0 || fs <= 0) {
        *err = ERROR_PARAMS;
        return nullptr;
    }
    int isMemFailed = 0;
    audioDenoiseHandle *handle = (audioDenoiseHandle *) malloc(sizeof(struct audioDenoiseHandle));
    if (handle == nullptr) return nullptr;
    // Compute hanning window
    handle->fs =fs;
    handle->win_size = roundup_pow_of_two(fs / 1000 * time_win);
    handle->half_win_size = handle->win_size / 2;
    handle->win_hanning = (float *) malloc(sizeof(float) * (handle->win_size));
    if (!(handle->win_hanning)) {
        isMemFailed = 1;
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
        isMemFailed = 1;
    }
    for (int32_t i = 0; i < handle->nblk_time; i++) {
        handle->SURE_matrix[i] = (float *) malloc(sizeof(float) * (handle->nblk_freq));
        if (!(handle->SURE_matrix[i])) {
            isMemFailed = 1;
            break;
        }
        memset(handle->SURE_matrix[i], 0, sizeof(float) * (handle->nblk_freq));
    }

    handle->inbuf = (float *) malloc(sizeof(float) * handle->win_size);
    if (!(handle->inbuf)) {
        isMemFailed = 1;
    }
    memset(handle->inbuf, 0, sizeof(float) * (handle->win_size));
    handle->inbuf_win = (float *) malloc(sizeof(float) * handle->win_size);
    if (!(handle->inbuf_win)) {
        isMemFailed = 1;
    }
    memset(handle->inbuf_win, 0, sizeof(float) * (handle->win_size));

    handle->outbuf = (float *) malloc(sizeof(float) * (handle->macro_size + handle->half_win_size));
    if (!(handle->outbuf)) {
        isMemFailed = 1;
    }
    memset(handle->outbuf, 0, sizeof(float) * (handle->macro_size + handle->half_win_size));

    handle->audio_coef = (fft_complex **) malloc(sizeof(fft_complex *) * (handle->max_nblk_time));
    if (!(handle->audio_coef)) {
        isMemFailed = 1;
    }
    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        handle->audio_coef[i] = (fft_complex *) malloc(sizeof(fft_complex) * (handle->win_size / 2 + 1));
        if (!(handle->audio_coef[i])) {
            isMemFailed = 1;
            break;
        }
    }
    handle->audio_thre = (fft_complex **) malloc(sizeof(fft_complex *) * (handle->max_nblk_time));
    if (!(handle->audio_thre)) {
        isMemFailed = 1;
    }
    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        handle->audio_thre[i] = (fft_complex *) malloc(sizeof(fft_complex) * (handle->win_size / 2 + 1));
        if (!(handle->audio_thre[i])) {
            isMemFailed = 1;
            break;
        }
    }
    handle->audio_coef_block = (fft_complex **) malloc(sizeof(fft_complex *) * (handle->max_nblk_time));
    if (!(handle->audio_coef_block)) {
        isMemFailed = 1;
    }
    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        handle->audio_coef_block[i] = (fft_complex *) malloc(sizeof(fft_complex) * (handle->max_nblk_freq));
        if (!(handle->audio_coef_block[i])) {
            isMemFailed = 1;
            break;
        }
    }
    handle->audio_coef_block_norm = (fft_complex **) malloc(sizeof(fft_complex *) * (handle->max_nblk_time));
    if (!(handle->audio_coef_block_norm)) {
        isMemFailed = 1;
    }
    for (int32_t i = 0; i < handle->max_nblk_time; i++) {
        handle->audio_coef_block_norm[i] = (fft_complex *) malloc(sizeof(fft_complex) * (handle->max_nblk_freq));
        if (!(handle->audio_coef_block_norm[i])) {
            isMemFailed = 1;
            break;
        }
    }
    if (isMemFailed == 0) {
        *err = OK;
        return handle;
    }

    audioDenoise_free(handle);
    handle = nullptr;
    *err = ERROR_MEMORY;
    return handle;
}

int32_t audioDenoise_reset(audioDenoiseHandle *handle) {
	if (!handle) {
		return ERROR_PARAMS;
	}
	handle->have_nblk_time = 0;
	for (int32_t i = 0; i < handle->nblk_time; i++) {
		memset(handle->SURE_matrix[i], 0, sizeof(float) * (handle->nblk_freq));
	}
	memset(handle->inbuf, 0, sizeof(float) * (handle->win_size));
	memset(handle->outbuf, 0, sizeof(float) * (handle->macro_size + handle->half_win_size));

	return OK;
}


static void audioDenoise_audio(audioDenoiseHandle *handle) {
	//filter with window
	for (int32_t i = 0; i < handle->win_size; i++) {
		(handle->inbuf_win)[i] = (handle->inbuf)[i] * (handle->win_hanning)[i];
	}
	fft_plan forward_plan = fft_plan_dft_r2c_1d(handle->win_size, handle->inbuf_win,
		handle->audio_coef[handle->have_nblk_time], 0);
	fft_execute(forward_plan);
	fft_destroy_plan(forward_plan);
}

static void audioDenoise_inverse_audio(audioDenoiseHandle *handle) {
	int32_t half_win_size = handle->half_win_size;
	memcpy(handle->outbuf,
		handle->outbuf + handle->macro_size,
		sizeof(float) * half_win_size);
	memset(handle->outbuf + half_win_size, 0,
		sizeof(float) * (handle->macro_size));

	for (int32_t i = 0; i < handle->max_nblk_time; i++) {
		fft_plan backward_plan = fft_plan_dft_c2r_1d(handle->win_size, handle->audio_coef[i], handle->inbuf_win, 0);
		fft_execute(backward_plan);
		fft_destroy_plan(backward_plan);
		float inv_winsize = 1.0f / (handle->win_size);
		for (int32_t j = 0; j < handle->win_size; j++) {
			handle->outbuf[half_win_size * i + j] += handle->inbuf_win[j] * inv_winsize;
		}
	}
}

// calculate the power of audio in block [row_start:row_end, col_start:col_end]
static float power_audio(fft_complex **data,
	int32_t row_start, int32_t row_end,
	int32_t col_start, int32_t col_end) {
	float sum = 0;

	for (int32_t row = row_start; row <= row_end; row++) {
		for (int32_t col = col_start; col <= col_end; col++) {
			//sum += pow(data[row][col].r, 2) + pow(data[row][col].i, 2);
			float r = data[row][col].real;
			float i = data[row][col].imag;
			sum += (r * r) + (i * i);
		}
	}

	return sum;
}

// calculate the energy of audio in real dimension
static float energy_real_audio(fft_complex **data,
	int32_t row_start, int32_t row_end,
	int32_t col_start, int32_t col_end) {
	float sum = 0;
	float r = 0;
	for (int32_t row = row_start; row <= row_end; row++) {
		for (int32_t col = col_start; col <= col_end; col++) {
			r = data[row][col].real;
			sum += (r * r);
		}
	}

	return sum;
}

// implement scalar multiply: dst_matrix = src_matrix * a
static void scalar_multiply(fft_complex **dst_matrix, fft_complex **src_matrix,
	int32_t row_start, int32_t row_end,
	int32_t col_start, int32_t col_end,
	float a) {
	for (int32_t row = row_start; row <= row_end; row++) {
		for (int32_t col = col_start; col <= col_end; col++) {
			dst_matrix[row][col].real = src_matrix[row][col].real * a;
			dst_matrix[row][col].imag = src_matrix[row][col].imag * a;
		}
	}
}

static void audioDenoise_adaptive_block(audioDenoiseHandle *handle,
	int32_t ith_half_macroblk_frq,
	int32_t *seg_time, int32_t *seg_freq) {
	float SURE_real = 0;
	float energy_real = 0;
	float size_blk = 0;
	float min_SURE_real = 0;
	float lambda = 0;
	float temp = 0;
	int32_t TT, FF;
	float norm = sqrtf(2.0f) / (sqrtf((float)handle->win_size) * (handle->sigma_hanning_noise));

	//Get audio coef macro block and block norm
	for (int32_t index_blk_time = 0; index_blk_time < handle->max_nblk_time; index_blk_time++) {
		int32_t index_blk_freq = 1 + ith_half_macroblk_frq * (handle->max_nblk_freq);
		for (int32_t i = 0; i < handle->max_nblk_freq; i++) {
			(handle->audio_coef_block)[index_blk_time][i].real =
				(handle->audio_coef)[index_blk_time][index_blk_freq + i].real;
			(handle->audio_coef_block)[index_blk_time][i].imag =
				(handle->audio_coef)[index_blk_time][index_blk_freq + i].imag;
			(handle->audio_coef_block_norm)[index_blk_time][i].real =
				(handle->audio_coef_block)[index_blk_time][i].real * norm;
			(handle->audio_coef_block_norm)[index_blk_time][i].imag =
				(handle->audio_coef_block)[index_blk_time][i].imag * norm;
		}
	}

	//Compute adaptive block
	for (int32_t T = 0; T < handle->nblk_time; T++) {//loop over time 
		TT = (handle->max_nblk_time) * fastPow(2, -T);
		for (int32_t F = 0; F < handle->nblk_freq; F++) {//loop over frequency
			FF = (handle->max_nblk_freq) * fastPow(2, -F);
			lambda = m_lambda[T][F];
			SURE_real = 0;
			size_blk = (float)TT * FF;
			temp = (lambda * lambda) * (size_blk * size_blk) - 2 * lambda * size_blk * (size_blk - 2);
			for (int32_t ii = 0; ii < fastPow(2, T); ii++) {
				for (int32_t jj = 0; jj < fastPow(2, F); jj++) {
					energy_real = energy_real_audio(handle->audio_coef_block_norm,
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

static void audioDenoise_compute_thre(audioDenoiseHandle *handle,
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
			a = 1.0f - L_Weight / power_audio(handle->audio_coef_block,
				TT_ii, TT_ii + TT - 1,
				FF_jj, FF_jj + FF - 1);
			a = a * (a > 0);
			// udpate attenuation map
			int32_t idx_base = 1 + ith_half_macro_freq * (handle->max_nblk_freq);
			for (int32_t kk = 0; kk < TT; kk++) {
				int32_t idx_row = TT_ii + kk;
				for (int32_t ww = 0; ww < FF; ww++) {
					int32_t idx_col = FF_jj + ww;
					(handle->audio_thre)[idx_row][idx_base + idx_col].real =
						(handle->audio_coef_block)[idx_row][idx_col].real * a;
					(handle->audio_thre)[idx_row][idx_base + idx_col].imag =
						(handle->audio_coef_block)[idx_row][idx_col].imag * a;
				}
			}
		}
	}
}

static void audioDenoise_wiener(audioDenoiseHandle *handle) {
	float wiener = 0;
	float sigma = handle->sigma_hanning_noise;
    float low_f = 500.0;
    int low_f_index = low_f / (handle->fs / handle->half_win_size);

    float w_sigma = (handle->win_size) * (sigma * sigma);
	for (int32_t t = 0; t < handle->max_nblk_time; t++) {
		for (int32_t f = 0; f < (handle->win_size + 1) / 2; f++) {
			float r = (handle->audio_thre)[t][f].real;
			float i = (handle->audio_thre)[t][f].imag;
			wiener = (r * r) + (i * i);
			wiener = wiener / (wiener + w_sigma);
			handle->audio_coef[t][f].real *= wiener;
			handle->audio_coef[t][f].imag *= wiener;
            // attenuate more below low_f Hz
            if (f < low_f_index) {
                handle->audio_coef[t][f].real *= 0.45f;
                handle->audio_coef[t][f].imag *= 0.45f;
            }
		}
	}
}

static void audioDenoise_core(audioDenoiseHandle *handle) {
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
	//        / power_audio(handle->audio_coef, 0, handle->max_nblk_time-1, 0, 0);
	a = 1.0f - (Lambda_pi * L_pi * L_sigma)
		/ power_audio(handle->audio_coef, 0, handle->max_nblk_time - 1, 0, 0);
	if (a < 0) {
		a = 0;
	}
	scalar_multiply(handle->audio_thre, handle->audio_coef, 0, handle->max_nblk_time - 1, 0, 0, a);

	// negative frequency part
	for (int32_t i = 0; i < half_nb_macroblk_frq; i++) {
		//adaptive block
		audioDenoise_adaptive_block(handle, i, &seg_time, &seg_freq);

		//compute the attenuation map base on adaptive block segmentation
		audioDenoise_compute_thre(handle, i, seg_time, seg_freq);
	}

	// for last few frequency that do not match 2D MarcroBlock
	idx_freq_last = 1 + half_nb_macroblk_frq * (handle->max_nblk_freq);
	if (idx_freq_last < (handle->win_size / 2 + 1)) {
		for (int32_t i = idx_freq_last; i < (handle->win_size / 2 + 1); i++) {
			//a = Lambda_pi*L_pi*pow(handle->sigma_hanning_noise, 2)*(handle->win_size);
			a = Lambda_pi * L_pi * L_sigma;
			a = 1 - a / power_audio(handle->audio_coef, 0, handle->max_nblk_time - 1, i, i);
			if (a < 0) {
				a = 0;
			}
			scalar_multiply(handle->audio_thre, handle->audio_coef,
				0, handle->max_nblk_time - 1,
				i, i, a);
		}
	}
	// wiener filter
	audioDenoise_wiener(handle);
}

int32_t audioDenoise_denoise_scalar(audioDenoiseHandle *handle,
	float *in, int32_t in_len) {
	if ((in_len != handle->half_win_size) || (!in)) {
		return ERROR_PARAMS;
	}

	// update inbuf
	int32_t half_win_size = handle->half_win_size;
	memcpy(handle->inbuf, handle->inbuf + half_win_size, sizeof(float) * half_win_size);
	memcpy(handle->inbuf + half_win_size, in, sizeof(float) * half_win_size);

	// do audio
	audioDenoise_audio(handle);

	(handle->have_nblk_time)++;

	if (handle->have_nblk_time != handle->max_nblk_time) {
		return NEED_MORE_SAMPLES;
	}

	// block thresholding
	audioDenoise_core(handle);

	// do inverse audio
	audioDenoise_inverse_audio(handle);

	handle->have_nblk_time = 0;

	return CAN_OUTPUT;
}


int32_t audioDenoise_output_scalar(audioDenoiseHandle *handle,
	float *out, int32_t out_len) {
	if (out_len < handle->macro_size) {
		return 0;
	}

	memcpy(out, handle->outbuf, handle->macro_size * sizeof(float));

	return handle->macro_size;
}


int32_t audioDenoise_flush_scalar(audioDenoiseHandle *handle,
	float *out, int32_t out_len) {
	int32_t half_win_size = handle->half_win_size;
	int32_t out_size = (handle->have_nblk_time) * half_win_size;

	if (out_len < out_size) {
		return -1;
	}

	memcpy(handle->outbuf,
		handle->outbuf + handle->macro_size,
		sizeof(float) * half_win_size);
	memset(handle->outbuf + half_win_size, 0,
		sizeof(float) * (handle->macro_size));

	for (int32_t i = 0; i < handle->have_nblk_time; i++) {
		fft_plan backward_plan = fft_plan_dft_c2r_1d(handle->win_size, handle->audio_coef[i], handle->inbuf_win, 0);
		fft_execute(backward_plan);
		fft_destroy_plan(backward_plan);
		float inv_winsize = 1.0f / (handle->win_size);
		for (int32_t j = 0; j < handle->win_size; j++) {
			handle->outbuf[half_win_size * i + j] += handle->inbuf_win[j] * inv_winsize;
		}
	}

	memcpy(out, handle->outbuf, out_size * sizeof(float));

	return out_size;
}

void audioDenoise_free(audioDenoiseHandle *handle) {
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

		if (handle->audio_coef) {
			for (int32_t i = 0; i < handle->max_nblk_time; i++) {
				if (handle->audio_coef[i])
					free(handle->audio_coef[i]);
			}
			free(handle->audio_coef);
		}
		if (handle->audio_thre) {
			for (int32_t i = 0; i < handle->max_nblk_time; i++) {
				if (handle->audio_thre[i])
					free(handle->audio_thre[i]);
			}
			free(handle->audio_thre);
		}
		if (handle->audio_coef_block) {
			for (int32_t i = 0; i < handle->max_nblk_time; i++) {
				if (handle->audio_coef_block[i])
					free(handle->audio_coef_block[i]);
			}
			free(handle->audio_coef_block);
		}
		if (handle->audio_coef_block_norm) {
			for (int32_t i = 0; i < handle->max_nblk_time; i++) {
				if (handle->audio_coef_block_norm[i])
					free(handle->audio_coef_block_norm[i]);
			}
			free(handle->audio_coef_block_norm);
		}
		free(handle);
	}
}

int32_t audioDenoise_max_output(const audioDenoiseHandle *handle) {
	return handle->macro_size;
}

int32_t audioDenoise_samples_per_time(const audioDenoiseHandle *handle) {
	return handle->half_win_size;
}

