//-----------------------------------------------------------------------------

#ifndef _FFT_H_
#define _FFT_H_


#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#define FFT_FORWARD 1
#define FFT_BACKWARD 2
#define FFT_ESTIMATE 3

typedef struct {
    float real;
    float imag;
} fft_complex;

typedef struct {
    int n;
    int sign;
    unsigned int flags;
    fft_complex *c_in;
    float *in;
    fft_complex *c_out;
    float *out;
    float *input;
    int *ip;
    float *w;
} fft_plan;

fft_plan fft_plan_dft_1d(size_t n, fft_complex *in, fft_complex *out, int sign,
                         unsigned int flags);

fft_plan fft_plan_dft_c2r_1d(size_t n, fft_complex *in, float *out,
                             unsigned int flags);

fft_plan fft_plan_dft_r2c_1d(size_t n, float *in, fft_complex *out,
                             unsigned int flags);

void fft_execute(fft_plan p);

void fft_destroy_plan(fft_plan p);
 
#ifdef __cplusplus
}
#endif

#endif

//-----------------------------------------------------------------------------