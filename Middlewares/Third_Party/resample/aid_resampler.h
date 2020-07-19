/* Copyright (C) 2007 Jean-Marc Valin
      
   File: aid_resampler.h
   Resampling code
      
   The design goals of this code are:
      - Very fast algorithm
      - Low memory requirement
      - Good *perceptual* quality (and not best SNR)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef AID_RESAMPLER_H
#define AID_RESAMPLER_H

#ifdef OUTSIDE_AID

/********* WARNING: MENTAL SANITY ENDS HERE *************/

/* If the resampler is defined outside of AID, we change the symbol names so that 
   there won't be any clash if linking with AID later on. */

/* #define RANDOM_PREFIX your software name here */
#ifndef RANDOM_PREFIX
#error "Please define RANDOM_PREFIX (above) to something specific to your project to prevent symbol name clashes"
#endif

#define CAT_PREFIX2(a,b) a ## b
#define CAT_PREFIX(a,b) CAT_PREFIX2(a, b)
      
#define aid_resampler_init CAT_PREFIX(RANDOM_PREFIX,_resampler_init)
#define aid_resampler_init_frac CAT_PREFIX(RANDOM_PREFIX,_resampler_init_frac)
#define aid_resampler_destroy CAT_PREFIX(RANDOM_PREFIX,_resampler_destroy)
#define aid_resampler_process_float CAT_PREFIX(RANDOM_PREFIX,_resampler_process_float)
#define aid_resampler_process_int CAT_PREFIX(RANDOM_PREFIX,_resampler_process_int)
#define aid_resampler_process_interleaved_float CAT_PREFIX(RANDOM_PREFIX,_resampler_process_interleaved_float)
#define aid_resampler_process_interleaved_int CAT_PREFIX(RANDOM_PREFIX,_resampler_process_interleaved_int)
#define aid_resampler_set_rate CAT_PREFIX(RANDOM_PREFIX,_resampler_set_rate)
#define aid_resampler_get_rate CAT_PREFIX(RANDOM_PREFIX,_resampler_get_rate)
#define aid_resampler_set_rate_frac CAT_PREFIX(RANDOM_PREFIX,_resampler_set_rate_frac)
#define aid_resampler_get_ratio CAT_PREFIX(RANDOM_PREFIX,_resampler_get_ratio)
#define aid_resampler_set_quality CAT_PREFIX(RANDOM_PREFIX,_resampler_set_quality)
#define aid_resampler_get_quality CAT_PREFIX(RANDOM_PREFIX,_resampler_get_quality)
#define aid_resampler_set_input_stride CAT_PREFIX(RANDOM_PREFIX,_resampler_set_input_stride)
#define aid_resampler_get_input_stride CAT_PREFIX(RANDOM_PREFIX,_resampler_get_input_stride)
#define aid_resampler_set_output_stride CAT_PREFIX(RANDOM_PREFIX,_resampler_set_output_stride)
#define aid_resampler_get_output_stride CAT_PREFIX(RANDOM_PREFIX,_resampler_get_output_stride)
#define aid_resampler_get_input_latency CAT_PREFIX(RANDOM_PREFIX,_resampler_get_input_latency)
#define aid_resampler_get_output_latency CAT_PREFIX(RANDOM_PREFIX,_resampler_get_output_latency)
#define aid_resampler_skip_zeros CAT_PREFIX(RANDOM_PREFIX,_resampler_skip_zeros)
#define aid_resampler_reset_mem CAT_PREFIX(RANDOM_PREFIX,_resampler_reset_mem)
#define aid_resampler_strerror CAT_PREFIX(RANDOM_PREFIX,_resampler_strerror)

#define aid_int16_t short
#define aid_int32_t int
#define aid_uint16_t unsigned short
#define aid_uint32_t unsigned int
      
#else /* OUTSIDE_AID */

#include "aiddsp_types.h"

#endif /* OUTSIDE_AID */

#ifdef __cplusplus
extern "C" {
#endif

#define AID_RESAMPLER_QUALITY_MAX 10
#define AID_RESAMPLER_QUALITY_MIN 0
#define AID_RESAMPLER_QUALITY_DEFAULT 4
#define AID_RESAMPLER_QUALITY_VOIP 3
#define AID_RESAMPLER_QUALITY_DESKTOP 5

enum {
   RESAMPLER_ERR_SUCCESS         = 0,
   RESAMPLER_ERR_ALLOC_FAILED    = 1,
   RESAMPLER_ERR_BAD_STATE       = 2,
   RESAMPLER_ERR_INVALID_ARG     = 3,
   RESAMPLER_ERR_PTR_OVERLAP     = 4,
   
   RESAMPLER_ERR_MAX_ERROR
};

struct AIDResamplerState_;
typedef struct AIDResamplerState_ AIDResamplerState;

/** Create a new resampler with integer input and output rates.
 * @param nb_channels Number of channels to be processed
 * @param in_rate Input sampling rate (integer number of Hz).
 * @param out_rate Output sampling rate (integer number of Hz).
 * @param quality Resampling quality between 0 and 10, where 0 has poor quality
 * and 10 has very high quality.
 * @return Newly created resampler state
 * @retval NULL Error: not enough memory
 */
AIDResamplerState *aid_resampler_init(aid_uint32_t nb_channels, 
                                          aid_uint32_t in_rate, 
                                          aid_uint32_t out_rate, 
                                          int quality,
                                          int *err);

/** Create a new resampler with fractional input/output rates. The sampling 
 * rate ratio is an arbitrary rational number with both the numerator and 
 * denominator being 32-bit integers.
 * @param nb_channels Number of channels to be processed
 * @param ratio_num Numerator of the sampling rate ratio
 * @param ratio_den Denominator of the sampling rate ratio
 * @param in_rate Input sampling rate rounded to the nearest integer (in Hz).
 * @param out_rate Output sampling rate rounded to the nearest integer (in Hz).
 * @param quality Resampling quality between 0 and 10, where 0 has poor quality
 * and 10 has very high quality.
 * @return Newly created resampler state
 * @retval NULL Error: not enough memory
 */
AIDResamplerState *aid_resampler_init_frac(aid_uint32_t nb_channels, 
                                               aid_uint32_t ratio_num, 
                                               aid_uint32_t ratio_den, 
                                               aid_uint32_t in_rate, 
                                               aid_uint32_t out_rate, 
                                               int quality,
                                               int *err);

/** Destroy a resampler state.
 * @param st Resampler state
 */
void aid_resampler_destroy(AIDResamplerState *st);

/** Resample a float array. The input and output buffers must *not* overlap.
 * @param st Resampler state
 * @param channel_index Index of the channel to process for the multi-channel 
 * base (0 otherwise)
 * @param in Input buffer
 * @param in_len Number of input samples in the input buffer. Returns the 
 * number of samples processed
 * @param out Output buffer
 * @param out_len Size of the output buffer. Returns the number of samples written
 */
int aid_resampler_process_float(AIDResamplerState *st, 
                                   aid_uint32_t channel_index, 
                                   const float *in, 
                                   aid_uint32_t *in_len, 
                                   float *out, 
                                   aid_uint32_t *out_len);

/** Resample an int array. The input and output buffers must *not* overlap.
 * @param st Resampler state
 * @param channel_index Index of the channel to process for the multi-channel 
 * base (0 otherwise)
 * @param in Input buffer
 * @param in_len Number of input samples in the input buffer. Returns the number
 * of samples processed
 * @param out Output buffer
 * @param out_len Size of the output buffer. Returns the number of samples written
 */
int aid_resampler_process_int(AIDResamplerState *st, 
                                 aid_uint32_t channel_index, 
                                 const aid_int16_t *in, 
                                 aid_uint32_t *in_len, 
                                 aid_int16_t *out, 
                                 aid_uint32_t *out_len);

/** Resample an interleaved float array. The input and output buffers must *not* overlap.
 * @param st Resampler state
 * @param in Input buffer
 * @param in_len Number of input samples in the input buffer. Returns the number
 * of samples processed. This is all per-channel.
 * @param out Output buffer
 * @param out_len Size of the output buffer. Returns the number of samples written.
 * This is all per-channel.
 */
int aid_resampler_process_interleaved_float(AIDResamplerState *st, 
                                               const float *in, 
                                               aid_uint32_t *in_len, 
                                               float *out, 
                                               aid_uint32_t *out_len);

/** Resample an interleaved int array. The input and output buffers must *not* overlap.
 * @param st Resampler state
 * @param in Input buffer
 * @param in_len Number of input samples in the input buffer. Returns the number
 * of samples processed. This is all per-channel.
 * @param out Output buffer
 * @param out_len Size of the output buffer. Returns the number of samples written.
 * This is all per-channel.
 */
int aid_resampler_process_interleaved_int(AIDResamplerState *st, 
                                             const aid_int16_t *in, 
                                             aid_uint32_t *in_len, 
                                             aid_int16_t *out, 
                                             aid_uint32_t *out_len);

/** Set (change) the input/output sampling rates (integer value).
 * @param st Resampler state
 * @param in_rate Input sampling rate (integer number of Hz).
 * @param out_rate Output sampling rate (integer number of Hz).
 */
int aid_resampler_set_rate(AIDResamplerState *st, 
                              aid_uint32_t in_rate, 
                              aid_uint32_t out_rate);

/** Get the current input/output sampling rates (integer value).
 * @param st Resampler state
 * @param in_rate Input sampling rate (integer number of Hz) copied.
 * @param out_rate Output sampling rate (integer number of Hz) copied.
 */
void aid_resampler_get_rate(AIDResamplerState *st, 
                              aid_uint32_t *in_rate, 
                              aid_uint32_t *out_rate);

/** Set (change) the input/output sampling rates and resampling ratio 
 * (fractional values in Hz supported).
 * @param st Resampler state
 * @param ratio_num Numerator of the sampling rate ratio
 * @param ratio_den Denominator of the sampling rate ratio
 * @param in_rate Input sampling rate rounded to the nearest integer (in Hz).
 * @param out_rate Output sampling rate rounded to the nearest integer (in Hz).
 */
int aid_resampler_set_rate_frac(AIDResamplerState *st, 
                                   aid_uint32_t ratio_num, 
                                   aid_uint32_t ratio_den, 
                                   aid_uint32_t in_rate, 
                                   aid_uint32_t out_rate);

/** Get the current resampling ratio. This will be reduced to the least
 * common denominator.
 * @param st Resampler state
 * @param ratio_num Numerator of the sampling rate ratio copied
 * @param ratio_den Denominator of the sampling rate ratio copied
 */
void aid_resampler_get_ratio(AIDResamplerState *st, 
                               aid_uint32_t *ratio_num, 
                               aid_uint32_t *ratio_den);

/** Set (change) the conversion quality.
 * @param st Resampler state
 * @param quality Resampling quality between 0 and 10, where 0 has poor 
 * quality and 10 has very high quality.
 */
int aid_resampler_set_quality(AIDResamplerState *st, 
                                 int quality);

/** Get the conversion quality.
 * @param st Resampler state
 * @param quality Resampling quality between 0 and 10, where 0 has poor 
 * quality and 10 has very high quality.
 */
void aid_resampler_get_quality(AIDResamplerState *st, 
                                 int *quality);

/** Set (change) the input stride.
 * @param st Resampler state
 * @param stride Input stride
 */
void aid_resampler_set_input_stride(AIDResamplerState *st, 
                                      aid_uint32_t stride);

/** Get the input stride.
 * @param st Resampler state
 * @param stride Input stride copied
 */
void aid_resampler_get_input_stride(AIDResamplerState *st, 
                                      aid_uint32_t *stride);

/** Set (change) the output stride.
 * @param st Resampler state
 * @param stride Output stride
 */
void aid_resampler_set_output_stride(AIDResamplerState *st, 
                                      aid_uint32_t stride);

/** Get the output stride.
 * @param st Resampler state copied
 * @param stride Output stride
 */
void aid_resampler_get_output_stride(AIDResamplerState *st, 
                                      aid_uint32_t *stride);

/** Get the latency introduced by the resampler measured in input samples.
 * @param st Resampler state
 */
int aid_resampler_get_input_latency(AIDResamplerState *st);

/** Get the latency introduced by the resampler measured in output samples.
 * @param st Resampler state
 */
int aid_resampler_get_output_latency(AIDResamplerState *st);

/** Make sure that the first samples to go out of the resamplers don't have 
 * leading zeros. This is only useful before starting to use a newly created 
 * resampler. It is recommended to use that when resampling an audio file, as
 * it will generate a file with the same length. For real-time processing,
 * it is probably easier not to use this call (so that the output duration
 * is the same for the first frame).
 * @param st Resampler state
 */
int aid_resampler_skip_zeros(AIDResamplerState *st);

/** Reset a resampler so a new (unrelated) stream can be processed.
 * @param st Resampler state
 */
int aid_resampler_reset_mem(AIDResamplerState *st);

/** Returns the English meaning for an error code
 * @param err Error code
 * @return English string
 */
const char *aid_resampler_strerror(int err);

#ifdef __cplusplus
}
#endif

#endif
