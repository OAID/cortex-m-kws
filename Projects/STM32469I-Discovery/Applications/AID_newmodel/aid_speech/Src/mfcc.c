/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description: MFCC feature extraction to match with TensorFlow MFCC Op
 */

#include <string.h>
#include <stdio.h>

#include "mfcc.h"
#include "float.h"
#include "stdlib.h"

static int32_t frame_len_padded = 0;
static float * frame = NULL;
static float * buffer = NULL;
static float * mel_energies = NULL;
static float * window_func = NULL;
static float * dct_matrix = NULL;
static arm_rfft_fast_instance_f32 * rfft = NULL;
static float *center_frequencies_ = NULL;
static float *band_mapper_ = NULL;
static float *weights_ = NULL;
static int start_index_ = 0;
static int end_index_ = 0;

static inline float InverseMelScale(float mel_freq) {
  return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
}

static inline float MelScale(float freq) {
  return 1127.0f * logf (1.0f + freq / 700.0f);
}

void MFCC_init()
{

  // Round-up to nearest power of 2.
  frame_len_padded = pow(2,ceil((log(MFCC_FRAME_LEN)/log(2))));
  
  //printf("frame_len_padded: %d\n", frame_len_padded);
  
  frame = (float*)calloc(frame_len_padded, sizeof(float));
  buffer = (float*)calloc(frame_len_padded, sizeof(float));
  mel_energies = (float*)calloc(NUM_FBANK_BINS, sizeof(float));

  //create window function
  window_func = (float*)calloc(MFCC_FRAME_LEN, sizeof(float));
  for (int i = 0; i < MFCC_FRAME_LEN; i++)
    window_func[i] = 0.5 - 0.5*cos(M_2PI * ((float)i) / (MFCC_FRAME_LEN));

  //create mel filterbank implement in tesnorflow 
  //commit 775f42a845353ea8525bc54a2ddb5852acf3c6eb

  // An extra center frequency is computed at the top to get the upper
  // limit on the high side of the final triangular filter.
  center_frequencies_ = (float*)calloc(NUM_FBANK_BINS+1, sizeof(float));
  int32_t num_fft_bins = frame_len_padded/2;
  float mel_low_freq = MelScale(MEL_LOW_FREQ);
  float mel_high_freq = MelScale(MEL_HIGH_FREQ); 
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS+1);
  for (int i = 0; i < NUM_FBANK_BINS + 1; ++i) 
  {
    center_frequencies_[i] = mel_low_freq + (mel_freq_delta * (i + 1));
  } 
  float fft_bin_width = ((float)SAMP_FREQ) / frame_len_padded;
  // Always exclude DC; emulate HTK.
  start_index_ = (int)(1.5 + ((float)MEL_LOW_FREQ / fft_bin_width));
  end_index_ = (int)((float)MEL_HIGH_FREQ / fft_bin_width);

  // Maps the input spectrum bin indices to filter bank channels/indices. For
  // each FFT bin, band_mapper tells us which channel this bin contributes to
  // on the right side of the triangle.  Thus this bin also contributes to the
  // left side of the next channel's triangle response.
  band_mapper_ = (float*)calloc(num_fft_bins+1, sizeof(float));
  int channel = 0;
  for (int i = 0; i < num_fft_bins+1; ++i) {
    float melf = MelScale(i * fft_bin_width);
    if ((i < start_index_) || (i > end_index_)) {
      band_mapper_[i] = -2;  // Indicate an unused Fourier coefficient.
    } else {
      while ((center_frequencies_[channel] < melf) &&
             (channel < NUM_FBANK_BINS)) {
        ++channel;
      }
      band_mapper_[i] = channel - 1;  // Can be == -1
    }
  }

  // Create the weighting functions to taper the band edges.  The contribution
  // of any one FFT bin is based on its distance along the continuum between two
  // mel-channel center frequencies.  This bin contributes weights_[i] to the
  // current channel and 1-weights_[i] to the next channel.
  weights_ = (float*)calloc(num_fft_bins+1, sizeof(float));
  for (int i = 0; i < num_fft_bins+1; ++i) {
    channel = band_mapper_[i];
    if ((i < start_index_) || (i > end_index_)) {
      weights_[i] = 0.0;
    } else {
      if (channel >= 0) {
        weights_[i] =
            (center_frequencies_[channel + 1] - MelScale(i * fft_bin_width)) /
            (center_frequencies_[channel + 1] - center_frequencies_[channel]);
      } else {
        weights_[i] = (center_frequencies_[0] - MelScale(i * fft_bin_width)) /
                      (center_frequencies_[0] - mel_low_freq);
      }
    }
  }

  //create DCT matrix
  dct_matrix = MFCC_create_dct_matrix(NUM_FBANK_BINS, NUM_MFCC_COEFFS);

  //initialize FFT
  rfft = (arm_rfft_fast_instance_f32 *)calloc(1, sizeof(arm_rfft_fast_instance_f32));
  arm_rfft_fast_init_f32(rfft, frame_len_padded);

}

void MFCC_delete()
{
  free(frame);
  free(buffer);
  free(mel_energies);
  free(window_func);
  free(dct_matrix);
  free(rfft);
}

float * MFCC_create_dct_matrix(int32_t input_length, int32_t coefficient_count)
{
  int32_t k, n;
  float * M = (float*)calloc(input_length*coefficient_count, sizeof(float));
  float normalizer;
  arm_sqrt_f32(2.0/(float)input_length,&normalizer);
  for (k = 0; k < coefficient_count; k++) {
    for (n = 0; n < input_length; n++) {
      M[k*input_length+n] = normalizer * cos( ((double)M_2PI)/2/input_length * (n + 0.5) * k );
    }
  }
  return M;
}



// Compute the mel spectrum from the squared-magnitude FFT input by taking the
// square root, then summing FFT magnitudes under triangular integration windows
// whose widths increase with frequency.
void MFCC_mfcc_compute(const int16_t * data, float * mfcc_out) 
{
//printf("enter MFCC_mfcc_compute\n");
  int32_t i, j, bin;

  //TensorFlow way of normalizing .wav data to (-1,1)
  for (i = 0; i < MFCC_FRAME_LEN; i++) {
    frame[i] = (float)data[i]/(1<<15); 
  }

  //Fill up remaining with zeros
  memset(&frame[MFCC_FRAME_LEN], 0, sizeof(float) * (frame_len_padded-MFCC_FRAME_LEN));

  for (i = 0; i < MFCC_FRAME_LEN; i++) {
    frame[i] *= window_func[i];
  }

  //Compute FFT
  arm_rfft_fast_f32(rfft, frame, buffer, 0);

  //Convert to power spectrum
  //frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int32_t half_dim = frame_len_padded/2;
  float first_energy = buffer[0] * buffer[0],
        last_energy =  buffer[1] * buffer[1];  // handle this special case
		
  
  for (i = 1; i < half_dim; i++) {
    float real = buffer[i*2], im = buffer[i*2 + 1];
    buffer[i] = real*real + im*im;
  }
  buffer[0] = first_energy;
  buffer[half_dim] = last_energy;  
  memset(mel_energies, 0, sizeof(float)*NUM_FBANK_BINS);
  for (int i = start_index_; i <= end_index_; i++) 
  { // For each FFT bin
    float spec_val;
    arm_sqrt_f32(buffer[i],&spec_val);
    float weighted = spec_val * weights_[i];
    int channel = band_mapper_[i];
    if (channel >= 0)
    {
      mel_energies[channel] += weighted;  // Right side of triangle, downward slope
    }
    channel++;
    if (channel < NUM_FBANK_BINS)
    {
      mel_energies[channel] += spec_val - weighted;  // Left side of triangle
    }
  }
  
  //Take log
  for (bin = 0; bin < NUM_FBANK_BINS; bin++)
  {
    if(mel_energies[bin] == 0.0)
    {
      mel_energies[bin] = FLT_MIN;
    }
    mel_energies[bin] = logf(mel_energies[bin]);
  }

  //Take DCT. Uses matrix mul.

  for (i = 0; i < NUM_MFCC_COEFFS; i++) {
    float sum = 0.0;
    for (j = 0; j < NUM_FBANK_BINS; j++) {
      sum += dct_matrix[i*NUM_FBANK_BINS+j] * mel_energies[j];
    }
    mfcc_out[i] = sum;
  }
//printf("finish mel MFCC_mfcc_compute\n");
}
