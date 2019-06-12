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

int32_t frame_len_padded = 0;
float * frame = NULL;
float * buffer = NULL;
float * mel_energies = NULL;
float * window_func = NULL;
int32_t * fbank_filter_first = NULL;
int32_t * fbank_filter_last = NULL;
float ** mel_fbank = NULL;
float * dct_matrix = NULL;
arm_rfft_fast_instance_f32 * rfft = NULL;


static inline float InverseMelScale(float mel_freq) {
  return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
}

static inline float MelScale(float freq) {
  return 1127.0f * logf (1.0f + freq / 700.0f);
}

void MFCC_init()
{

  // Round-up to nearest power of 2.
  frame_len_padded = pow(2,ceil((log(FRAME_LEN)/log(2))));
  
  //printf("frame_len_padded: %d\n", frame_len_padded);
  
  frame = (float*)calloc(frame_len_padded, sizeof(float));
  buffer = (float*)calloc(frame_len_padded, sizeof(float));
  mel_energies = (float*)calloc(NUM_FBANK_BINS, sizeof(float));

  //create window function
  window_func = (float*)calloc(FRAME_LEN, sizeof(float));
  for (int i = 0; i < FRAME_LEN; i++)
    window_func[i] = 0.5 - 0.5*cos(M_2PI * ((float)i) / (FRAME_LEN));

  //create mel filterbank
  fbank_filter_first = (int32_t*)calloc(NUM_FBANK_BINS, sizeof(int32_t));
  fbank_filter_last = (int32_t*)calloc(NUM_FBANK_BINS, sizeof(int32_t));
  mel_fbank = MFCC_create_mel_fbank();
  
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
  free(fbank_filter_first);
  free(fbank_filter_last);
  free(dct_matrix);
  free(rfft);
  for(int i=0;i<NUM_FBANK_BINS;i++)
    free(mel_fbank[i]);
  free(mel_fbank);
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

float ** MFCC_create_mel_fbank() {

  int32_t bin, i;

  int32_t num_fft_bins = frame_len_padded/2;
  float fft_bin_width = ((float)SAMP_FREQ) / frame_len_padded;
  float mel_low_freq = MelScale(MEL_LOW_FREQ);
  float mel_high_freq = MelScale(MEL_HIGH_FREQ); 
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS+1);

  float *this_bin = (float*)calloc(num_fft_bins, sizeof(float));

  float ** mel_fbank = (float**)calloc(NUM_FBANK_BINS, sizeof(float*));

  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {

    float left_mel = mel_low_freq + bin * mel_freq_delta;
    float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
    float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    int32_t first_index = -1, last_index = -1;

    for (i = 0; i < num_fft_bins; i++) {

      float freq = (fft_bin_width * i);  // center freq of this fft bin.
      float mel = MelScale(freq);
      this_bin[i] = 0.0;

      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel-mel) / (right_mel-center_mel);
        }
        this_bin[i] = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }

    fbank_filter_first[bin] = first_index;
    fbank_filter_last[bin] = last_index;
    mel_fbank[bin] = (float*)calloc(last_index-first_index+1, sizeof(float)); 

    int32_t j = 0;
    //copy the part we care about
    for (i = first_index; i <= last_index; i++) {
      mel_fbank[bin][j++] = this_bin[i];
    }
  }
  free(this_bin);
  return mel_fbank;
}

void MFCC_mfcc_compute(const int16_t * data, uint16_t dec_bits, int8_t * mfcc_out) 
{
//printf("enter MFCC_mfcc_compute\n");
  int32_t i, j, bin;

  //TensorFlow way of normalizing .wav data to (-1,1)
  for (i = 0; i < FRAME_LEN; i++) {
    frame[i] = (float)data[i]/(1<<15); 
  }
  //Fill up remaining with zeros
  memset(&frame[FRAME_LEN], 0, sizeof(float) * (frame_len_padded-FRAME_LEN));

  for (i = 0; i < FRAME_LEN; i++) {
    frame[i] *= window_func[i];
  }
//printf("start arm_rfft_fast_f32\n");
  //Compute FFT
  arm_rfft_fast_f32(rfft, frame, buffer, 0);
//printf("finish arm_rfft_fast_f32\n");
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
  
//printf("start mel filterbanks\n"); 
  float sqrt_data;
  //Apply mel filterbanks
  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
    j = 0;
    float mel_energy = 0;
    int32_t first_index = fbank_filter_first[bin];
    int32_t last_index = fbank_filter_last[bin];
    for (i = first_index; i <= last_index; i++) {
      arm_sqrt_f32(buffer[i],&sqrt_data);
      mel_energy += (sqrt_data) * mel_fbank[bin][j++];
    }
    mel_energies[bin] = mel_energy;
	
    //avoid log of zero
    if (mel_energy == 0.0)
      mel_energies[bin] = FLT_MIN;
  }
//printf("finish mel filterbanks\n");
  
  //Take log
  for (bin = 0; bin < NUM_FBANK_BINS; bin++)
    mel_energies[bin] = logf(mel_energies[bin]);


  //Take DCT. Uses matrix mul.
  for (i = 0; i < NUM_MFCC_COEFFS; i++) {
    float sum = 0.0;
    for (j = 0; j < NUM_FBANK_BINS; j++) {
      sum += dct_matrix[i*NUM_FBANK_BINS+j] * mel_energies[j];
    }

    //Input is Qx.dec_bits (from quantization step)
    sum *= (0x1<<dec_bits);
    sum = round(sum); 
    if(sum >= 127)
      mfcc_out[i] = 127;
    else if(sum <= -128)
      mfcc_out[i] = -128;
    else
      mfcc_out[i] = sum; 
  }
  
//printf("finish mel MFCC_mfcc_compute\n");
}
