/*
 * This header file is generated automatically, please do not modify.
 */

#ifndef KWS_CNN_H
#define KWS_CNN_H

#include "cnn_weights.h"
#include "arm_nnfunctions.h"
#include "arm_math.h"
#include "mfcc.h"

#define IN_DIM (NUM_FRAMES*NUM_MFCC_COEFFS)
#define INPUT_FEATURE_DIM_W     (10)
#define OUT_DIM 12
#define FIRST_CONV_WT_DIM       (10*10*1*96)
#define FIRST_CONV_BIAS_DIM     (96)
#define SECOND_CONV_WT_DIM      (8*1*96*80)
#define SECOND_CONV_BIAS_DIM    (80)
#define THIRD_CONV_WT_DIM       (4*1*80*72)
#define THIRD_CONV_BIAS_DIM     (72)
#define FOURTH_CONV_WT_DIM      (3*1*72*64)
#define FOURTH_CONV_BIAS_DIM    (64)
#define LINEAR_WT_DIM           (64*8*64)
#define LINEAR_BIAS_DIM         (64)
#define FIRST_FC_WT_DIM         (64*128)
#define FIRST_FC_BIAS_DIM       (128)
#define FINAL_FC_WT_DIM         (128*OUT_DIM)
#define FINAL_FC_BIAS_DIM       (OUT_DIM)

#define TEST_CNN_OPT 0

//time gap between two detect
// 2*2/12.5s = 0.32s
#define TIME_SHIFT (1)

// convelution procesee CONV_DATA_LEN*20 ms data
// must be even
#define CONV_DATA_LEN (TIME_SHIFT*8)


#define SHARED_BUFFER_SIZE  1536

// void aid_DNN_init();
// void aid_DNN_delet();
// void aid_DNN_run(float* in_data, q7_t* out_data);
int inputdata_preprocess(float* in_data, q7_t* out_data);


#endif
