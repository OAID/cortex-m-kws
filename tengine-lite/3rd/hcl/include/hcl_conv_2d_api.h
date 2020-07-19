/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __HCL_CONV_2D_API_H__
#define __HCL_CONV_2D_API_H__

#include <stdint.h>

#ifndef __HCL_API_H__
#error __FILE__ should not be included outside hcl_api.h
#endif

/* -------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

typedef void* hcl_conv_2d_t;

/*!
 * @brief Create a conv 2d operator.
 *
 * @param The pointer to the instance.
 * @param The height of filter.
 * @param The width of filter.
 * @param The height of stride.
 * @param The width of stride.
 * @param The top padding rows.
 * @param The bottom padding rows
 * @param The left padding columns.
 * @param The right padding columns.
 * @param The height of dilation.
 * @param The width of dilation.
 * @param The input channel.
 * @param The output channel.
 * @param The group number.
 *
 * @return The pointer to the new conv 2d operator.
 */
hcl_conv_2d_t hcl_create_conv_2d(hcl_instance_t instance, int kernel_h, int kernel_w, int stride_h, int stride_w,
                                 int pad_h0, int pad_h1, int pad_w0, int pad_w1, int dilation_h, int dilation_w,
                                 int input_channel, int output_channel, int group);

/*!
 * @brief Release the conv 2d operator.
 * @param The pointer to the conv 2d operator.
 */
void hcl_release_conv_2d(hcl_conv_2d_t conv_op);

/*!
 * @brief  set the dev to run conv
 *
 *
 * @param The pointer to the conv 2d operator.
 * @param dev:  dev_type, could be HCL_DEV_CPU/HCL_DEV_GPU
 *
 * @return 0: success, -1: failure
 */

int hcl_conv_2d_bind_dev(hcl_conv_2d_t conv_op, int dev);

/*!
 * @brief Configurate the conv 2d operator.
 *
 * @param The pointer to the conv 2d operator.
 * @param The dynamic shape:
 *         0 : DYNAMIC_SHAPE_NONE
 *         1 : DYNAMIC_SHAPE_FIXED_HW
 *         2 : DYNAMIC_SHAPE_GENERIC
 * @param The relu slope.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_config_dynamic(hcl_conv_2d_t conv_op, int dynamic_shape);

/*!
 * @brief Configurate the conv 2d operator.
 * @param The pointer to the conv 2d operator.
 * @param The relu method:
 *         0 : RELU_NOT_FUSED
 *         1 : RELU_FUSED
 *         2 : RELU6_FUSED
 *         3 : LEAKY_RELU_FUSED
 * @param The relu slope.
 *
 * @return  0 : Success  -1 : Failure
 */

int hcl_conv_2d_config_relu_fuse(hcl_conv_2d_t conv_op, int relu_method, float relu_slope);

/*!
 * @brief Set the input data area to the conv 2d operator.
 *        If input or input_shape is NULL, the counterpart in conv_op will not be updated
 *
 * @param The pointer to the conv 2d operator.
 * @param The pointer to the input data.
 * @param The input shape(NCHW).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_set_input(hcl_conv_2d_t conv_op, const void* input, const int* input_shape);

/*!
 * @brief Set the bias data area to the conv 2d operator.
 *
 * @param The pointer to the conv 2d operator.
 * @param The pointer to the bias data area.
 * @param The bias shape
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_set_bias(hcl_conv_2d_t conv_op, const void* bias, int bias_shape);

/*!
 * @brief Set the filter data area to the conv 2d operator.
 *
 * @param The pointer to the conv 2d operator.
 * @param The pointer to the filter data area.
 * @param The filter shape(NCHW).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_set_filter(hcl_conv_2d_t conv_op, const void* filter, const int* filter_shape);

/*!
 * @brief Set the output data area to the conv 2d operator.
 *        If output or output_shape is NULL, the counterpart in conv_op will not be updated
 *
 * @param The pointer to the conv 2d operator.
 * @param The pointer to the output data area.
 * @param The output shape(NCHW).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_set_output(hcl_conv_2d_t conv_op, void* output, const int* output_shape);

/*!
 * @brief Get the address of output buffer
 *
 * @param The pointer to the conv 2d operator
 *
 * @return the address of output buffer
 */

void* hcl_conv_2d_get_output(hcl_conv_2d_t conv_op);

/*!
 * @brief Get the size of conv 2d shared memory.
 *
 * @param The pointer to the conv 2d operator.
 * @return   Bytes needed
 */
unsigned int hcl_conv_2d_get_shared_mem_size(hcl_conv_2d_t conv_op);

/*!
 * @brief Set the shared memory buffer to the conv 2d operator.
 *
 * @param The pointer to the conv 2d operator.
 * @param The pointer to the shared memory buffer.
 * @param The size of the shared memory buffer.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_set_shared_mem(hcl_conv_2d_t conv_op, void* mem, int size);

/*!
 * @brief Get the size of conv 2d private memory.
 *
 * @param The pointer to the conv 2d operator.
 *
 * @return  Bytes needed
 */
unsigned int hcl_conv_2d_get_private_mem_size(hcl_conv_2d_t conv_op);

/*!
 * @brief Set the private memory buffer to the conv 2d operator.
 *
 * @param The pointer to the conv 2d operator.
 * @param The pointer to the private memory buffer.
 * @param The size of the private memory buffer.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_set_private_mem(hcl_conv_2d_t conv_op, void* mem, int size);

/*!
 * @brief Prerun the conv 2d operator.
 *
 * @param The pointer to the conv 2d operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_prerun(hcl_conv_2d_t conv_op);

/*!
 * @brief check if filter memory can be released or not after prerun
 *
 * @param conv_op: the pointer to conv 2d operator
 *
 * @return 0: can't be released
 *              1: can be released
 */

int hcl_conv_2d_filter_reclaimable(hcl_conv_2d_t conv_op);

/*!
 * @brief check if bias memory can be released or not after prerun
 *
 * @param conv_op: the pointer to conv 2d operator
 *
 * @return 0: can't be released
 *         1: can be released
 */

int hcl_conv_2d_bias_reclaimable(hcl_conv_2d_t conv_op);

/*!
 * @brief called when input shape has been changed after prerun
 *
 * @param The pointer to the conv 2d operator
 *
 * @return  0 : Success  -1 : Failure
 */

int hcl_conv_2d_reshape(hcl_conv_2d_t conv_op);

/*!
 * @brief Run the conv 2d operator.
 *
 * @param The pointer to the conv 2d operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_run(hcl_conv_2d_t conv_op);

/*!
 * @brief wait the op run done
 *        only useful for non-CPU device
 *
 * @param The pointer to the conv 2d operator.
 *
 * @return  0 : Success  -1 : try again
 */

int hcl_conv_2d_wait_done(hcl_conv_2d_t conv_op);

/*!
 * @brief Postrun the conv 2d operator.
 *
 * @param The pointer to the conv 2d operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_conv_2d_postrun(hcl_conv_2d_t conv_op);

/* Advanced APIs for conv_2d */

/*!
 * @brief Set the data type of components.
 *				 This function must be called before prerun() and get/set mem family
 *
 * @param conv_op: The pointer to the conv 2d operator.
 * @param i_type: input data type
 * @param o_type: output_data_type
 * @param f_type: filter_data_type
 * @param b_type: bias_data_type
 *
 * @return  0 : Success
 *         -1 : Failure, try to set the data type in advance
 */

int hcl_conv_2d_config_data_type(hcl_conv_2d_t conv_op, int i_type, int o_type, int f_type, int b_type);

int hcl_conv_2d_config_data_elemsize(hcl_conv_2d_t conv_op, int i_size, int o_size, int f_size, int b_size);

/*!
 * @brief Set the kernel method to do the core computing
 *				 This function must be called before prerun() and get/set mem family
 *
 * @param conv_op: The pointer to the conv 2d operator.
 * @param k_method: Could be HCL_KERNEL_F32/HCL_KERENEL_F16/HCL_KERNEL_INT8/HCL_KERNEL_INT8_CHANNEL
 * @param layout:  HCL_LAYOUT_NCHW or HCL_LAYOUT_NHWC
 *
 * @return  0 : Success
 *         -1 : Failure, try to set the data type in advance
 */

int hcl_conv_2d_config_kernel(hcl_conv_2d_t conv_op, int k_method, int layout);

/* Interface used for int8
 *    It is to pass the scale/bias to the kernel
 *
 */

int hcl_conv_2d_config_set_input_quant_param(hcl_conv_2d_t conv_op, const float* scale, const int* zero, int size);
int hcl_conv_2d_config_set_output_quant_param(hcl_conv_2d_t conv_op, const float* scale, const int* zero, int size);
int hcl_conv_2d_config_set_filter_quant_param(hcl_conv_2d_t conv_op, const float* scale, const int* zero, int size);
int hcl_conv_2d_config_set_bias_quant_param(hcl_conv_2d_t conv_op, const float* scale, const int* zero, int size);

/* Interface to support outer AI Framework to do mix precison computing */
/*!
 * @brief return the support kernel methods
 *
 * @param k_method: pointer an int array to hold the returned data
 * @param k_method_number:  in and out parameter.
 *                          in - pass the number of int of array k_method refers
 *                          out - the number returned
 * @return 0: Success  -1: Failure
 */

int hcl_conv_2d_supported_kernel_method(hcl_conv_2d_t conv_op, int* k_method, int* k_method_number);

/* For a given kernel method, return the supported data type of inputs and outputs

   Returned types are filled in the int array provided by caller.
   Preferred data type are filled first
*/

int hcl_conv_2d_supported_intput_type(hcl_conv_2d_t conv_op, int k_method, int* data_type, int* number);
int hcl_conv_2d_supported_output_type(hcl_conv_2d_t conv_op, int k_method, int* data_type, int* number);
int hcl_conv_2d_supported_filter_type(hcl_conv_2d_t conv_op, int k_method, int* data_type, int* number);
int hcl_conv_2d_supported_bias_type(hcl_conv_2d_t conv_op, int k_method, int* data_type, int* number);

void* hcl_conv_2d_get_output_dev_mem(hcl_conv_2d_t conv_op);

int hcl_conv_2d_get_output_dev_type(hcl_conv_2d_t conv_op);

void hcl_conv_2d_set_input_dev(hcl_conv_2d_t conv_op, int dev_type, void* dev_mem);

#ifdef __cplusplus
}
#endif

#endif    // __HCL_CONV_2D_API_H__
