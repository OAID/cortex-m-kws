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
#ifndef __HCL_POOLING_API_H__
#define __HCL_POOLING_API_H__

#ifndef __HCL_API_H__
#error __FILE__ should not be included outside hcl_api.h
#endif

/* Pooling method */
#define HCL_POOL_MAX 0 /* Max pooling     */
#define HCL_POOL_AVG 1 /* Average pooling */

#ifdef __cplusplus
extern "C" {
#endif

typedef void* hcl_pooling_t;

/*!
* @brief Create a pooling operator.
*
* @param The pointer to the instance.
* @param The pooling method:
         0 : Max pooling
         1 : Average pooling
         2 : Random pooling
* @param The height of kernel.
* @param The width of kernel.
* @param The height of stride.
* @param The width of stride.
* @param The top padding rows.
* @param The bottom padding rows
* @param The left padding columns.
* @param The right padding columns.
* @param The global flag:
         0 : not global
         1 : is global
*
* @return The pointer to the new pooling operator.
*/
hcl_pooling_t hcl_create_pooling(hcl_instance_t instance, int pool_method, int kernel_h, int kernel_w, int stride_h,
                                 int stride_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int global);

/*!
 * @brief Release the pooling operator.
 * @param The pointer to the pooling operator.
 */
void hcl_release_pooling(hcl_pooling_t pool_op);

/*!
 * @brief Configurate the pooling operator dynamic attribute
 *
 * @param The pointer to the pooling operator.
 * @param The dynamic shape:
 *         0 : DYNAMIC_SHAPE_NONE
 *         1 : DYNAMIC_SHAPE_FIXED_HW
 *         2 : DYNAMIC_SHAPE_GENERIC
 *
 * @return  0 : Success  -1 : Failure
 */

int hcl_pooling_config_dynamic(hcl_pooling_t pool_op, int dynamic_shape);

/*!
 * @brief Set the input data area to the pooling operator.
 *
 * @param The pointer to the pooling operator.
 * @param The pointer to the input data area.
 * @param The input shape(NCHW).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_pooling_set_input(hcl_pooling_t pool_op, const void* input, const int* input_shape);

/*!
 * @brief Set the output data area to the pooling operator.
 *
 * @param The pointer to the pooling operator.
 * @param The pointer to the output data area.
 * @param The output shape(NCHW).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_pooling_set_output(hcl_pooling_t pool_op, void* output, const int* output_shape);

/*!
 * @brief Prerun the pooling operator.
 *
 * @param The pointer to the pooling operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_pooling_prerun(hcl_pooling_t pool_op);

/*!
 * @brief Run the pooling operator.
 *
 * @param The pointer to the pooling operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_pooling_run(hcl_pooling_t pool_op);

/*!
 * @brief Postrun the pooling operator.
 *
 * @param The pointer to the pooling operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_pooling_postrun(hcl_pooling_t pool_op);

/*!
 * @brief Get the address of output buffer
 *
 * @param The pointer to the conv 2d operator
 *
 * @return the address of output buffer
 */

void* hcl_pooling_get_output(hcl_pooling_t pool_op);

/*!
 * @brief configure the layout and caffe mode
 *
 * @param the pointer to the pooling oeprator
 * @param layout: could be HCL_LAYOUT_NCHW(default) or HCL_LAYOUT_NHWC
 * @param caffe_mode: if calculate the output width using caffe's method
 */

int hcl_pooling_config_op(hcl_pooling_t pool_op, int layout, int caffe_mode, int dev, int mt_enabled);

/*!
 * @brief Set the data type
 *
 * @param The pointer to the pooling operator.
 * @param data_type: Could be HCL_DT_FP32/HCL_DT_FP16/HCL_DT_INT8
 *
 * @return  0 : Success
 *         -1 : Failure
 */

int hcl_pooling_config_data_type(hcl_pooling_t pool_op, int data_type);

/*!
 * @brief return the support data types
 *
 * @param pool_op: The pointer to the pooling operator.
 * @param data_type: pointer an int array to hold the returned data
 * @param number:  in and out parameter.
 *                          in - pass the number of int of array data_type refers
 *                          out - the number returned
 * @return 0: Success  -1: Failure
 */

int hcl_pooling_supported_data_type(hcl_pooling_t pool_op, int* data_type, int* number);

#ifdef __cplusplus
}
#endif

#endif    // __HCL_POOLING_API_H__
