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
#ifndef __HCL_FC_API_H__
#define __HCL_FC_API_H__

#ifndef __HCL_API_H__
#error __FILE__ should not be included outside hcl_api.h
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* hcl_fc_t;

/*!
 * @brief Create a fc operator.
 *
 * @param The pointer to the instance.
 * @param The hidden number.
 * @param The output number.
 *
 * @return The pointer to the new fc operator.
 */
hcl_fc_t hcl_create_fc(hcl_instance_t instance, int hidden_number, int output_number);

/*!
 * @brief Release the fc operator.
 * @param The pointer to the fc operator.
 */
void hcl_release_fc(hcl_fc_t fc_op);

/*!
 * @brief Configurate the fc operator.
 *
 * @param The pointer to the fc operator.
 * @param The dynamic shape:
 *         0 : DYNAMIC_SHAPE_NONE
 *         1 : DYNAMIC_SHAPE_FIXED_HW
 *         2 : DYNAMIC_SHAPE_GENERIC
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_config_dynamic(hcl_fc_t fc_op, int dynamic_shape);

/*!
 * @brief Configurate the fc operator.
 *
 * @param The pointer to the fc operator.
 * @param The relu method:
 *         0 : RELU_NOT_FUSED
 *         1 : RELU_FUSED
 *         2 : RELU6_FUSED
 *         3 : LEAKY_RELU_FUSED
 * @param The relu slope.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_config_relu_fused(hcl_fc_t fc_op, int relu_method, float relu_slope);

/*!
 * @brief Set the input data area to the fc operator.
 *
 * @param The pointer to the fc operator.
 * @param The pointer to the input data area.
 * @param The input shape(2 dimensions).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_set_input(hcl_fc_t fc_op, const void* input, const int* input_shape);

/*!
 * @brief Set the bias data area to the fc operator.
 *
 * @param The pointer to the fc operator.
 * @param The pointer to the bias data area.
 * @param The bias shape(1 dimension).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_set_bias(hcl_fc_t fc_op, const void* bias, const int bias_shape);

/*!
 * @brief Set the weight data area to the fc operator.
 *
 * @param The pointer to the fc operator.
 * @param The pointer to the weight data area.
 * @param The weight shape(2 dimensions).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_set_weight(hcl_fc_t fc_op, const void* weight, const int* weight_shape);

/*!
 * @brief Set the output data area to the fc operator.
 *
 * @param The pointer to the fc operator.
 * @param The pointer to the output data area.
 * @param The output shape(2 dimensions).
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_set_output(hcl_fc_t fc_op, void* output, const int* output_shape);

/*!
 * @brief Get the size of fc shared memory.
 *
 * @param The pointer to the fc operator.
 *
 * @return  >=0: Bytes needed
 *          -1 : Failure
 */
int hcl_fc_get_shared_mem_size(hcl_fc_t fc_op);

/*!
 * @brief Set the shared memory buffer to the fc operator.
 *
 * @param The pointer to the fc operator.
 * @param The pointer to the shared memory buffer.
 * @param The size of the shared memory buffer.
 *
 * @return 0 : Success  -1 : Failure
 */
int hcl_fc_set_shared_mem(hcl_fc_t fc_op, void* mem, int size);

/*!
 * @brief Get the size of fc private memory.
 *
 * @param The pointer to the fc operator.
 *
 * @return  >=0: Bytes needed
 *          -1 : Failure
 */
int hcl_fc_get_private_mem_size(hcl_fc_t fc_op);

/*!
 * @brief Set the private memory buffer to the fc operator.
 *
 * @param The pointer to the fc operator.
 * @param The pointer to the private memory buffer.
 * @param The size of the private memory buffer.
 *
 *  @return 0 : Success  -1 : Failure
 */
int hcl_fc_set_private_mem(hcl_fc_t fc_op, void* mem, int size);

/*!
 * @brief Prerun the fc operator.
 *
 * @param The pointer to the fc operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_prerun(hcl_fc_t fc_op);

/*!
 * @brief Run the fc operator.
 *
 * @param The pointer to the fc operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_run(hcl_fc_t fc_op);

/*!
 * @brief Postrun the fc operator.
 *
 * @param The pointer to the fc operator.
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_fc_postrun(hcl_fc_t fc_op);

/*!
 * @brief Set the kernel method to do the core computing
 *				 This function must be called before prerun() and get/set mem family
 *
 * @param The pointer to the fc operator.
 * @param k_method: Could be HCL_KERNEL_F32/HCL_KERENEL_F16/HCL_KERNEL_INT8/HCL_KERNEL_INT8_CHANNEL
 * @param dev: HCL_DEV_CPU/HCL_DEV_GPU/HCL_DEV_DSP
 * @param mt_enabled: should enable mt mode
 *
 * @return  0 : Success
 *         -1 : Failure, try to set the data type in advance
 */
int hcl_fc_config_op(hcl_fc_t hcl_op, int k_method, int dev, int mt_enabled);

/*!
 * @brief return the support kernel methods
 *
 * @param The pointer to the fc operator.
 * @param k_method: pointer an int array to hold the returned data
 * @param k_method_number:  in and out parameter.
 *                          in - pass the number of int of array k_method refers
 *                          out - the number returned
 * @return 0: Success  -1: Failure
 */

int hcl_fc_supported_kernel_method(hcl_fc_t fc_op, int* k_method, int* k_method_number);

/*!
 * @brief Set the data type of components.
 *				 This function must be called before prerun() and get/set mem family
 *
 * @param The pointer to the fc operator.
 * @param i_type: input data type
 * @param o_type: output_data_type
 * @param w_type: weight_data_type
 * @param b_type: bias_data_type
 *
 * @return  0 : Success
 *         -1 : Failure, data conflict with kernel_method or call it too late
 */

int hcl_fc_config_data_type(hcl_fc_t fc_op, int i_type, int o_type, int w_type, int b_type);

void* hcl_fc_get_output(hcl_fc_t fc_op);

#ifdef __cplusplus
}
#endif

#endif    //
