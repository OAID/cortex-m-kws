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
#ifndef __HCL_API_H__
#define __HCL_API_H__

#include <stdio.h>

#include "cpu_model.h"

/* Dynamic shape */
#define DYNAMIC_SHAPE_NONE 0 /* fixed shape, not dynamic */
#define DYNAMIC_SHAPE_FIXED_HW 1 /* dynamic shape, fixed hw  */
#define DYNAMIC_SHAPE_GENERIC 2 /* generic dynamic shape    */

/* Relu method */
#define RELU_NOT_FUSED -1
#define RELU_FUSED 0
#define RELU6_FUSED 6
#define LEAKY_RELU_FUSED -1

/* Data type */
#define HCL_DT_FP32 0
#define HCL_DT_FP16 1
#define HCL_DT_INT8 2 /* -128-127 */
#define HCL_DT_UINT8 3 /* 0-255 */
#define HCL_DT_INT32 4 /* scale * quant */

/* Kernel method */
#define HCL_KERNEL_FP32 0
#define HCL_KERNEL_FP16 1
#define HCL_KERNEL_INT8 2
#define HCL_KERNEL_INT8_CHANNEL 3

/* Layout */
#define HCL_LAYOUT_NCHW 0
#define HCL_LAYOUT_NHWC 1

/* Quant type */
#define HCL_QUANT_NONE 0
#define HCL_QUANT_INT8 1
#define HCL_QUANT_INT8_CHANNEL 2

#define MAX_DIM_NUM 4

/* device type */
#define HCL_DEV_CPU 0
#define HCL_DEV_GPU 1
#define HCL_DEV_DSP 2

typedef void* (*mem_alloc_t)(size_t);
typedef void (*mem_free_t)(void*);
typedef void* hcl_instance_t;

#ifdef CONFIG_MT_SUPPORT
#include <hcl_api_mt.h>
#endif

/* -------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @brief Initialize the hcl library.
 *
 * @param request_version: pointer to the string of required hcl version(e.g. "0.1.0").
 * @param soc_name: if not NULL, HCL will set the available computing resources according to predefined soc
 *                  if NULL, HCL will probe the computing resources
 *
 * @return  0 : Success  -1 : Failure
 */
int hcl_init_library(const int* cpu_list, const int* cpu_model_list, int list_number, int master_idx);

/*!
 *
 * @brief Release the hcl library.
 */
void hcl_release_library(void);

/*!
 * @brief Request specific version from run-time environment
 *
 * @return  0 success, -1 failure
 */
int hcl_request_version(const char* version);

/*!
 * @brief Get the version of present hcl library.
 *
 * @return The string of present hcl version.
 */
const char* hcl_get_library_version(void);

/*!
 * @brief Create an instance.
 *        Instance is a resource container for all operators to be executed.
 *        By default, all available computing resources(CPU/GPU/DSP) in system
 *        are available for an instance.
 *        By calling hcl_config_instance_XXX to limit the resource an instance can use
 *
 * @param The function pointer of memory alloc
 * @param The funtcion pointer of memory free
 *
 * @return The pointer to the new instance.
 */
hcl_instance_t hcl_create_instance(mem_alloc_t mem_alloc, mem_free_t mem_free);

/*!
 * @brief Release the instance.
 *
 * @param The pointer to the instance.
 */
void hcl_release_instance(hcl_instance_t instance);

/*!
* @brief Set the CPU list an instance can use (Optional)
         If this API is not called, HCL library will decide the CPU to be used.
*
* @param The cpu number to be used.
* @param The list of cpu id to be used.
*
* @return  0 : Success  -1 : Failure
*/

int hcl_config_instance_cpu(hcl_instance_t ins, const int* cpu, int cpu_number);

/*!
 * @brief Set the GPU list an instance can use (Optional)
 *
 *
 * @param The GPU number to be used.
 * @param The list of GPU id to be used.
 *
 * @return  0 : Success  -1 : Failure
 */

int hcl_config_instance_gpu(hcl_instance_t ins, const int* gpu, int cpu_number);

/*!
 * @brief Set the DSP list an instance can use
 *
 * @param The DSP number to be used.
 * @param The list of DSP id to be used.
 *
 * @return  0 : Success  -1 : Failure
 */

int hcl_config_instance_dsp(hcl_instance_t ins, const int* dsp, int dsp_number);

#ifdef __cplusplus
}
#endif

/* include operator api files */

#include "hcl_conv_2d_api.h"
#include "hcl_pooling_api.h"
#include "hcl_fc_api.h"

#endif    // __HCL_API_H__
