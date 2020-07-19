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
 * Copyright (c) 2019, OPEN AI LAB
 * Author: bhu@openailab.com
 * Date : 2019-09-09
 */

#include <stdio.h>

#include "tiny_graph.h"
#include "tiny_param_generated.h"

#define FIRST_CONV_KERNEL_DIM_H      (10)
#define FIRST_CONV_KERNEL_DIM_W      (10)
#define FIRST_CONV_KERNEL_STRIDE_H   (2)
#define FIRST_CONV_KERNEL_STRIDE_W   (1)
#define FIRST_CONV_KERNEL_C          (96)
#define SECOND_CONV_KERNEL_DIM_H     (8)
#define SECOND_CONV_KERNEL_DIM_W     (1)
#define SECOND_CONV_KERNEL_STRIDE_H  (2)
#define SECOND_CONV_KERNEL_STRIDE_W  (1)
#define SECOND_CONV_KERNEL_C         (80)
#define THIRD_CONV_KERNEL_DIM_H      (4)
#define THIRD_CONV_KERNEL_DIM_W      (1)
#define THIRD_CONV_KERNEL_STRIDE_H   (1)
#define THIRD_CONV_KERNEL_STRIDE_W   (1)
#define THIRD_CONV_KERNEL_C          (72)
#define FOURTH_CONV_KERNEL_DIM_H     (3)
#define FOURTH_CONV_KERNEL_DIM_W     (1)
#define FOURTH_CONV_KERNEL_STRIDE_H  (2)
#define FOURTH_CONV_KERNEL_STRIDE_W  (1)
#define FOURTH_CONV_KERNEL_C         (64)
#define LINEAR_DIM                   (64)
#define FIRST_FC_DIM                 (128)

#define INPUT_FEATURE_DIM_W          (10)
#define FIRST_CONV_OUTPUT_DIM_W      (1)
#define SECOND_CONV_OUTPUT_DIM_W     (1)
#define THIRD_CONV_OUTPUT_DIM_W      (1)
#define FOURTH_CONV_OUTPUT_DIM_H     (8)
#define FOURTH_CONV_OUTPUT_DIM_W     (1)

#define OUT_DIM						(12)

static const struct tiny_tensor move_0_input = {
    .dims = {1, 8, INPUT_FEATURE_DIM_W, 1},
    .dim_num = 4,
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_INPUT,
    .data = NULL,
};

static const struct tiny_tensor move_0_output = {
    .dim_num = 4,
    .dims = {1, 16, INPUT_FEATURE_DIM_W , 1},
    .shift = FIRST_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_move_param move_0_param = {
    .start_mv_addr = (FIRST_CONV_KERNEL_DIM_H - FIRST_CONV_KERNEL_STRIDE_H) * INPUT_FEATURE_DIM_W, //80
	.keep_size = (FIRST_CONV_KERNEL_DIM_H - FIRST_CONV_KERNEL_STRIDE_H) * INPUT_FEATURE_DIM_W , //=80
	.buffer_size = 160 ,
	.current_buf_size = 0 ,
	.flag = 0 ,
};

static const struct tiny_node move_0_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_MOVE,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &move_0_param,
    .input = {&move_0_input, },
    .output = &move_0_output,
};
/* first conv node */


/* for conv weight, the layout is hwio */
static const struct tiny_tensor conv_0_weight = {
    .dim_num = 4,
    .dims = {10, 10, 1, 96},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_0_weight_data,
};

static const struct tiny_tensor conv_0_bias = {
    .dim_num = 1,
    .dims = {96},
    .shift = FIRST_CONV_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_0_bias_data,
};

static const struct tiny_tensor conv_0_output = {
    .dim_num = 4,
    .dims = {1, 4, 1, 96},
    .shift = FIRST_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_conv_param conv_0_param = {
    .kernel_h = 10,
    .kernel_w = 10,
    .stride_h = 2,
    .stride_w = 1,
    .pad_h = NN_PAD_VALID,
    .pad_w = NN_PAD_VALID,
    .activation = -1,
};

static const struct tiny_node conv_0_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_CONV,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &conv_0_param,
    .input = {&move_0_output, &conv_0_weight, &conv_0_bias},
    .output = &conv_0_output,
};


/* the relu node */
static const struct tiny_tensor relu_0_output = {
    .dim_num = 4,
    .dims = {1, 4, 1, 96},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_0_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&conv_0_output},
    .output = &relu_0_output,
};

#if 1
/* the memmove node */
static const struct tiny_tensor move_1_output = {
    .dim_num = 4,
    .dims = {1, 10, 1, 96},
    .shift = FIRST_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_move_param move_1_param = {
    .start_mv_addr = 384 ,
	.keep_size = 576,
	.buffer_size = 960 ,
	.current_buf_size = 0 ,
	.flag = 1 ,
};

static const struct tiny_node move_1_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_MOVE,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &move_1_param,
    .input = {&relu_0_output, },
    .output = &move_1_output,
};

/* the conv node */
static const struct tiny_tensor conv_1_weight = {
    .dim_num = 4,
    .dims = {8, 1, 96 , 80},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_1_weight_data,
};

static const struct tiny_tensor conv_1_bias = {
    .dim_num = 1,
    .dims = {80},
    .shift = SECOND_CONV_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_1_bias_data,
};

static const struct tiny_tensor conv_1_output = {
    .dim_num = 4,
    .dims = {1, 2, 1, 80},
    .shift = SECOND_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_conv_param conv_1_param = {
    .kernel_h = 8,
    .kernel_w = 1,
    .stride_h = 2,
    .stride_w = 1,
    .pad_h = NN_PAD_VALID,
    .pad_w = NN_PAD_VALID,
    .activation = -1,
};

static const struct tiny_node conv_1_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_CONV,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &conv_1_param,
    .input = {&move_1_output, &conv_1_weight, &conv_1_bias},
    .output = &conv_1_output,
};

/* the relu node */
static const struct tiny_tensor relu_1_output = {
    .dim_num = 4,
    .dims = {1, 2, 1, 80},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_1_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&conv_1_output},
    .output = &relu_1_output,
};


/* the memmove node */
static const struct tiny_tensor move_2_output = {
    .dim_num = 4,
    .dims = {1, 5, 1, 80},
    .shift = FIRST_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_move_param move_2_param = {
    .start_mv_addr = 160 ,
	.keep_size = 240,
	.buffer_size = 400 ,
	.current_buf_size = 0 ,
	.flag = 0 ,
};

static const struct tiny_node move_2_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_MOVE,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &move_2_param,
    .input = {&relu_1_output, },
    .output = &move_2_output,
};

/* for conv weight, the layout is hwio */
static const struct tiny_tensor conv_2_weight = {
    .dim_num = 4,
    .dims = {4, 1, 80, 72},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_2_weight_data,
};

static const struct tiny_tensor conv_2_bias = {
    .dim_num = 1,
    .dims = {72},
    .shift = THIRD_CONV_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_2_bias_data,
};

static const struct tiny_tensor conv_2_output = {
    .dim_num = 4,
    .dims = {1, 2, 1, 72},
    .shift = THIRD_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_conv_param conv_2_param = {
    .kernel_h = 4,
    .kernel_w = 1,
    .stride_h = 1,
    .stride_w = 1,
    .pad_h = NN_PAD_VALID,
    .pad_w = NN_PAD_VALID,
    .activation = -1,
};

static const struct tiny_node conv_2_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_CONV,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &conv_2_param,
    .input = {&move_2_output, &conv_2_weight, &conv_2_bias},
    .output = &conv_2_output,
};

/* the relu node */
static const struct tiny_tensor relu_2_output = {
    .dim_num = 4,
    .dims = {1, 3, 1, 72},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_2_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&conv_2_output},
    .output = &relu_2_output,
};


/* the memmove node */
static const struct tiny_tensor move_3_output = {
    .dim_num = 4,
    .dims = {1, 3, 1, 72},
    .shift = FIRST_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_move_param move_3_param = {
	.start_mv_addr = 144 ,
	.keep_size = 144,
	.buffer_size = 288 ,
	.current_buf_size = 0 ,
	.flag = 0 ,
};

static const struct tiny_node move_3_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_MOVE,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &move_3_param,
    .input = {&relu_2_output, },
    .output = &move_3_output,
};


/* for conv weight, the layout is hwio */
static const struct tiny_tensor conv_3_weight = {
    .dim_num = 4,
    .dims = {3, 1, 72, 64},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_3_weight_data,
};

static const struct tiny_tensor conv_3_bias = {
    .dim_num = 1,
    .dims = {64 },
    .shift = FOURTH_CONV_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_3_bias_data,
};

static const struct tiny_tensor conv_3_output = {
    .dim_num = 4,
    .dims = {1, 1, 1, 64},
    .shift = FOURTH_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_conv_param conv_3_param = {
    .kernel_h = 3,
    .kernel_w = 1,
    .stride_h = 2,
    .stride_w = 1,
    .pad_h = NN_PAD_VALID,
    .pad_w = NN_PAD_VALID,
    .activation = -1,
};

static const struct tiny_node conv_3_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_CONV,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &conv_3_param,
    .input = {&move_3_output, &conv_3_weight, &conv_3_bias},
    .output = &conv_3_output,
};

/* the relu node */
static const struct tiny_tensor relu_3_output = {
    .dim_num = 4,
    .dims = {1, 1, 1, 64},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_3_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&conv_3_output},
    .output = &relu_3_output,
};


/* the memmove node */
static const struct tiny_tensor move_4_output = {
    .dim_num = 4,
    .dims = {1, 1, 1, 512},
    .shift = FIRST_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_move_param move_4_param = {
    .start_mv_addr = 64 ,
	.keep_size = 448 ,
	.buffer_size = 512 ,
	.current_buf_size = 0 ,
	.flag = 0 ,
};

static const struct tiny_node move_4_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_MOVE,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &move_4_param,
    .input = {&relu_3_output, },
    .output = &move_4_output,
};

/* the fc node */
static const struct tiny_tensor fc_4_weight = {
    .dim_num = 2,
    .dims = { LINEAR_DIM , FOURTH_CONV_OUTPUT_DIM_H*FOURTH_CONV_OUTPUT_DIM_W*FOURTH_CONV_KERNEL_C }, //(64,8*64)
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_4_weight_data,
};

static const struct tiny_tensor fc_4_bias = {
    .dim_num = 1,
    .dims = {LINEAR_DIM},
    .shift = LINEAR_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_4_bias_data,
};

static const struct tiny_tensor fc_4_output = {
    .dim_num = 2,
    .dims = {1, LINEAR_DIM},
    .shift = LINEAR_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node fc_4_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_FC,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&move_4_output, &fc_4_weight, &fc_4_bias},
    .output = &fc_4_output,
};

/* fc node */
static const struct tiny_tensor fc_5_weight = {
    .dim_num = 2,
    .dims = {FIRST_FC_DIM, LINEAR_DIM },		//(128,64)
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_5_weight_data,
};

static const struct tiny_tensor fc_5_bias = {
    .dim_num = 1,
    .dims = {FIRST_FC_DIM},			//128
    .shift = FIRST_FC_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_5_bias_data,
};

static const struct tiny_tensor fc_5_output = {
    .dim_num = 2,
    .dims = {1, FIRST_FC_DIM},
    .shift = FIRST_FC_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node fc_5_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_FC,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&fc_4_output, &fc_5_weight, &fc_5_bias},
    .output = &fc_5_output,
};

/* the relu node */
static const struct tiny_tensor relu_6_output = {
    .dim_num = 2,
    .dims = {1, FIRST_FC_DIM},						//128
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_6_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&fc_5_output},
    .output = &relu_6_output,
};

/* fc node */
static const struct tiny_tensor fc_7_weight = {
    .dim_num = 2,
    .dims = {OUT_DIM, FIRST_FC_DIM},	//12,128
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_7_weight_data,
};

static const struct tiny_tensor fc_7_bias = {
    .dim_num = 1,
    .dims = {OUT_DIM},					//12
    .shift = FINAL_FC_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_7_bias_data,
};

static const struct tiny_tensor fc_7_output = {
    .dim_num = 2,
    .dims = {1, OUT_DIM},		//(1,12)
    .shift = FINAL_FC_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node fc_7_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_FC,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&relu_6_output, &fc_7_weight, &fc_7_bias},
    .output = &fc_7_output,
};

/* SoftMax node */

static const struct tiny_tensor softmax_8_output = {
    .dim_num = 2,
    .dims = {1, OUT_DIM},		///(1,12)
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node softmax_8_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_SOFTMAX,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&fc_7_output},
    .output = &softmax_8_output,
};

/* the graph node list */
static const struct tiny_node* node_list[] = {
    &move_0_node, &conv_0_node, &relu_0_node, &move_1_node, &conv_1_node, &relu_1_node, 
	&move_2_node, &conv_2_node, &relu_2_node, &move_3_node, &conv_3_node, &relu_3_node,
	&move_4_node, &fc_4_node ,  &fc_5_node,   &relu_6_node, &fc_7_node,   &softmax_8_node,
};
#endif
#if 0
/* the graph node list */
static const struct tiny_node* node_list[] = {
    &move_0_node, &conv_0_node, &relu_0_node ,
};
#endif

static const struct tiny_graph tiny_graph = {
    .name = "speech model",
    .tiny_version = NN_TINY_VERSION_1,
    .nn_id = 0xdeadbeaf,
    .create_time = 0,
    .layout = NN_LAYOUT_NHWC,
    .node_num = sizeof(node_list) / sizeof(void*),
    .node_list = node_list,
};

const struct tiny_graph* get_tiny_graph(void)
{
    return &tiny_graph;
}

void free_tiny_graph(const struct tiny_graph* tiny_graph)
{
    /* NOTHING NEEDS TO DO */
}
