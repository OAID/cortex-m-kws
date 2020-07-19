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
 * Copyright (c) 2019, Open AI Lab
 * Author: haitao@openailab.com
 */

#include "hcl_api.h"
#include "hcl_cpu.h"

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "cpu_node_ops.h"
#include "tengine_op.h"
#include "convolution_param.h"

struct hcl_info
{
    hcl_instance_t ins;
    hcl_conv_2d_t conv_op;
};

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* ir_tensor;

    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;
    hcl_conv_2d_t conv_op = hcl_info->conv_op;

    if(exec_graph->shared_mem &&
       hcl_conv_2d_set_shared_mem(conv_op, exec_graph->shared_mem, exec_node->shared_mem_size) < 0)
    {
        TLOG_ERR("hcl conv: set shared memory failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    hcl_conv_2d_set_input(conv_op, ir_tensor->data, ir_tensor->dims);

    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    hcl_conv_2d_set_output(conv_op, ir_tensor->data, ir_tensor->dims);

    /* prerun now */
    if(hcl_conv_2d_prerun(conv_op) < 0)
    {
        TLOG_ERR("hcl conv prerun failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* ir_tensor;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    hcl_conv_2d_set_input(hcl_info->conv_op, ir_tensor->data, ir_tensor->dims);

    if(hcl_conv_2d_run(hcl_info->conv_op) < 0)
    {
        TLOG_ERR("hcl conv run failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* ir_tensor;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    hcl_conv_2d_set_input(hcl_info->conv_op, ir_tensor->data, ir_tensor->dims);

    if(hcl_conv_2d_reshape(hcl_info->conv_op) < 0)
    {
        TLOG_ERR("hcl conv reshape failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;

    if(hcl_conv_2d_postrun(hcl_info->conv_op) < 0)
    {
        TLOG_ERR("hcl conv run failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* ir_tensor;

    struct hcl_info* hcl_info = ( struct hcl_info* )sys_malloc(sizeof(struct hcl_info));

    if(hcl_info == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    hcl_info->ins = get_hcl_instance(exec_graph->dev);

    if(hcl_info->ins == NULL)
    {
        set_tengine_errno(EFAULT);
        return -1;
    }

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    hcl_instance_t ins = hcl_info->ins;

    hcl_conv_2d_t conv_op = hcl_create_conv_2d(
        ins, conv_param->kernel_h, conv_param->kernel_w, conv_param->stride_h, conv_param->stride_w, conv_param->pad_h0,
        conv_param->pad_h1, conv_param->pad_w0, conv_param->pad_w1, conv_param->dilation_h, conv_param->dilation_w,
        conv_param->input_channel, conv_param->output_channel, conv_param->group);

    if(conv_op == NULL)
    {
        TLOG_ERR("create hcl conv handle failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    hcl_info->conv_op = conv_op;

    exec_node->ops_priv = hcl_info;

    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    hcl_conv_2d_set_input(conv_op, ir_tensor->data, ir_tensor->dims);

    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    hcl_conv_2d_set_filter(conv_op, ir_tensor->data, ir_tensor->dims);

    if(ir_node->input_num > 2)
    {
        ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        hcl_conv_2d_set_bias(conv_op, ir_tensor->data, ir_tensor->dims[0]);
    }

    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    hcl_conv_2d_set_output(conv_op, ir_tensor->data, ir_tensor->dims);

    /* activatioin fuse */
    if(conv_param->activation == 0)
    {
        hcl_conv_2d_config_relu_fuse(conv_op, RELU_FUSED, 0);
    }
    else if(conv_param->activation > 0)
    {
        hcl_conv_2d_config_relu_fuse(conv_op, RELU6_FUSED, 0);
    }

    /* get shared memory size */
    exec_node->shared_mem_size = hcl_conv_2d_get_shared_mem_size(conv_op);

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;

    if(hcl_info->conv_op != NULL)
        hcl_release_conv_2d(hcl_info->conv_op);

    sys_free(hcl_info);

    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_conv_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

static int unreg_conv_hcl_ops(void* arg)
{
    unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
    return 0;
}

AUTO_REGISTER_OPS(reg_conv_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_conv_hcl_ops);
