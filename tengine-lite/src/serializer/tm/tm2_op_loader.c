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
 * Author: haitao@openailab.com
 */

#include <stdio.h>
#include <stdlib.h>

#include "sys_port.h"
#include "module.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_serializer.h"
#include "tm2_serializer.h"
#include "tengine_op.h"
#include "op/conv_param.h"
#include "op/pool_param.h"

static int const_op_map(int op)
{
    return OP_CONST;
}

static int input_op_map(int op)
{
    return OP_INPUT;
}

#ifdef CONFIG_OP_CONV

static int conv_op_map(int op)
{
    return OP_CONV;
}

static int tm2_load_conv(struct ir_graph* ir_graph, struct ir_node* ir_node, const TM2_Node* tm_node,
                         const TM2_Operator* tm_op)
{
    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = ( struct tm2_priv* )ir_graph->serializer_priv;
    const char* mem_base = tm2_priv->base;
    const TM2_ConvParam* tm_param = ( TM2_ConvParam* )(mem_base + tm_op->offset_t_param);

    conv_param->kernel_h = tm_param->kernel_h;
    conv_param->kernel_w = tm_param->kernel_w;
    conv_param->stride_h = tm_param->stride_h;
    conv_param->stride_w = tm_param->stride_w;

    /* todo: using new TM2 model definition*/

    conv_param->pad_h0 = tm_param->pad_h;
    conv_param->pad_h1 = tm_param->pad_h;
    conv_param->pad_w0 = tm_param->pad_w;
    conv_param->pad_w1 = tm_param->pad_w;

    conv_param->dilation_h = tm_param->dilation_h;
    conv_param->dilation_w = tm_param->dilation_w;
    conv_param->output_channel = tm_param->output_channel;
    conv_param->activation = tm_param->activation;

    conv_param->group = tm_param->group;

    /* TODO: get input_channel from tm_param */

    struct ir_tensor* weight = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    conv_param->input_channel = weight->dims[1] * conv_param->group;

    if(ir_node->input_num > 2)
    {
        struct ir_tensor* bias = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        int shape = bias->elem_num;

        set_ir_tensor_shape(bias, &shape, 1);
    }

    return 0;
}

#endif

#ifdef CONFIG_OP_POOL
static int pooling_op_map(int op)
{
    return OP_POOL;
}

static int tm2_load_pooling(struct ir_graph* ir_graph, struct ir_node* ir_node, const TM2_Node* tm_node,
                            const TM2_Operator* tm_op)
{
    struct pool_param* pool_param = ( struct pool_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = ( struct tm2_priv* )ir_graph->serializer_priv;
    const char* mem_base = tm2_priv->base;
    const TM2_PoolParam* tm_param = ( TM2_PoolParam* )(mem_base + tm_op->offset_t_param);

    pool_param->kernel_h = tm_param->kernel_h;
    pool_param->kernel_w = tm_param->kernel_w;
    pool_param->stride_h = tm_param->stride_h;
    pool_param->stride_w = tm_param->stride_w;
    pool_param->global = tm_param->global;
    pool_param->caffe_flavor = tm_param->caffe_flavor;

    pool_param->pad_h0 = tm_param->pads[0];
    pool_param->pad_h1 = tm_param->pads[1];
    pool_param->pad_w0 = tm_param->pads[2];
    pool_param->pad_w1 = tm_param->pads[3];

    pool_param->pool_method = tm_param->alg;

    return 0;
}

#endif

/* the auto register functions */

static struct serializer* tm2_s = NULL;

static int reg_tm2_ops(void* arg)
{
    if(tm2_s == NULL)
    {
        tm2_s = find_serializer("tengine");
    }

    if(tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_INPUTOP, 1, NULL_TM2_OP_LOADER, input_op_map, NULL);
    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_CONST, 1, NULL_TM2_OP_LOADER, const_op_map, NULL);

#ifdef CONFIG_OP_CONV
    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_CONVOLUTION, 1, tm2_load_conv, conv_op_map, NULL);
#endif

#ifdef CONFIG_OP_POOL
    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_POOLING, 1, tm2_load_pooling, pooling_op_map, NULL);
#endif

    return 0;
}

static int unreg_tm2_ops(void* arg)
{
    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_INPUTOP, 1, NULL_TM2_OP_LOADER);
    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_CONST, 1, NULL_TM2_OP_LOADER);
#ifdef CONFIG_OP_CONV
    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_CONVOLUTION, 1, tm2_load_conv);
#endif

#ifdef CONFIG_OP_POOL
    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_POOLING, 1, tm2_load_pooling);
#endif

    return 0;
}

REGISTER_MODULE_INIT(MOD_OP_LEVEL, NULL, reg_tm2_ops);
REGISTER_MODULE_EXIT(MOD_OP_LEVEL, NULL, unreg_tm2_ops);
