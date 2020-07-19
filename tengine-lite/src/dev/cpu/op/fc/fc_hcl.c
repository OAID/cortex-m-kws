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
#include "op/pool_param.h"

struct hcl_info
{
    hcl_instance_t ins;
    hcl_fc_t fc_op;
};

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* ir_tensor;

    hcl_info->ins = get_hcl_instance(exec_graph->dev);

    if(hcl_info->ins == NULL)
    {
        set_tengine_errno(EFAULT);
        return -1;
    }

    hcl_instance_t ins = hcl_info->ins;

    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    hcl_fc_t fc_op = hcl_create_fc(ins, ir_tensor->dims[1], ir_tensor->dims[0]);

    if(fc_op == NULL)
    {
        TLOG_ERR("create hcl fc handle failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    hcl_info->fc_op = fc_op;

    hcl_fc_set_weight(fc_op, ir_tensor->data, ir_tensor->dims);

    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    hcl_fc_set_input(fc_op, ir_tensor->data, ir_tensor->dims);

    ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    hcl_fc_set_output(fc_op, ir_tensor->data, ir_tensor->dims);

    /* prerun now */
    if(hcl_fc_prerun(fc_op) < 0)
    {
        TLOG_ERR("hcl fc prerun failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;

    if(hcl_fc_run(hcl_info->fc_op) < 0)
    {
        TLOG_ERR("hcl fc run failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return -1;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;

    if(hcl_fc_postrun(hcl_info->fc_op) < 0)
    {
        TLOG_ERR("hcl fc postrun failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    hcl_release_fc(hcl_info->fc_op);
    hcl_info->fc_op = NULL;

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* dev)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )sys_malloc(sizeof(struct hcl_info));

    if(hcl_info == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    hcl_info->ins = NULL;
    hcl_info->fc_op = NULL;

    exec_node->ops_priv = hcl_info;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* dev)
{
    struct hcl_info* hcl_info = ( struct hcl_info* )exec_node->ops_priv;

    /* in case prerun() is done, while postrun does not called */

    if(hcl_info->fc_op != NULL)
        hcl_release_fc(hcl_info->fc_op);

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

static int reg_fc_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_FC, &hcl_node_ops);
}

static int unreg_fc_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_FC, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_fc_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_fc_hcl_ops);
