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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int batch_number = input->dims[0];
    int hidden_number = input->elem_num / batch_number;

    int output_number;

    if(hidden_number == weight->dims[1])
        output_number = weight->dims[0];
    else if(hidden_number == weight->dims[0])
        output_number = weight->dims[1];
    else
    {
        TLOG_ERR("fc: input tensor and weight tensor shape does not match, hidden_number: %d\n", hidden_number);
        set_tengine_errno(EFAULT);
        return -1;
    }

    int dims[2];

    dims[0] = batch_number;
    dims[1] = output_number;

    set_ir_tensor_shape(output, dims, 2);

    return 0;
}

static int init_op(struct ir_op* op)
{
    op->same_shape = 0;
    op->infer_shape = infer_shape;
    return 0;
}

static int register_fc_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = NULL;
    m.access_param_entry = NULL;

    return register_op(OP_FC, OP_FC_NAME, &m);
}

AUTO_REGISTER_OP(register_fc_op);
