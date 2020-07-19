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

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "op/mv_param.h"

DEFINE_PARM_PARSE_ENTRY(mv_param, kernel_h, kernel_w, stride_h, stride_w, pad_h0, pad_h1, pad_w0, pad_w1, dilation_h,
                        dilation_w, input_channel, output_channel, group);

static int infer_shape(struct ir_node* node)
{
    
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct mv_param* mv_param = ( struct mv_param* )(node->op.param_mem);
   
    if(mv_param->flag==1)
    {   
//        static int times = 0 ;
//        int dims[4];
//        dims[0] = output->dims[0];
//        dims[2] = output->dims[2];
//        static int out_h = 0 ;
//        out_h += input->dims[1] ; 

//        if( times==0 || times >=3 ){
//            dims[1] = 10;
//        }
//        else if(times==1 || times==2 ){
//            dims[1] = 8;
//        }
//        times++ ;
//        dims[3] = output->dims[3];
//        set_ir_tensor_shape(output, dims, 4);
    
        static int out_h = 0 ;
        int dims[4];
        dims[0] = output->dims[0];
        dims[2] = output->dims[2];
     
        if(out_h > 8 ){
            dims[1] = 10;
        }
        else{
            out_h += input->dims[1] ; 
            dims[1] = 8;
        }
        
        dims[3] = output->dims[3];
        set_ir_tensor_shape(output, dims, 4);        

    }
        
    int ele_num = output->dims[1]*(output->dims[2])*(output->dims[3]);
    mv_param->buffer_out_size = ele_num ;
    //printf("    mv_param->buffer_out_size= %d \n", mv_param->buffer_out_size);
    
    return 0;
}

static int init_op(struct ir_op* op)
{
    struct mv_param* mv_param = ( struct mv_param* )sys_malloc(sizeof(struct mv_param));

    if(mv_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }
    
		/*set the param default value */
		mv_param->start_mv_addr = 0 ;
		mv_param->mv_size = 0 ;
		mv_param->current_buffer_size = 0 ;
		mv_param->buffer_out_size = 0 ;	
		mv_param->buffer = NULL ; 
        mv_param->tmp_buffer_out_size = 0 ;
		mv_param->flag = 0 ;

    op->param_mem = mv_param;
    op->param_size = sizeof(struct mv_param);
		
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_mv_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
	m.access_param_entry = access_param_entry;

    return register_op(OP_MOVE, OP_MV_NAME, &m);
}

static int unregister_mv_op(void* arg)
{

    return unregister_op(OP_MOVE, 1);
}

AUTO_REGISTER_OP(register_mv_op);
AUTO_UNREGISTER_OP(unregister_mv_op);
