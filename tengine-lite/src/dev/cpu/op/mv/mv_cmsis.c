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

#include "arm_math.h"
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "cpu_node_ops.h"
#include "tengine_op.h"
#include "op/mv_param.h"

static int move_op(const signed char *new_buffer ,  \
                                     const int new_buffer_size ,      \
                                     signed char *buffer ,            \
                                     const int start_move_addr ,      \
                                     const int move_size ,            \
                                     int *current_buffer_size ,       \
                                     int flag 	,                     \
                                     signed char *buffer_out ,        \
                                     const int buffer_out_size        \
                                    )
{
    memmove(buffer + *current_buffer_size  , new_buffer , new_buffer_size);
    
    //printf("                 insert addr = %d \n", *current_buffer_size);
    *current_buffer_size += new_buffer_size ;
        
    if(flag){
        if( *current_buffer_size < 768 ){
            return *current_buffer_size ;
            }
        }else{
            if(*current_buffer_size < buffer_out_size ){
                return *current_buffer_size ;
        }
    }
    
    memcpy(buffer_out , buffer , buffer_out_size);
     
    if(*current_buffer_size < buffer_out_size ){
        memmove(buffer, buffer + buffer_out_size - *current_buffer_size  , move_size);
        //printf("                 start_move_addr : %d , move_size : %d \n",buffer_out_size - *current_buffer_size , move_size );
    }else{
        memmove(buffer, buffer+start_move_addr , move_size);
    }
    
    //printf("                  current buffer size = %d \n", *current_buffer_size  );

    *current_buffer_size -= new_buffer_size ;

    if(flag)
        *current_buffer_size = buffer_out_size - new_buffer_size ;

    return 0 ;
}	


static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
        
    struct mv_param* mv_param = ( struct mv_param* )ir_node->op.param_mem;
    mv_param->buffer	=  (char *)sys_malloc(mv_param->buffer_size*sizeof(char));

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node; 
    struct mv_param* mv_param = ( struct mv_param* )ir_node->op.param_mem;
    sys_free(mv_param->buffer);
        
    exec_node->inplace_map_num = 0;
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;
    
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    
    int input_ele_num = input_tensor->dims[1]*(input_tensor->dims[2])*(input_tensor->dims[3]);  
    //printf("        MV:dims[0, 1, 2, 3]=[%d  %d  %d  %d ]\n",input_tensor->dims[0] , input_tensor->dims[1], input_tensor->dims[2] ,input_tensor->dims[3] );
    
    struct mv_param* mv_param = ( struct mv_param* )ir_node->op.param_mem;

    int ret = move_op(input_tensor->data, input_ele_num , \
                                            mv_param->buffer , mv_param->start_mv_addr, \
                                            mv_param->mv_size , &mv_param->current_buffer_size , mv_param->flag, \
                                            output_tensor->data , mv_param->buffer_out_size  );

    if ( ret > 0 )
        return 1 ;
    else
        return 0 ;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops mv_node_ops = {.prerun = NULL,
                                         .run = run,
                                         .reshape = NULL,
                                         .postrun = NULL,
                                         .init_node = init_node,
                                         .release_node = release_node,
                                         .score = score};

static int reg_mv_cmsis_ops(void* arg)
{
    return register_builtin_node_ops(OP_MOVE, &mv_node_ops);
}

static int unreg_mv_cmsis_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_MOVE, &mv_node_ops);
}

AUTO_REGISTER_OPS(reg_mv_cmsis_ops);
AUTO_UNREGISTER_OPS(unreg_mv_cmsis_ops);
