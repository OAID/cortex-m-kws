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
#include <stdlib.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_errno.h"
#include "tengine_utils.h"
#include "tengine_ir.h"
#include "nn_device.h"
#include "cpu_device.h"
#include "cpu_node_ops.h"
#include "tengine_log.h"
#include "tengine_op.h"

#define INPLACE_BLOCK_FLAG 0x40
static void release_mem_pool(struct mem_pool* mem_pool);

struct mem_record
{
    struct ir_tensor* ir_tensor;
    int used;
    int block_id;
};

static int find_tensor_mem_list(struct vector* tensor_mem_list, const struct ir_tensor* ir_tensor)
{
    int rec_number = get_vector_num(tensor_mem_list);

    for(int i = 0; i < rec_number; i++)
    {
        struct mem_record* rec = ( struct mem_record* )get_vector_data(tensor_mem_list, i);

        if(rec->ir_tensor == ir_tensor)
            return i;
    }

    return -1;
}

static int init_exec_node(struct exec_graph* exec_graph, struct exec_node* exec_node, struct ir_node* ir_node,
                          struct node_ops* node_ops)
{
    exec_node->ir_node = ir_node;
    exec_node->node_ops = node_ops;
    exec_node->ops_priv = NULL;
    exec_node->inplace_map_num = 0;
    exec_node->inplace_map_ptr = NULL;
    exec_node->shared_mem_size = 0;
    exec_node->output_num = ir_node->output_num;

    int8_t* block_id = exec_node->block_id;

    if(exec_node->output_num > 4)
    {
        exec_node->block_id_ptr = ( int8_t* )sys_malloc(sizeof(int8_t) * exec_node->output_num);
        block_id = exec_node->block_id_ptr;
    }

    for(int i = 0; i < exec_node->output_num; i++)
        block_id[i] = -1;

    if(node_ops->init_node && node_ops->init_node(node_ops, exec_node, exec_graph) < 0)
        return -1;

    return 0;
}

static void release_exec_node(struct exec_graph* exec_graph, struct exec_node* exec_node, struct node_ops* node_ops)
{
    if(node_ops->release_node)
        node_ops->release_node(node_ops, exec_node, exec_graph);

    if(exec_node->inplace_map_num > 2)
        sys_free(exec_node->inplace_map_ptr);

    if(exec_node->output_num > 4)
        sys_free(exec_node->block_id_ptr);
}

static struct exec_graph* new_exec_graph(void)
{
    struct exec_graph* exec_graph = ( struct exec_graph* )sys_malloc(sizeof(struct exec_graph));

    if(exec_graph == NULL)
        return NULL;

    exec_graph->exec_node_list = create_vector(sizeof(struct exec_node), NULL);

    if(exec_graph->exec_node_list == NULL)
    {
        sys_free(exec_graph);
        return NULL;
    }

    exec_graph->shared_mem = NULL;
    exec_graph->shared_mem_size = 0;
    exec_graph->mem_pool = NULL;

    return exec_graph;
}

static void free_exec_graph_mem(struct exec_graph* graph)
{
    /* free the shared memory */
    if(graph->shared_mem)
    {
        sys_free(graph->shared_mem);
        graph->shared_mem = NULL;
        graph->shared_mem_size = 0;
    }

    /* free the mem pool */
    if(graph->mem_pool)
    {
        release_mem_pool(graph->mem_pool);
        graph->mem_pool = NULL;
    }
}

static void release_exec_graph(void* exec_graph)
{
    struct exec_graph* graph = ( struct exec_graph* )exec_graph;

    int node_num = get_vector_num(graph->exec_node_list);

    for(int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = ( struct exec_node* )get_vector_data(graph->exec_node_list, i);
        struct node_ops* node_ops = exec_node->node_ops;

        release_exec_node(graph, exec_node, node_ops);
    }

    free_exec_graph_mem(graph);

    release_vector(graph->exec_node_list);

    sys_free(graph);
}

static struct exec_graph* create_exec_graph(struct subgraph* subgraph, int num_thread)
{
    /* generate exec_graph */
    int node_num = subgraph->node_num;
    struct ir_graph* ir_graph = subgraph->graph;
    struct exec_graph* exec_graph = new_exec_graph();
    struct cpu_device* dev = ( struct cpu_device* )subgraph->nn_dev;

    if(exec_graph == NULL)
    {
        set_tengine_errno(ENOMEM);
        return NULL;
    }

    exec_graph->dev = dev;
    exec_graph->num_thread = num_thread;

    for(int i = 0; i < node_num; i++)
    {
        struct ir_node* ir_node = get_ir_graph_node(ir_graph, subgraph->node_list[i]);

        if(ir_node->op.op_type == OP_CONST || ir_node->op.op_type == OP_INPUT)
            continue;

        struct node_ops* node_ops = find_node_ops(exec_graph, ir_node);

        if(node_ops == NULL)
        {
            TLOG_ERR("%s: failed to find node ops for node: %d\n", dev->base.name, ir_node->idx);
            set_tengine_errno(EFAULT);
            goto error;
        }

        struct exec_node exec_node;

        if(init_exec_node(exec_graph, &exec_node, ir_node, node_ops) < 0)
        {
            TLOG_ERR("%s: failed to init exec node for node: %d\n", dev->base.name, ir_node->idx);
            set_tengine_errno(EFAULT);
            goto error;
        }

        push_vector_data(exec_graph->exec_node_list, &exec_node);
    }

    return exec_graph;

error:
    release_exec_graph(exec_graph);
    return NULL;
}

static int find_inplace_input(struct exec_node* exec_node, int output_slot, struct ir_node* ir_node,
                              struct ir_graph* ir_graph)
{
    if(exec_node->inplace_map_num == 0)
        return -1;

    uint8_t* inplace_map;

    if(exec_node->inplace_map_num > 2)
        inplace_map = exec_node->inplace_map_ptr;
    else
        inplace_map = exec_node->inplace_map;

    int i;
    for(i = 0; i < 2 * exec_node->inplace_map_num; i += 2)
    {
        if(inplace_map[i] == output_slot)
            break;
    }

    /* no map */
    if(i == 2 * exec_node->inplace_map_num)
        return -1;

    int input_slot = inplace_map[i + 1];

    struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[input_slot]);

    if(tensor->consumer_num > 1)
        return -1;

    return input_slot;
}

static void mem_pool_dump(struct mem_pool* mem_pool)
{
    int block_number = get_vector_num(mem_pool->block_list);

    TLOG_INFO("block number: %d align size: %d\n", block_number, mem_pool->align_size);

    for(int i = 0; i < block_number; i++)
    {
        struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, i);

        TLOG_INFO("%d: %p (%d) used: %d free: %d\n", i, entry->addr, entry->block_size, entry->alloc_count,
                  entry->free_count);
    }
}

static void* mem_pool_get_mem_block(struct mem_pool* mem_pool, int block_id)
{
    struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, block_id);

    unsigned long addr = ( long )(entry->addr);
    unsigned long aligned_addr = (addr + 4 + mem_pool->align_size) & (~(mem_pool->align_size - 1));

    return ( void* )aligned_addr;
}

static int mem_pool_get_backend_mem(struct mem_pool* mem_pool)
{
    int block_num = get_vector_num(mem_pool->block_list);

    for(int i = 0; i < block_num; i++)
    {
        struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, i);

        entry->block_size = entry->max_req_size + mem_pool->align_size + 4;

        entry->addr = sys_malloc(entry->block_size);

        if(entry->addr == NULL)
            return -1;
    }

    return 0;
}

static int mem_pool_allocate(struct mem_pool* mem_pool, int size)
{
    int block_num = get_vector_num(mem_pool->block_list);
    ;

    for(int i = 0; i < block_num; i++)
    {
        struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, i);

        if(entry->free_count != entry->alloc_count)
            continue;

        /* TODO: use the best match alg */

        entry->alloc_count++;

        if(entry->max_req_size < size)
            entry->max_req_size = size;

        return i;
    }

    /* create new block */

    struct mem_block_entry e;

    e.addr = NULL;
    e.max_req_size = size;
    e.alloc_count = 1;
    e.free_count = 0;

    push_vector_data(mem_pool->block_list, &e);

    return block_num;
}

static void mem_pool_free(struct mem_pool* mem_pool, int block_id)
{
    struct mem_block_entry* block = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, block_id);

    block->free_count++;
}

static void release_mem_pool(struct mem_pool* mem_pool)
{
    if(mem_pool->block_list != NULL)
    {
        int block_num = get_vector_num(mem_pool->block_list);

        for(int i = 0; i < block_num; i++)
        {
            struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, i);

            sys_free(entry->addr);
        }

        release_vector(mem_pool->block_list);
    }

    sys_free(mem_pool);
}

static struct mem_pool* create_mem_pool(void)
{
    struct mem_pool* mem_pool = ( struct mem_pool* )sys_malloc(sizeof(struct mem_pool));

    if(mem_pool == NULL)
        return NULL;

    mem_pool->align_size = 16;
    mem_pool->block_list = create_vector(sizeof(struct mem_block_entry), NULL);

    if(mem_pool->block_list == NULL)
        goto error;

    mem_pool->allocate = mem_pool_allocate;
    mem_pool->free = mem_pool_free;
    mem_pool->dump = mem_pool_dump;
    mem_pool->get_backend_mem = mem_pool_get_backend_mem;
    mem_pool->get_mem_block = mem_pool_get_mem_block;

    return mem_pool;

error:

    release_mem_pool(mem_pool);

    return NULL;
}

static int alloc_exec_graph_mem(struct exec_graph* exec_graph)
{
    struct mem_pool* mem_pool;
    int max_shared_mem_size = 0;

    int node_num = get_vector_num(exec_graph->exec_node_list);

    struct vector* tensor_mem_list = create_vector(sizeof(struct mem_record), NULL);

    if(tensor_mem_list == NULL)
        return -1;

    mem_pool = create_mem_pool();

    if(mem_pool == NULL)
        return -1;

    exec_graph->mem_pool = mem_pool;

    for(int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct ir_node* ir_node = exec_node->ir_node;
        struct ir_graph* ir_graph = ir_node->graph;

        int8_t* block_id;

        if(exec_node->output_num > 4)
            block_id = exec_node->block_id_ptr;
        else
            block_id = exec_node->block_id;

        for(int j = 0; j < ir_node->output_num; j++)
        {
            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[j]);

            if(ir_tensor->data != NULL)
                continue;

            int inplace_input = find_inplace_input(exec_node, j, ir_node, ir_graph);

            if(inplace_input >= 0)
            {
                struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[inplace_input]);

                int idx = find_tensor_mem_list(tensor_mem_list, input_tensor);

                /* if the input is from outside buffer, input_r should be NULL */
                if(idx < 0)
                    continue;

                struct mem_record* input_r = ( struct mem_record* )get_vector_data(tensor_mem_list, idx);

                input_r->ir_tensor = ir_tensor;
                input_r->used = ir_tensor->consumer_num;
                block_id[j] = INPLACE_BLOCK_FLAG | inplace_input;
                continue;
            }

            /* allocate mem from pool */
            int mem_size = ir_tensor->elem_size * ir_tensor->elem_num;

            struct mem_record r;

            r.ir_tensor = ir_tensor;
            r.block_id = mem_pool->allocate(mem_pool, mem_size);
            r.used = ir_tensor->consumer_num;

            block_id[j] = r.block_id;

            push_vector_data(tensor_mem_list, &r);
        }

        /* clear input tensor count */
        for(int j = 0; j < ir_node->input_num; j++)
        {
            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[j]);

            if(ir_tensor->data != NULL)
                continue;

            int idx = find_tensor_mem_list(tensor_mem_list, ir_tensor);

            if(idx < 0)
                continue;

            struct mem_record* input_r = ( struct mem_record* )get_vector_data(tensor_mem_list, idx);

            input_r->used--;

            if(input_r->used == 0)
            {
                mem_pool->free(mem_pool, input_r->block_id);
                remove_vector_by_idx(tensor_mem_list, idx);
            }
        }

        /* handle shared mem */
        if(exec_node->shared_mem_size > max_shared_mem_size)
            max_shared_mem_size = exec_node->shared_mem_size;
    }

    TLOG_DEBUG("final tensor_mem_list number: %d\n", get_vector_num(tensor_mem_list));

    release_vector(tensor_mem_list);

    exec_graph->shared_mem_size = max_shared_mem_size;

    if(max_shared_mem_size > 0)
    {
        exec_graph->shared_mem = sys_malloc(max_shared_mem_size);

        if(exec_graph->shared_mem == NULL)
        {
            TLOG_ERR("cannot allocate shared memory. size=%d\n", max_shared_mem_size);
            return -1;
        }
    }

    TLOG_ERR("shared memory: %p size=%d\n", exec_graph->shared_mem, max_shared_mem_size);

    if(mem_pool->get_backend_mem(mem_pool) < 0)
    {
        TLOG_ERR("cannot allocate enough memory from backend\n");
        return -1;
    }

    mem_pool->dump(mem_pool);

    /* now, the real allocate */
    for(int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct ir_node* ir_node = exec_node->ir_node;
        struct ir_graph* ir_graph = ir_node->graph;
        struct mem_pool* mem_pool = exec_graph->mem_pool;

        int8_t* block_id;

        if(exec_node->output_num > 4)
            block_id = exec_node->block_id_ptr;
        else
            block_id = exec_node->block_id;

        for(int j = 0; j < ir_node->output_num; j++)
        {
            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[j]);

            if(block_id[j] < 0)
                continue;

            if(block_id[j] & INPLACE_BLOCK_FLAG)
            {
                int input_idx = block_id[j] & (INPLACE_BLOCK_FLAG - 1);

                struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[input_idx]);
                ir_tensor->data = input_tensor->data;
                ir_tensor->free_host_mem = 0;
                ir_tensor->internal_allocated = MEM_POOL_ALLOCATED;
            }
            else
            {
                ir_tensor->data = mem_pool->get_mem_block(mem_pool, block_id[j]);
                ir_tensor->free_host_mem = 0;
                ir_tensor->internal_allocated = MEM_POOL_ALLOCATED;
            }
        }
    }

    return 0;
}

static int prerun_exec_graph(struct exec_graph* exec_graph)
{
    int node_num = get_vector_num(exec_graph->exec_node_list);

    for(int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = exec_node->node_ops;

        if(node_ops->prerun && node_ops->prerun(node_ops, exec_node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to prerun node %d\n", exec_graph->dev->base.name, exec_node->ir_node->idx);
            return -1;
        }
    }

    return 0;
}

static int prerun(struct nn_device* dev, struct subgraph* subgraph, int num_thread)
{
    struct exec_graph* exec_graph;

    /* create exec_graph */
    exec_graph = create_exec_graph(subgraph, num_thread);

    if(exec_graph == NULL)
        return -1;

    if(alloc_exec_graph_mem(exec_graph) < 0 || prerun_exec_graph(exec_graph) < 0)
    {
        release_exec_graph(exec_graph);
        return -1;
    }

    subgraph->exec_graph = exec_graph;

    return 0;
}

static int run(struct nn_device* dev, struct subgraph* subgraph)
{
    struct exec_graph* exec_graph = subgraph->exec_graph;

    int node_num = get_vector_num(exec_graph->exec_node_list);
	
    for(int i = 0; i < node_num; i++)
    {
        struct exec_node* node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = node->node_ops;

        /* TODO: handle the shape changed  and dynamic shape case */

        /* TODO: add dynamic skip feature */
        struct ir_node *ir_node = node->ir_node ;
        struct ir_op* op = &ir_node->op;
               
        if(op->infer_shape && op->infer_shape(ir_node) < 0 )
        {
            TLOG_ERR("%s: failed to run node %d\n", dev->name, node->ir_node->idx);
        }
        
        int ret = node_ops->run(node_ops, node, exec_graph) ; 
				
		if ( ret > 0 )
			break ;
				
		if ( ret < 0 )
        {
            TLOG_ERR("%s: failed to run node %d\n", dev->name, node->ir_node->idx);
            return -1;
        }
				
//#define DUMP_NODE_OUTPUT
#ifdef DUMP_NODE_OUTPUT
        /* dump the node output */
        struct ir_node* ir_node = node->ir_node;
        struct ir_graph* ir_graph = ir_node->graph;

        for(int i = 0; i < ir_node->input_num; i++)
        {
            char fname[128];
            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);

            sprintf(fname, "/tmp/dump/node%s%d.%d", (ir_node->idx < 10 ? "0" : ""), ir_node->idx, i);

            dump_float(fname, ir_tensor->data, ir_tensor->elem_num);
        }

#endif

    }

    return 0;
}

static int postrun(struct nn_device* dev, struct subgraph* subgraph)
{
    struct exec_graph* exec_graph = subgraph->exec_graph;

    int node_num = get_vector_num(exec_graph->exec_node_list);

    for(int i = 0; i < node_num; i++)
    {
        struct exec_node* node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = node->node_ops;

        if(node_ops->postrun && node_ops->postrun(node_ops, node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to postrun node %d\n", dev->name, node->ir_node->idx);
        }
    }

    release_exec_graph(exec_graph);

    subgraph->exec_graph = NULL;

    return 0;
}

static int cpu_dev_release_exec_graph(struct nn_device* dev, void* exec_graph)
{
    release_exec_graph(exec_graph);
    return 0;
}

static struct cpu_device cpu_dev = {
    .base = {.name = "cpu_dev",
             .prerun = prerun,
             .run = run,
             .postrun = postrun,
             .async_run = NULL,
             .async_wait = NULL,
             .release_exec_graph = cpu_dev_release_exec_graph,
             .init = NULL,
             .release = NULL},
    .master_cpu = 0,
    .cpu_model = 0,
};

REGISTER_NN_DEVICE(&cpu_dev.base);
