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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "tengine_serializer.h"
#include "tm2_serializer.h"

struct op_loader_entry
{
    int op_type;
    int op_version;
    tm2_op_loader_t loader;
    tm2_map_t op_map;
    tm2_map_t ver_map;
};

struct tm2_serializer
{
    struct serializer base;

    struct vector* loader_list;
};

static const char* tm2_name = "tengine";

static int unload_graph(struct serializer* s, struct ir_graph* graph, void* s_priv, void* dev_priv);

static char* strdup_name(char* buf, int size)
{
    char* p = sys_malloc(size + 1);
    memcpy(p, buf, size);
    p[size] = 0x0;

    return p;
}

static inline const TM2_Header* get_tm_file_header(const char* base)
{
    return ( const TM2_Header* )(base);
}

static inline const TM2_Model* get_tm_file_model(const char* base, const TM2_Header* header)
{
    return ( const TM2_Model* )(base + header->offset_root);
}

static inline const TM2_Subgraph* get_tm_file_subgraph(const char* base, const TM2_Model* model)
{
    const TM2_Vector_offsets* v_graphs = ( TM2_Vector_offsets* )(base + model->offset_vo_subgraphs);
    const TM2_Subgraph* tm_graph = ( TM2_Subgraph* )(base + v_graphs->offsets[0]);

    return tm_graph;
}

static struct op_loader_entry* find_op_loader(struct tm2_serializer* s, int op_type, int op_version)
{
    int loader_num = get_vector_num(s->loader_list);

    for(int i = 0; i < loader_num; i++)
    {
        struct op_loader_entry* e = ( struct op_loader_entry* )get_vector_data(s->loader_list, i);

        if(e->op_type == op_type)
            return e;
    }

    return NULL;
}

static int register_tm2_op_loader(struct tm2_serializer* s, int op_type, int op_version, tm2_op_loader_t op_loader,
                                  tm2_map_t op_map, tm2_map_t ver_map)
{
    if(find_op_loader(s, op_type, op_version) != NULL)
    {
        TLOG_ERR("serializer: op: %d version %d has loader already\n", op_type, op_version);
        set_tengine_errno(EEXIST);
        return -1;
    }

    struct op_loader_entry e;

    e.op_type = op_type;
    e.op_version = op_version;
    e.loader = op_loader;
    e.op_map = op_map;
    e.ver_map = ver_map;

    push_vector_data(s->loader_list, &e);

    return 0;
}

static int unregister_tm2_op_loader(struct tm2_serializer* s, int op_type, int op_version, tm2_op_loader_t op_loader)
{
    int n = get_vector_num(s->loader_list);

    for(int i = 0; i < n; i++)
    {
        struct op_loader_entry* e = ( struct op_loader_entry* )get_vector_data(s->loader_list, i);

        if(e->op_type == op_type && e->loader == op_loader)
        {
            remove_vector_data(s->loader_list, e);
            return 0;
        }
    }

    return -1;
}

static int load_graph_tensors(struct tm2_serializer* tm2_s, struct ir_graph* graph, struct tm2_priv* priv)
{
    char* mem_base = ( char* )priv->base;
    const TM2_Subgraph* tm_graph = priv->subgraph;

    const TM2_Vector_offsets* v_tensors = ( TM2_Vector_offsets* )(mem_base + tm_graph->offset_vo_tensors);
    const TM2_Vector_offsets* v_buffers = ( TM2_Vector_offsets* )(mem_base + tm_graph->offset_vo_buffers);

    graph->graph_layout = tm_graph->graph_layout ;
    graph->model_layout = tm_graph->model_layout ;

    for(unsigned int i = 0; i < v_tensors->v_num; i++)
    {
        const TM2_Tensor* tm_tensor = ( TM2_Tensor* )(mem_base + v_tensors->offsets[i]);

        /* TODO: check type definition */
        struct ir_tensor* ir_tensor = create_ir_tensor(graph, NULL, tm_tensor->data_type);

        if(ir_tensor == NULL)
        {
            set_tengine_errno(ENOMEM);
            return -1;
        }

        ir_tensor->tensor_type = tm_tensor->type;

        /* name */
        if(tm_tensor->offset_s_tname != TM2_NOT_SET)
        {
            // TODO: using update the TM2 model
            const TM2_String* tm_str = ( TM2_String* )(mem_base + tm_tensor->offset_s_tname);
            ir_tensor->name = strdup_name(mem_base + tm_str->offset_data, tm_str->size);
        }

        /* shape */

        if(tm_tensor->offset_vd_dims != TM2_NOT_SET)
        {
            const TM2_Vector_dims* v_dims = ( TM2_Vector_dims* )(mem_base + tm_tensor->offset_vd_dims);

            set_ir_tensor_shape(ir_tensor, v_dims->dims, v_dims->v_num);
        }

        if(ir_tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            const TM2_Buffer* tm_buf = ( TM2_Buffer* )(mem_base + v_buffers->offsets[tm_tensor->buffer_id]);

            ir_tensor->data = mem_base + tm_buf->offset_data;

            if(ir_tensor->elem_size * ir_tensor->elem_num > tm_buf->size)
            {
                TLOG_ERR("serializer: const tensor size in model is too small\n");
                set_tengine_errno(EFAULT);
                return -1;
            }
        }
        if(tm_tensor->offect_vo_quantparams != TM2_NOT_SET)
        {
            const TM2_Vector_offsets* v_quantparams = (TM2_Vector_offsets *)(mem_base + tm_tensor->offect_vo_quantparams); 
            
            /* currently only support one quant param */
            ir_tensor->quant_param_num = v_quantparams->v_num ;
            if(v_quantparams->v_num == 1 )
            {
                const TM2_QuantParam* tm_qtparam = (TM2_QuantParam *) (mem_base + v_quantparams->offsets[0]) ;
                ir_tensor->scale =  tm_qtparam->scale ;
                ir_tensor->zero_point = tm_qtparam->zero_point ;
            }else if(v_quantparams->v_num > 1){
                //to do : need to be updated
                ir_tensor->scale_list = (float *) sys_malloc(sizeof(float) * v_quantparams->v_num) ; 
                ir_tensor->zp_list = (int *) sys_malloc(sizeof(int) * v_quantparams->v_num) ;

                for(unsigned int i = 0 ; i < v_quantparams->v_num ; ++i)
                {
                    const TM2_QuantParam* tm_qtparam = (TM2_QuantParam *) (mem_base + v_quantparams->offsets[i]) ;
                    ir_tensor->scale_list[i] = tm_qtparam->scale ;
                    ir_tensor->zp_list[i] = tm_qtparam->zero_point ;
                }
            }
        }
    }
    return 0;
}

static int load_graph_nodes(struct tm2_serializer* tm2_s, struct ir_graph* ir_graph, struct tm2_priv* priv)
{
    char* mem_base = ( char* )priv->base;
    const TM2_Subgraph* tm_graph = priv->subgraph;
    const TM2_Vector_offsets* v_nodes = ( TM2_Vector_offsets* )(mem_base + tm_graph->offset_vo_seq_nodes);

    unsigned int i;

    for(i = 0; i < v_nodes->v_num; i++)
    {
        const TM2_Node* tm_node = ( TM2_Node* )(mem_base + v_nodes->offsets[i]);
        const TM2_Operator* tm_operator = ( TM2_Operator* )(mem_base + tm_node->offset_t_operator);
        int op_type = tm_operator->operator_type;
        int op_version = tm_operator->op_ver;

        struct op_loader_entry* e = find_op_loader(tm2_s, op_type, op_version);

        if(e == NULL)
        {
            TLOG_ERR("serializer: cannot find op loader for op: %d version: %d\n", op_type, op_version);
            break;
        }

        int op_type_mapped = op_type;

        if(e->op_map)
            op_type_mapped = e->op_map(op_type);

        int op_ver_mapped = op_version;

        if(e->ver_map)
            op_ver_mapped = e->ver_map(op_version);

        struct ir_node* ir_node = create_ir_node(ir_graph, NULL, op_type_mapped, op_ver_mapped);

        if(ir_node == NULL)
        {
            set_tengine_errno(ENOMEM);
            break;
        }

        if(tm_node->offset_s_nname != TM2_NOT_SET)
        {
            const TM2_String* str = ( TM2_String* )(mem_base + tm_node->offset_s_nname);
            // TODO: update with new tm2
            ir_node->name = strdup_name(mem_base + str->offset_data, str->size);
        }

        /* node inputs */
        if(tm_node->offset_vi_input_tensors != TM2_NOT_SET)
        {
            const TM2_Vector_indices* v_input_tensors =
                ( TM2_Vector_indices* )(mem_base + tm_node->offset_vi_input_tensors);

            for(unsigned int j = 0; j < v_input_tensors->v_num; j++)
            {
                int tensor_idx = v_input_tensors->indices[j];

                if(tensor_idx < 0 || tensor_idx >= ir_graph->tensor_num)
                {
                    TLOG_ERR("invalid input tensor slot: %d idx: %d for node: %d\n", j, tensor_idx, ir_node->idx);
                    break;
                }

                struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, tensor_idx);

                set_ir_node_input_tensor(ir_node, j, ir_tensor);
            }
        }

        if(tm_node->offset_vi_output_tensors == TM2_NOT_SET)
        {
            TLOG_ERR("node: %d has no output\n", ir_node->idx);
            break;
        }

        const TM2_Vector_indices* v_output_tensors =
            ( TM2_Vector_indices* )(mem_base + tm_node->offset_vi_output_tensors);

        for(unsigned int k = 0; k < v_output_tensors->v_num; k++)
        {
            int tensor_idx = v_output_tensors->indices[k];

            if(tensor_idx < 0 || tensor_idx >= ir_graph->tensor_num)
            {
                TLOG_ERR("invalid output tensor slot: %d idx: %d for node: %d\n", k, tensor_idx, ir_node->idx);
                break;
            }

            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, tensor_idx);

            set_ir_node_output_tensor(ir_node, k, ir_tensor);
        }

        /* load the op parameters */
        if(e->loader != NULL_TM2_OP_LOADER && e->loader(ir_graph, ir_node, tm_node, tm_operator) < 0)
        {
            TLOG_ERR("failed to load op: %d version: %d for node: %d\n", op_type, op_version, ir_node->idx);
            break;
        }
    }

    if(i < v_nodes->v_num)
    {
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int set_graph_io_nodes(struct tm2_serializer* tm2_s, struct ir_graph* ir_graph, struct tm2_priv* priv)
{
    char* mem_base = ( char* )priv->base;
    const TM2_Subgraph* tm_graph = priv->subgraph;
    const TM2_Vector_indices* v_input_nodes = ( TM2_Vector_indices* )(mem_base + tm_graph->offset_vi_input_indices);
    const TM2_Vector_indices* v_output_nodes = ( TM2_Vector_indices* )(mem_base + tm_graph->offset_vi_output_indices);

    int16_t* node_idx = ( int16_t* )sys_malloc(sizeof(int16_t) * v_input_nodes->v_num);

    if(node_idx == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    for(unsigned int i = 0; i < v_input_nodes->v_num; i++)
    {
        node_idx[i] = v_input_nodes->indices[i];
    }

    set_ir_graph_input_node(ir_graph, node_idx, v_input_nodes->v_num);

    sys_free(node_idx);

    node_idx = ( int16_t* )sys_malloc(sizeof(int16_t) * v_output_nodes->v_num);

    for(unsigned int i = 0; i < v_output_nodes->v_num; i++)
    {
        node_idx[i] = v_output_nodes->indices[i];
    }

    set_ir_graph_output_node(ir_graph, node_idx, v_output_nodes->v_num);

    sys_free(node_idx);

    return 0;
}

static int load_graph(struct serializer* s, struct ir_graph* graph, struct tm2_priv* priv)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;

    /* version check */
    if(priv->header->ver_main != TM2_FILE_VER_MAIN)
    {
        TLOG_ERR("model is not version 2\n");
        set_tengine_errno(ENOTSUP);
        return -1;
    }

    if(load_graph_tensors(tm2_s, graph, priv) < 0)
        goto error;

    if(load_graph_nodes(tm2_s, graph, priv) < 0)
        goto error;

    if(set_graph_io_nodes(tm2_s, graph, priv) < 0)
        goto error;

    return 0;

error:
    unload_graph(s, graph, priv, NULL);
    return -1;
}

static int load_model(struct serializer* s, struct ir_graph* graph, const char* fname, va_list ap)
{
    struct stat stat;

    int fd = open(fname, O_RDONLY);

    if(fd < 0)
    {
        set_tengine_errno(ENOENT);
        TLOG_ERR("cannot open file %s\n", fname);
        return -1;
    }

    fstat(fd, &stat);

    int file_len = stat.st_size;

    void* mem_base = mmap(NULL, file_len, PROT_READ, MAP_PRIVATE, fd, 0);

    if(mem_base == MAP_FAILED)
    {
        set_tengine_errno(errno);
        close(fd);
        return -1;
    }

    struct tm2_priv* priv = ( struct tm2_priv* )sys_malloc(sizeof(struct tm2_priv));

    if(priv == NULL)
    {
        set_tengine_errno(ENOMEM);
        close(fd);
        return -1;
    }

    priv->fd = fd;
    priv->mem_len = file_len;
    priv->base = mem_base;
    priv->header = get_tm_file_header(mem_base);
    priv->model = get_tm_file_model(mem_base, priv->header);
    priv->subgraph = get_tm_file_subgraph(mem_base, priv->model);

    graph->serializer = s;
    graph->serializer_priv = priv;
    graph->dev_priv = NULL;

    return load_graph(s, graph, priv);
}

static int load_mem(struct serializer* s, struct ir_graph* graph, const void* addr, int size, va_list ap)
{
    struct tm2_priv* priv = ( struct tm2_priv* )sys_malloc(sizeof(struct tm2_priv));

    if(priv == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    priv->fd = -1;
    priv->mem_len = size;
    priv->base = addr;
    priv->header = get_tm_file_header(addr);
    priv->model = get_tm_file_model(addr, priv->header);
    priv->subgraph = get_tm_file_subgraph(addr, priv->model);

    graph->serializer = s;
    graph->serializer_priv = priv;
    graph->dev_priv = NULL;

    return load_graph(s, graph, priv);
}

static int unload_graph(struct serializer* s, struct ir_graph* graph, void* s_priv, void* dev_priv)
{
    struct tm2_priv* priv = ( struct tm2_priv* )s_priv;

    graph->serializer = NULL;
    graph->serializer_priv = NULL;

    if(priv->fd >= 0)
    {
        munmap(( void* )priv->base, priv->mem_len);
        close(priv->fd);
    }

    sys_free(priv);

    return 0;
}

/* a simple wrapper for type convsion */
static int register_op_loader(struct serializer* s, int op_type, int op_ver, void* op_load_func, void* op_map_func,
                              void* ver_map_func)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;
    tm2_op_loader_t op_load = op_load_func;
    tm2_map_t op_map = op_map_func;
    tm2_map_t ver_map = ver_map_func;

    return register_tm2_op_loader(tm2_s, op_type, op_ver, op_load, op_map, ver_map);
}

static int unregister_op_loader(struct serializer* s, int op_type, int op_ver, void* op_load_func)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;
    tm2_op_loader_t op_load = op_load_func;

    return unregister_tm2_op_loader(tm2_s, op_type, op_ver, op_load);
}

static const char* get_name(struct serializer* s)
{
    return tm2_name;
}

static int const_op_map(int op)
{
    return OP_CONST;
}

static int input_op_map(int op)
{
    return OP_INPUT;
}

static int init_tm2_serializer(struct serializer* s)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;

    tm2_s->loader_list = create_vector(sizeof(struct op_loader_entry), NULL);

    if(tm2_s->loader_list == NULL)
        return -1;

    s->register_op_loader(s, TM2_OPTYPE_INPUTOP, 1, NULL_TM2_OP_LOADER, input_op_map, NULL);
    s->register_op_loader(s, TM2_OPTYPE_CONST, 1, NULL_TM2_OP_LOADER, const_op_map, NULL);

    return 0;
}

static int release_tm2_serializer(struct serializer* s)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;

    s->unregister_op_loader(s, TM2_OPTYPE_INPUTOP, 1, NULL_TM2_OP_LOADER);
    s->unregister_op_loader(s, TM2_OPTYPE_CONST, 1, NULL_TM2_OP_LOADER);

    release_vector(tm2_s->loader_list);

    return 0;
}

static struct tm2_serializer tm2_serializer = {
    .base =
        {
            .get_name = get_name,
            .load_model = load_model,
            .load_mem = load_mem,
            .unload_graph = unload_graph,
            .register_op_loader = register_op_loader,
            .unregister_op_loader = unregister_op_loader,
            .init = init_tm2_serializer,
            .release = release_tm2_serializer,
        },
    .loader_list = NULL,
};

static int reg_tm2_serializer(void* arg)
{
    return register_serializer(( struct serializer* )&tm2_serializer);
}

static int unreg_tm2_serializer(void* arg)
{
    return unregister_serializer(( struct serializer* )&tm2_serializer);
}

REGISTER_MODULE_INIT(MOD_DEVICE_LEVEL, "reg_tm2_serializer", reg_tm2_serializer);
REGISTER_MODULE_EXIT(MOD_DEVICE_LEVEL, "unreg_tm2_serializer", unreg_tm2_serializer);
