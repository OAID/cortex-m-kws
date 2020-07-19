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

#ifndef __TENGINE_OP_H__
#define __TENGINE_OP_H__

#include "tengine_op_name.h"

struct ir_op;

/* op type list */
enum
{
    OP_GENERIC = 0,
    OP_INPUT,
    OP_CONST,
    OP_CONV,
    OP_POOL,
    OP_FC,
    OP_RELU,
    OP_SOFTMAX,
    OP_MOVE,
    OP_BUILTIN_LAST
};

struct op_method
{
    int op_type;
    int op_version;
    int (*init_op)(struct ir_op* op);
    void (*release_op)(struct ir_op* op);
    int (*access_param_entry)(void* param_mem, const char* entry_name, int entry_type, void* buf, int size, int set);
};

int register_op(int op_type, const char* op_name, struct op_method* op_method);
int unregister_op(int op_type, int op_version);

int init_op_registry(void);
void release_op_registry(void);

struct op_method* find_op_method(int op_type, int op_version);

#define AUTO_REGISTER_OP(reg_func) REGISTER_MODULE_INIT(MOD_OP_LEVEL, #reg_func, reg_func)
#define AUTO_UNREGISTER_OP(unreg_func) REGISTER_MODULE_EXIT(MOD_OP_LEVEL, #unreg_func, unreg_func);

#endif
