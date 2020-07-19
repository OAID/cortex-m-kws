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
#include <malloc.h>

#include "module.h"
#include "sys_port.h"

#ifdef malloc
#undef malloc
#endif

#ifdef free
#undef free
#endif

#ifdef realloc
#undef realloc
#endif

extern void (*enable_intern_allocator)(void);
extern void (*disable_intern_allocator)(void);

static void* mem_addr;
static int mem_size;

static void setup_buddy_mem(void)
{
    mem_size = 32 << 20;
    mem_addr = malloc(mem_size + (1 << 20));

    long addr = ( long )(mem_addr);

    addr += (1 << 20);
    addr &= ~((1 << 20) - 1);

    insert_mem_block(( void* )addr, mem_size);

    set_buddy_mem_status(0);
}

static void release_buddy_mem(void)
{
    set_buddy_mem_status(1);
    free(mem_addr);
}

/*
    enable_intern_allocator will be called in init_tengine
    disable_intern_allocator will be called in release_tengine
*/

DECLARE_AUTO_INIT_FUNC(enable_buddy);
DECLARE_AUTO_EXIT_FUNC(disable_buddy);

static void enable_buddy(void)
{
    enable_intern_allocator = setup_buddy_mem;
}

static void disable_buddy(void)
{
    disable_intern_allocator = release_buddy_mem;
}
