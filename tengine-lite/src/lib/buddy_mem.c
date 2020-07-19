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

#include <stdlib.h>
#include <stdint.h>

#include "module.h"
#include "vector.h"
#include "tengine_c_api.h"
#include "tengine_log.h"

#ifdef malloc
#undef malloc
#endif

#ifdef free
#undef free
#endif

#ifdef realloc
#undef realloc
#endif

#define MIN_BUDDY_ORDER 4 /* at least 16 byte to be allocated */
#define MAX_BUDDY_ORDER 24 /* at most 16M byte to be allocated */

struct bucket_header
{
    void* block_list;
    int order;
};

struct block_header
{
    void* next;
};

struct mem_header
{
    void* ptr;
    int order;
    int data_size;
};

int DLLEXPORT insert_mem_block(void* ptr, size_t size);

static struct bucket_header bucket_list[MAX_BUDDY_ORDER + 1];
static int buddy_mem_skipped = 1;
static struct vector* mem_list = NULL;

DECLARE_AUTO_INIT_FUNC(init_buddy_mem);
DECLARE_AUTO_EXIT_FUNC(release_buddy_mem);

void DLLEXPORT set_buddy_mem_status(int disabled)
{
    buddy_mem_skipped = disabled;
}

static void init_buddy_mem(void)
{
    mem_list = create_vector(sizeof(struct mem_header), NULL);

    for(int i = MIN_BUDDY_ORDER; i <= MAX_BUDDY_ORDER; i++)
    {
        bucket_list[i].block_list = NULL;
        bucket_list[i].order = i;
    }
}

static void release_buddy_mem(void)
{
    buddy_mem_skipped = 1;
    release_vector(mem_list);
}

static int find_mem_list(void* ptr)
{
    int n = get_vector_num(mem_list);
    int i;

    for(i = 0; i < n; i++)
    {
        struct mem_header* mem = ( struct mem_header* )get_vector_data(mem_list, i);

        if(mem->ptr == ptr)
            break;
    }

    if(i == n)
        return -1;

    return i;
}

static struct bucket_header* find_proper_bucket(int order)
{
    struct bucket_header* bucket = NULL;

    for(int i = order; i <= MAX_BUDDY_ORDER; i++)
    {
        bucket = &bucket_list[i];

        if(bucket->block_list == NULL)
            continue;

        return bucket;
    }

    return NULL;
}

static inline int cal_order(size_t size)
{
    int order = 0;

    while((1 << order) < size)
        order++;

    if(order < MIN_BUDDY_ORDER)
        return MIN_BUDDY_ORDER;

    if(order > MAX_BUDDY_ORDER)
        return -1;

    return order;
}

void* buddy_malloc(size_t size)
{
    if(buddy_mem_skipped)
        return malloc(size);

    int order = cal_order(size);

    if(order < 0)
        return NULL;

    struct bucket_header* bucket = find_proper_bucket(order);

    if(bucket == NULL)
        return NULL;

    struct block_header* blk = bucket->block_list;

    bucket->block_list = blk->next;

    if(bucket->order > order)
    {
        void* left_ptr = ( char* )blk + (1 << order);
        int left_size = (1 << bucket->order) - (1 << order);

        insert_mem_block(left_ptr, left_size);
    }

    /* for vector functions */
    buddy_mem_skipped = 1;

    struct mem_header mem;
    mem.ptr = blk;
    mem.order = order;
    mem.data_size = size;

    push_vector_data(mem_list, &mem);

    buddy_mem_skipped = 0;

    return blk;
}

static void free_mem_block(void* ptr, int order)
{
    long addr = ( long )ptr;

    for(int i = order; i < MAX_BUDDY_ORDER; i++)
    {
        long buddy_addr = addr ^ (1 << i);

        struct bucket_header* bucket = &bucket_list[i];

        struct block_header* blk = bucket->block_list;
        struct block_header** pblk = ( struct block_header** )&bucket->block_list;

        while(blk)
        {
            if(( void* )blk == ( void* )buddy_addr)
            {
                break;
            }
            pblk = ( struct block_header** )&blk->next;
            blk = blk->next;
        }

        if(blk == NULL)
        {
            struct block_header* new_block = ( struct block_header* )addr;

            new_block->next = bucket->block_list;
            bucket->block_list = new_block;
            return;
        }

        /* found buddy */
        addr = addr & buddy_addr;
        (*pblk) = blk->next;
    }
}

void buddy_free(void* ptr)
{
    int idx = find_mem_list(ptr);

    if(idx < 0)
    {
        free(ptr);
        return;
    }

    struct mem_header* mem_header = ( struct mem_header* )get_vector_data(mem_list, idx);

    free_mem_block(ptr, mem_header->order);

    remove_vector_by_idx(mem_list, idx);
}

void* buddy_realloc(void* ptr, size_t size)
{
    int idx = find_mem_list(ptr);

    if(idx < 0)
    {
        return realloc(ptr, size);
    }

    struct mem_header* mem_header = ( struct mem_header* )get_vector_data(mem_list, idx);
    int space_size = 1 << mem_header->order;

    if(size < space_size)
    {
        mem_header->data_size = size;
        return ptr;
    }

    void* new_ptr = buddy_malloc(size);

    if(new_ptr == NULL)
        return NULL;

    memcpy(new_ptr, ptr, mem_header->data_size);

    buddy_free(ptr);

    return new_ptr;
}

static int get_max_slice_order(long addr, int size)
{
    int order = MAX_BUDDY_ORDER;

    while(order > MIN_BUDDY_ORDER)
    {
        int slice_size = 1 << order;

        if(slice_size <= size && (addr & (slice_size - 1)) == 0)
            break;

        order--;
    }

    return order;
}

int insert_mem_block(void* ptr, size_t size)
{
    if(mem_list == NULL)
        return -1;

    long addr = ( long )ptr;

    if(addr & ((1 << MIN_BUDDY_ORDER) - 1))
    {
        TLOG_ERR("addr does not match the minimum align requirement\n");
        return -1;
    }

    if(size & ((1 << MIN_BUDDY_ORDER) - 1))
    {
        TLOG_ERR("size does not match the minimum align requirement\n");
        return -1;
    }

    int left_size = size;

    while(left_size)
    {
        int slice_order = get_max_slice_order(addr, left_size);

        struct bucket_header* bucket = &bucket_list[slice_order];

        struct block_header* blk = ( struct block_header* )addr;

        blk->next = bucket->block_list;
        bucket->block_list = blk;

        addr += (1 << slice_order);
        left_size -= (1 << slice_order);
    }

    return 0;
}

void dump_bucket_list(void)
{
    for(int i = MIN_BUDDY_ORDER; i <= MAX_BUDDY_ORDER; i++)
    {
        struct bucket_header* bucket = &bucket_list[i];

        if(bucket->block_list == NULL)
            continue;

        struct block_header* blk = bucket->block_list;
        TLOG_INFO("%d:\t", i);

        while(blk != NULL)
        {
            TLOG_INFO("%p\n\t", blk);
            blk = blk->next;
        }

        TLOG_INFO("\n");
    }
}
