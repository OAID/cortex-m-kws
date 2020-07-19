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

#include "module.h"
#include "hcl_cpu.h"
#include "cpu_device.h"
#include "vector.h"

struct hcl_dev_entry
{
    hcl_instance_t h;
    struct cpu_device* dev;
};

static struct vector* hcl_dev_list = NULL;

static struct hcl_dev_entry* find_hcl_instance(struct cpu_device* dev)
{
    int n = get_vector_num(hcl_dev_list);

    for(int i = 0; i < n; i++)
    {
        struct hcl_dev_entry* e = ( struct hcl_dev_entry* )get_vector_data(hcl_dev_list, i);

        if(e->dev == dev)
            return e->h;
    }

    return NULL;
}

hcl_instance_t get_hcl_instance(struct cpu_device* dev)
{
    /* TODO: add lock for MT safe */

    struct hcl_dev_entry* entry = find_hcl_instance(dev);

    if(entry)
        return entry;

    hcl_instance_t h = hcl_create_instance(sys_malloc, sys_free);

    if(h == NULL)
        return NULL;

    /* TODO: adjust instance according to dev cpu setting */

    struct hcl_dev_entry e;

    e.h = h;
    e.dev = dev;

    push_vector_data(hcl_dev_list, &e);

    return h;
}

static int init_hcl_dev_list(void* arg)
{
    hcl_dev_list = create_vector(sizeof(struct hcl_dev_entry), NULL);

    if(hcl_dev_list == NULL)
        return -1;

    return 0;
}

static int free_hcl_dev_list(void* arg)
{
    int n = get_vector_num(hcl_dev_list);

    for(int i = 0; i < n; i++)
    {
        struct hcl_dev_entry* e = ( struct hcl_dev_entry* )get_vector_data(hcl_dev_list, i);

        hcl_release_instance(e->h);
    }

    release_vector(hcl_dev_list);

    return 0;
}

REGISTER_MODULE_INIT(MOD_DEVICE_LEVEL, NULL, init_hcl_dev_list);
REGISTER_MODULE_EXIT(MOD_DEVICE_LEVEL, NULL, free_hcl_dev_list);
