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
#include <syslog.h>

#include "sys_port.h"
#include "tengine_errno.h"
#include "module.h"
#include "tengine_log.h"
#include "cpu_probe.h"
#include "hcl_api.h"

int init_hcl_module(void* arg)
{
    struct probed_cpu_info* probed_cpu = get_probed_cpu_info();

    if(probed_cpu == NULL)
    {
        TLOG_ERR("hcl_module: probe cpu failed\n");
        set_tengine_errno(ENOENT);
        return -1;
    }

    int cpu_num = probed_cpu->cpu_num;
    int* cpu_list = ( int* )sys_malloc(sizeof(int) * cpu_num);
    int* model_list = ( int* )sys_malloc(sizeof(int) * cpu_num);

    int max_freq = 0;
    int master_cpu = 0;

    for(int i = 0; i < cpu_num; i++)
    {
        struct cluster_entry* cluster;

        cpu_list[i] = probed_cpu->cpu_list[i].cpu_id;

        cluster = &probed_cpu->cluster_list[probed_cpu->cpu_list[i].cluster_id];

        model_list[i] = cluster->cpu_model;

        if(cluster->max_freq >= max_freq)
        {
            max_freq = cluster->max_freq;
            master_cpu = cpu_list[i];
        }
    }

    int ret = hcl_init_library(cpu_list, model_list, cpu_num, master_cpu);

    sys_free(cpu_list);
    sys_free(model_list);

    if(ret < 0)
        TLOG_ERR("hcl library init failed\n");

    return ret;
}

int release_hcl_module(void* arg)
{
    hcl_release_library();

    return 0;
}

REGISTER_MODULE_INIT(MOD_DEVICE_LEVEL, NULL, init_hcl_module);
REGISTER_MODULE_EXIT(MOD_DEVICE_LEVEL, NULL, release_hcl_module);
