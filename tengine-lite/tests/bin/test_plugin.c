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
#include <unistd.h>
#include <string.h>

#include "tengine_c_api.h"

const char * plugin_name[]={"hello0.so","hello1.so"};
const char * init_func_name[]={"init","init"};
const char * rel_func_name="release";


int main(int argc, char* argv[])
{
    init_tengine();

    for(int i=0;i<sizeof(plugin_name)/sizeof(const char*);i++)
        load_tengine_plugin(plugin_name[i],plugin_name[i],init_func_name[i]);

    printf("total plugin number: %d\n",get_tengine_plugin_number());

    const char * name=get_tengine_plugin_name(get_tengine_plugin_number()-1);

    printf("last plugin name: %s\n",name);

    for(int i=0;i<sizeof(plugin_name)/sizeof(const char*);i++)
    {
        unload_tengine_plugin(plugin_name[i],rel_func_name);   
    }

    release_tengine();

    return 0;
}


