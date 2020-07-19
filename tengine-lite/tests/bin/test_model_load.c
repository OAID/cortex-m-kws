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
 * Copyright (c) 2019, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "tengine_c_api.h"

void init_buffer(void* buf, int elem_num, int elem_size, int val)
{
    for(int i = 0; i < elem_num; i++)
    {
        float val0;
        float* fp;
        int16_t* i16;
        char* c;

        if(val > 0)
            val0 = val;
        else
            val0 = i;

        switch(elem_size)
        {
            case 4:
                fp = ( float* )buf;
                fp[i] = val0;
                break;
            case 2:
                i16 = ( int16_t* )buf;
                i16[i] = val0;
                break;
            case 1:
                c = ( char* )buf;
                c[i] = val0;
                break;
        }
    }
}

int main(int argc, char* argv[])
{
    const char* fname = argv[1];

    init_tengine();

    graph_t graph = create_graph(NULL, "tengine", fname);

    if(graph == NULL)
    {
        printf("load graph failed\n");
        return 1;
    }

    dump_graph(graph);

    destroy_graph(graph);

    release_tengine();

    return 0;
}
