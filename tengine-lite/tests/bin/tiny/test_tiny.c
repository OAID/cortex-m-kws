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
 * Copyright (c) 2019, OPEN AI Lab
 * Author: haitao@openailab.com
 */
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include "tengine_c_api.h"
#include "tiny_graph.h"

int main(int argc, char* argv[])
{
    init_tengine();

    //   set_log_output(( void* )puts);

    printf("run-time library version: %s\n", get_tengine_version());

    if(request_tengine_version("1.0") < 0)
        return -1;

    const struct tiny_graph* tiny_graph = get_tiny_graph();

    graph_t graph = create_graph(NULL, "tiny", ( void* )tiny_graph);

    if(graph == NULL)
    {
        printf("create graph from tiny model failed\n");
        return -1;
    }

    prerun_graph(graph);

    dump_graph(graph);

    destroy_graph(graph);

    free_tiny_graph(tiny_graph);

    printf("ALL TEST DONE\n");

    release_tengine();
    return 0;
}
