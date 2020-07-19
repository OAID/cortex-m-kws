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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <unistd.h>
#include <sys/time.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"
extern "C" {
#include "tengine_ir.h"
}

const char* model_file = "./models/mobilenet.tm";

int main(int argc, char* argv[])
{
    init_tengine();

    std::cout << "run-time library version: " << get_tengine_version() << "\n";

    if(request_tengine_version("1.0") < 0)
        return -1;

    graph_t graph = create_graph(nullptr, "tengine", model_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    //dump_graph(graph);

    void * graph_mem;
    int graph_size;

    if(pack_ir_graph((struct ir_graph *)graph,&graph_mem,&graph_size)<0)
    {
        std::cout<<"pack graph failed\n";
        return -1;
    }

    std::cout<<"packed graph: "<<graph_mem<<" size: "<<graph_size<<"\n";
    struct ir_graph * new_graph=unpack_ir_graph(graph_mem,graph_size);

    if(new_graph==NULL)
        return -1;

    dump_graph(new_graph);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
