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

#include "tengine_c_api.h"

void dump_bucket_list(void);
void* buddy_malloc(size_t size);
void buddy_free(void* ptr);

int main(int argc, char* argv[])
{
    init_tengine();

    dump_bucket_list();
    void* ptr = buddy_malloc(1024);

    printf("ptr %p\n", ptr);

    dump_bucket_list();

    void* ptr2 = buddy_malloc(4023);

    printf("ptr2 %p\n", ptr2);

    dump_bucket_list();

    printf("return ptr\n");

    buddy_free(ptr);
    dump_bucket_list();

    printf("return ptr2\n");

    buddy_free(ptr2);
    dump_bucket_list();

    release_tengine();

    dump_bucket_list();
    return 0;
}
