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

#include "parameter.h"

struct test
{
    int a;
    float b;
    const char* c;
};

DEFINE_PARM_PARSE_ENTRY(test, a, b, c);

int main(int argc, char* argv[])
{
    struct test t0;

    t0.a = 100;
    t0.b = 20;
    t0.c = "hello, world";

    float b0;
    char* c0;
    char* c1 = "this is new";

    access_param_entry(&t0, "b", GET_PARAM_ENTRY_TYPE(b0), &b0, sizeof(b0), 0);
    access_param_entry(&t0, "c", GET_PARAM_ENTRY_TYPE(c0), &c0, sizeof(c0), 0);

    printf("b0=%f\n", b0);
    printf("c0=%s\n", c0);

    access_param_entry(&t0, "c", GET_PARAM_ENTRY_TYPE(c1), &c1, sizeof(c1), 1);

    printf("[%s]\n", t0.c);

    b0 = 1;

    access_param_entry(&t0, "b", GET_PARAM_ENTRY_TYPE(b0), &b0, sizeof(b0), 1);

    printf("test.b is %f\n", t0.b);

    return 0;
}
