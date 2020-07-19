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

#include <unistd.h>
#include <string.h>

#include <iostream>
#include <thread>
#include <fstream>
#include <mutex>

#include "tengine_c_api.h"
#include "tengine_log.h"

FILE* os_fp = nullptr;

int total_size = 0;
int total_count = 0;

void output_file(const char* s)
{
    std::string buf("MY:");
    buf = buf + s;

    total_count++;
    total_size += buf.size();

    fwrite(buf.c_str(), buf.size(), 1, os_fp);
}

void output_stdout(const char* s)
{
    std::string buf("MY:");
    buf = buf + s;

    total_count++;
    total_size += buf.size();

    std::cout << buf;
}

void thread0_func(void)
{
    int i = 0;

    while(i++ < 1000)
    {
        TLOG_INFO("111111");
        TLOG_INFO("111111");
        TLOG_INFO("111111");
        TLOG_INFO("111111");
        TLOG_INFO("111111");
        TLOG_INFO("11111\n");
    }
}

void thread1_func(void)
{
    int i = 0;

    while(i++ < 1000)
    {
        TLOG_INFO("aaaaaa");
        TLOG_INFO("aaaaaa");
        TLOG_INFO("aaaaaa");
        TLOG_INFO("aaaaaa");
        TLOG_INFO("aaaaaa");
        TLOG_INFO("aaaaa\n");
    }
}

void thread2_func(void)
{
    int i = 0;

    while(i++ < 1000)
    {
        TLOG_INFO("AAAAAA");
        TLOG_INFO("AAAAAA");
        TLOG_INFO("AAAAAA");
        TLOG_INFO("AAAAAA");
        TLOG_INFO("AAAAAA");
        TLOG_INFO("AAAAA\n");
    }
}

int main(int argc, char* argv[])
{
    if(argv[1])
    {
        os_fp = fopen(argv[1], "w");

        set_log_output(output_file);
    }
    else
    {
        set_log_output(output_stdout);
    }

    std::thread* tr0 = new std::thread(thread0_func);
    std::thread* tr1 = new std::thread(thread1_func);
    std::thread* tr2 = new std::thread(thread2_func);

    tr0->join();
    tr1->join();
    tr2->join();

    delete tr0;
    delete tr1;
    delete tr2;

    if(os_fp)
        fclose(os_fp);

    std::cout << "total_count: " << total_count << "\n";
    std::cout << "total_size: " << total_size << "\n";
    return 0;
}
