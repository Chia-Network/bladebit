#pragma once
#include "threading/ThreadPool.h"
#include <string>

struct Plot128Context
{
    // Single-preallocation buffer
    byte*       buffer;

    ThreadPool* pool;

    std::string tmpPath;

};