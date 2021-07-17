#pragma once
#include "Platform.h"

/// These are lightweight single-process semaphores.
/// (As opposed to system-wide "named" semaphores)
class Semaphore
{
public:

    Semaphore( int initialCount = 0 );
    ~Semaphore();

    void Wait();
    bool Wait( long milliseconds );
    void Release();

    int GetCount();

private:
    SemaphoreId _id;
};