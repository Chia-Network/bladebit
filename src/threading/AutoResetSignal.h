#pragma once
#include "Platform.h"

class AutoResetSignal
{
public:
    enum
    {
        WaitInfinite = -1
    };

    enum WaitResult
    {
        WaitResultError   = 0,
        WaitResultOK      = 1,
        WaitResultTimeOut = -1
    };

    AutoResetSignal();
    ~AutoResetSignal();

    void Reset();
    void Signal();

    WaitResult Wait( int32 timeoutMS = WaitInfinite );

private:

#if PLATFORM_IS_WINDOWS
    HANDLE _object = NULL;
#elif PLATFORM_IS_UNIX
    struct AutoResetObject
    {
        pthread_mutex_t mutex;
        pthread_cond_t  cond;
        bool            signaled;
    };
    AutoResetObject _object;
#else
    #error Unsupported platform
#endif
};