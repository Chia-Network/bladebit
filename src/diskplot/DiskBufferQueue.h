#pragma once

class DiskBufferQueue
{
public:

    DiskBufferQueue( byte* buffers, size_t bufferSize, uint bufferCount );
    ~DiskBufferQueue();

    // Blocks until there's a buffer available for writing
    byte* GetBuffer();

    // Submit a buffer for writing on the background thread
    void SubmitBuffer();

private:
};