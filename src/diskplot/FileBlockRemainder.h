#pragma once
#include "DiskPlotConfig.h"
#include "Fence.h"

class FileBlockRemainder
{
public:
    FileBlockRemainder();
    ~FileBlockRemainder();

    void SaveRemainders( const byte* buckets, const uint32* sizes );
    void WriteFinalRemainders();

private:
    size_t _fileBlockSize;
    byte*  _buffers    [BB_DP_BUCKET_COUNT][2];     // 2 buffers ofr fileBlockSize per bucket
    uint32 _bufferSizes[BB_DP_BUCKET_COUNT];        // Current amount of bytes stored in the front buffers
    Fence  _fence;
    uint   _fenceSequence;
};

