#include "GroupScan.h"
#include "ChiaConsts.h"
#include "threading/MonoJob.h"

struct ScanJob : MTJob<ScanJob>
{
    const uint64* _yBuffer;
          uint32  _entryCount;
          uint32* _groupIndices;
          uint32* _finalGroupIndices;
          uint32  _maxGroups;
          std::atomic<uint64>* _totalGroupCount;
          std::atomic<uint32>* _jobAddSequence;

          // Internal job use
          uint64  _startOffset;
          uint64  _groupCount;

    virtual void Run() override;
};

uint64 ScanBCGroupThread32(
    const uint64* yBuffer,
    const uint64  scanStart,
    const uint64  scanEnd,
    uint32*       groupIndices,
    const uint32  maxGroups,
    const uint32  jobId )
{
    ASSERT( yBuffer );
    ASSERT( groupIndices );
    
    if( maxGroups < 1 )
    {
        ASSERT( 0 );
        return 0;
    }

    uint64 groupCount = 0;
    uint64 prevGroup  = yBuffer[scanStart] / kBC;

    for( uint64 i = scanStart + 1; i < scanEnd; i++ )
    {
        const uint64 group = yBuffer[i] / kBC;
        if( group == prevGroup )
            continue;
        
        ASSERT( group > prevGroup );
        prevGroup = group;
        
        groupIndices[groupCount++] = (uint32)i;

        if( groupCount == maxGroups )
        {
            ASSERT( 0 );    // We ought to always have enough space
                            // So this should be an error
            break;
        }
    }

    return groupCount;
}

uint64 ScanBCGroupMT32( 
    ThreadPool&   pool, 
          uint32  threadCount,
    const uint64* yBuffer,
    const uint32  entryCount,
          uint32* tmpGroupIndices,
          uint32* outGroupIndices,
    const uint32  maxGroups
    )
{
    // Each thread must a minimum # of entries, otherwise, reduce threads until have enough
    const uint64 minEntriesPerThreads = 10000;

    threadCount = std::min( threadCount, entryCount);
    while( threadCount > 1 && entryCount / threadCount < minEntriesPerThreads )
        threadCount--;

    if( maxGroups < threadCount || maxGroups < 3 )
        return 0;

    std::atomic<uint64> groupCount = 0;

    ASSERT( entryCount <= 0xFFFFFFFF );
    ScanJob job = {};
    job._yBuffer           = yBuffer;
    job._entryCount        = entryCount;
    job._groupIndices      = tmpGroupIndices;
    job._finalGroupIndices = outGroupIndices;
    job._maxGroups         = maxGroups;
    job._totalGroupCount   = &groupCount;

    MTJobRunner<ScanJob>::RunFromInstance( pool, threadCount, job );

    return groupCount;
}

void ScanJob::Run()
{
    // First get the starting index for each thread
    uint32 count, offset, _;
    GetThreadOffsets( this, _entryCount, count, offset, _ );
    
    // Find the start of our current group
    {
        const uint64 curGroup = _yBuffer[offset] / kBC;

        while( offset > 0 )
        {
            const uint64 group = _yBuffer[offset-1] / kBC;
            if( group != curGroup )
                break;
            
            offset--;
        }
    }

    _startOffset = offset;
    this->SyncThreads();

    const uint64 end = this->IsLastThread() ? _entryCount : this->GetNextJob()._startOffset;
    ASSERT( end > offset );

    uint32 maxGroups, groupOffset;
    GetThreadOffsets( this, _maxGroups, maxGroups, groupOffset, _ );

    uint32* groupIndices = _groupIndices + groupOffset;

    // Add initial boundary
    groupIndices[0] = offset;
    maxGroups--;

    uint64 groupCount = 1 + ScanBCGroupThread32( _yBuffer, offset, end, groupIndices+1, maxGroups, _jobId );

    // Copy groups into contiguous buffer
    _groupCount = groupCount;
    SyncThreads();

    uint64 copyOffset = 0;
    for( uint32 i = 0; i < _jobId; i++ )
        copyOffset += GetJob( i )._groupCount;

    bbmemcpy_t( _finalGroupIndices + copyOffset, groupIndices, groupCount );

    if( IsLastThread() )
    {
        // Add the last ghost group, but don't actually count it
        if( maxGroups > 0 )
            _finalGroupIndices[copyOffset + groupCount] = _entryCount;
        else
            groupCount--;   // Can't add a ghost group, so our last group will have to serve as ghost
        
        // Do not count the last group, as that one doesn't have any group to match with
        groupCount--;
    }

    _totalGroupCount->fetch_add( groupCount, std::memory_order_relaxed );
}