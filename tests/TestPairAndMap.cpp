#include "TestUtil.h"
#include "util/jobs/MemJobs.h"
#include "plotdisk/DiskFp.h"
#include "plotdisk/DiskPlotter.h"

#define WORK_TMP_PATH "/mnt/p5510a/disk_tmp/"



// Since pairs (back-pointers) and their corresponding
// maps, have assymetric bucket counts (or not synchronized with each other),
// as the maps have been sorted on y and written back to their original buckets,
// and pairs were never sorted on y.
template<uint32 _numBuckets>
struct DiskPairAndMapReader
{
    static constexpr uint32 _k         = _K;
    static constexpr uint32 _savedBits = bblog2( _numBuckets );
    static constexpr uint32 _pairBits  = _k + 1 - _savedBits + 9;
    static constexpr uint32 _mapBits   = _k + 1 + _k - _savedBits;

    //-----------------------------------------------------------
    inline void LoadNextBucket()
    {
        ASSERT( _lpBucket < _numBuckets );
        ASSERT( _table > TableId::Table1 );

        const FileId pairId = FileId::T1   + (FileId)_table;
        const FileId mapId  = FileId::MAP2 + (FileId)_table - 1;

        const uint64 tableLength      = _context.entryCounts[(int)_table];

        const uint64 pairBucketLength = _context.ptrTableBucketCounts[(int)_table][_lpBucket];
        
        // Load just as many entries we're loading from pairs from the maps buckets
        uint64 mapBucketSize      = _context.bucketCounts[(int)_table][_mapBucket];
        uint64 mapBucketLoadCount = std::min( pairBucketLength, mapBucketSize - _mapBucketEntryOffset );
        ASSERT( mapBucketLoadCount );

        DiskBufferQueue& ioQueue = _context.ioQueue;
        const uint32 loadIdx = _lpBucket & 1; // Same as % 2
        
        const size_t pairBlockSize = ioQueue.BlockSize( pairId );
        const size_t mapBlockSize  = ioQueue.BlockSize( mapId  );

        const size_t pairReadSize  = RoundUpToNextBoundaryT( (size_t)pairBucketLength   * _pairBits, pairBlockSize );
              size_t mapReadSize   = RoundUpToNextBoundaryT( (size_t)mapBucketLoadCount * _mapBits , mapBlockSize  );

        ioQueue.ReadFile( pairId, _lpBucket , _pairBuffers[loadIdx], pairReadSize );
        ioQueue.ReadFile( mapId , _mapBucket, _mapBuffers [loadIdx], pairReadSize );

        _mapBucketEntryOffset += mapBucketLoadCount;
        if( _mapBucketEntryOffset == mapBucketSize )
        {
            ++_mapBucket;

            _mapBucketEntryOffset = pairBucketLength - mapBucketLoadCount;

            if( _mapBucketEntryOffset > 0 )
            {
                // We need to load entries from the next map bucket to pull all the entries needed
                ASSERT( _mapBucket < _numBuckets );

            }
        }

        // ioQueue.SignalFence( _fence );
        ioQueue.CommitCommands();
        
    }

    // void WaitForBucket( const uint32 bucket );

private:
    DiskPlotContext& _context;
    TableId          _table;
    // FileId          _tableId;
    // FileId          _mapId;

    void*           _pairBuffers[2];
    void*           _mapBuffers [2];

    void*           _pairBitsLeftOver;
    uint32          _pairBitCont          = 0;
    
    uint32          _lpBucket             = 0;
    uint32          _mapBucket            = 0;
    uint64          _entriesLoaded        = 0;
    uint64          _mapBucketEntryOffset = 0;
};

//-----------------------------------------------------------
TEST_CASE( "PairsAndMap", "[sandbox]" )
{

}