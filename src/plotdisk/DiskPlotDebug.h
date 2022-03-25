#pragma once

#include "plotting/Tables.h"
#include "DiskPlotConfig.h"
#include "DiskBufferQueue.h"
#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/jobs/IOJob.h"
#include "io/FileStream.h"
#include "util/BitView.h"
#include "algorithm/RadixSort.h"

class ThreadPool;
struct DiskPlotContext;

namespace Debug
{
    void ValidateYFileFromBuckets( FileId yFileId, ThreadPool& pool, DiskBufferQueue& queue, 
                                   TableId table, uint32 bucketCounts[BB_DP_BUCKET_COUNT] );

    void ValidateMetaFileFromBuckets( const uint64* metaA, const uint64* metaB,
                                      TableId table, uint32 entryCount, uint32 bucketIdx, 
                                      uint32 bucketCounts[BB_DP_BUCKET_COUNT] );


    void ValidateLookupIndex( TableId table, ThreadPool& pool, DiskBufferQueue& queue, const uint32 bucketCounts[BB_DP_BUCKET_COUNT] );

    void ValidateLinePoints( DiskPlotContext& context, TableId table, uint32 bucketCounts[BB_DPP3_LP_BUCKET_COUNT] );

    template<TableId table, uint32 numBuckets, typename TYOut>
    void ValidateYForTable( const FileId fileId, DiskBufferQueue& queue, ThreadPool& pool, uint32 bucketCounts[numBuckets] );
}

template<TableId table, uint32 numBuckets, typename TYOut>
inline void Debug::ValidateYForTable( const FileId fileId, DiskBufferQueue& queue, ThreadPool& pool, uint32 bucketCounts[numBuckets] )
{
    Log::Line( "Validating table %u", table+1 );

    uint64 refEntryCount = 0;
    TYOut* yReference    = nullptr;

    // Load File
    {
        char path[1024];
        sprintf( path, "%st%d.y.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );
        Log::Line( " Loading reference table '%s'.", path );
        
        FileStream file;
        FatalIf( !file.Open( path, FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ),
            "Failed to open reference table file '%s'.", path );

        const size_t blockSize = file.BlockSize();
        uint64* blockBuffer = (uint64*)bbvirtalloc( blockSize );

        FatalIf( file.Read( blockBuffer, blockSize ) != (ssize_t)blockSize,
            "Failed to read entry count for reference table with error: %u.", file.GetError() );

        refEntryCount = *blockBuffer;
        yReference = bbcvirtalloc<TYOut>( refEntryCount );

        int err;
        FatalIf( !IOJob::ReadFromFile( file, yReference, sizeof( TYOut ) * refEntryCount, blockBuffer, blockSize, err ),
            "Failed to read table file with error %u.", err );

        bbvirtfree( blockBuffer );
    }

    queue.SeekBucket( fileId, 0, SeekOrigin::Begin );
    queue.CommitCommands();
    
    const size_t blockSize = queue.BlockSize( fileId );

    using Info = DiskPlotInfo<table, numBuckets>;
    const size_t maxBucketEntries = (size_t)Info::MaxBucketEntries;
    const size_t entrySizeBits    = Info::EntrySizePackedBits;
    const size_t entrySize        = CDiv( RoundUpToNextBoundary( entrySizeBits, 64 ), 8 );
    const size_t bucketAllocSize  = RoundUpToNextBoundaryT( entrySize * maxBucketEntries, blockSize );
    
    uint64* bucketBuffers[2] = {
        bbvirtalloc<uint64>( bucketAllocSize ),
        bbvirtalloc<uint64>( bucketAllocSize )
    };

    TYOut* entries = bbcvirtalloc<TYOut>( maxBucketEntries );
    TYOut* tmp     = bbcvirtalloc<TYOut>( maxBucketEntries );

    Fence fence;

    auto LoadBucket = [&]( const uint32 bucket ) {

        void* buffer = bucketBuffers[bucket % 2];

        ASSERT( bucketCounts[bucket] <= maxBucketEntries );
        const size_t size = CDivT( entrySizeBits * bucketCounts[bucket], blockSize * 8 ) * blockSize;

        queue.ReadFile( fileId, bucket, buffer, size );
        queue.SignalFence( fence, bucket + 1 );
        queue.CommitCommands();
    };

    LoadBucket( 0 );

    const TYOut* refReader     = yReference;
          uint64 entriesLoaded = 0;

    std::atomic<uint64> failCount = 0;

    Log::Line( " Validating buckets." );
    for( uint32 b = 0; b < numBuckets; b++ )
    {
        Log::Line( "Bucket %u", b );

        if( b + 1 < numBuckets )
            LoadBucket( b+1 );
        
        fence.Wait( b+1 );
        
        const uint64 bucketEntryCount = bucketCounts[b];

        AnonMTJob::Run( pool, [&]( AnonMTJob* self ) 
        {

            uint64 count, offset, end;
            GetThreadOffsets( self, bucketEntryCount, count, offset, end );
            // count = bucketEntryCount; offset = 0; end = bucketEntryCount;
            
            // Unpack entries
            {
                const size_t yBits = Info::YBitSize;
                const size_t bump  = entrySizeBits - yBits;
                ASSERT( yBits <= 32 );

                BitReader reader( bucketBuffers[b % 2], bucketEntryCount * entrySizeBits, offset * entrySizeBits );
                uint32* writer = ((uint32*)tmp) + offset;

                for( uint64 i = 0; i < count; i++ )
                {
                    writer[i] = (uint32)reader.ReadBits64( yBits );
                    reader.Bump( bump );
                }
            }
        }
        );

        // Sort
        RadixSort256::Sort<BB_DP_MAX_JOBS>( pool, (uint32*)tmp, (uint32*)entries, bucketEntryCount );

        AnonMTJob::Run( pool, [&]( AnonMTJob* self ) 
        {
            uint64 count, offset, end;
            GetThreadOffsets( self, bucketEntryCount, count, offset, end );
            // count = bucketEntryCount; offset = 0; end = bucketEntryCount;

            // Expand to full entry with bucket 
            {
                const size_t yBits      = Info::YBitSize;
                const uint64 bucketMask = ((uint64)b) << yBits;

                const uint32* reader = ((uint32*)tmp) + offset;
                TYOut* writer = entries + offset;
                
                for( uint64 i = 0; i < count; i++ )
                    writer[i] = (TYOut)( bucketMask | (uint64)reader[i] );
            }
            
            // Compare entries
            const TYOut* refs = refReader;
            const TYOut* ys   = entries;

            for( uint64 i = offset; i < end; i++ )
            {
                const TYOut ref = refs[i];
                const TYOut y   = ys[i];

                if( y != ref )
                {
                    if( y != refs[i+1] && ys[i+1] != ref )
                    {
                        Log::Line( " !Entry mismatch @ %llu : %llu != %llu.", 
                            i + entriesLoaded, ref, y );

                        ASSERT( 0 );
                        failCount++;
                    }
                    i++;
                }
            }
        }
        );

        entriesLoaded += bucketCounts[b];
        refReader += bucketEntryCount;
        ASSERT( refReader <= yReference + refEntryCount );
    }

    if( failCount == 0 )
        Log::Line( "*** Table validated successfully! ***" );
    else
        Log::Line( "! Validation failed with %llu / %llu entries failing. !", failCount.load(), refEntryCount );

    bbvirtfree( bucketBuffers[0] );
    bbvirtfree( bucketBuffers[1] );
    bbvirtfree( entries );
    bbvirtfree( tmp     );
}