#pragma once
#include "util/Util.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotting/PlotTools.h"
#include "pos/chacha8.h"
#include "threading/MTJob.h"
#include "BitBucketWriter.h"
#include "util/StackAllocator.h"

template<uint32 _numBuckets>
struct DiskF1
{
    using Info = DiskPlotInfo<TableId::Table1, _numBuckets>;
    
    static constexpr uint32 _k = _K;
    static constexpr int64  _entriesPerBucket = (int64)(( 1ull << _k ) / _numBuckets );

    //-----------------------------------------------------------
    inline DiskF1( DiskPlotContext& context, const FileId fileId )
        : _context( context )
        , _entriesPerThread( _entriesPerBucket / (int64)context.f1ThreadCount )
    {
        DiskBufferQueue& ioQueue = *context.ioQueue;

        const uint32 threadCount     = context.f1ThreadCount;
        const uint32 entriesPerBlock = kF1BlockSizeBits / _k;
        const uint64 blocksPerThread = CDiv( _entriesPerThread, entriesPerBlock ) + 1;      // +1 because we might start at an entry which is not the first entry of a block
        const size_t entryAllocCount = blocksPerThread * entriesPerBlock;
        
        // Alloc our data from the heap
        byte* fxBlocks = nullptr;

        StackAllocator stack( context.heapBuffer, context.heapSize );
        fxBlocks   = (byte*)stack.Alloc( _numBuckets * context.tmp2BlockSize, context.tmp2BlockSize );
        _blocks[0] = stack.CAlloc<uint32>( entryAllocCount * threadCount );
        _entries   = stack.CAlloc<uint64>( entryAllocCount * threadCount );
    
        for( uint32 i = 1; i < threadCount; i++ )
            _blocks[i] = _blocks[i-1] + entryAllocCount;

        _bitWriter = BitBucketWriter<_numBuckets>( *context.ioQueue, fileId, fxBlocks );

        Log::Line( "F1 working heap @ %u buckets: %.2lf / %.2lf MiB", 
            _numBuckets, (double)stack.Size() BtoMB, (double)context.heapSize BtoMB );

        // Must have at least n bits left
        const size_t ioBitsPerBucket  = RoundUpToNextBoundary( Info::EntrySizePackedBits * entryAllocCount * threadCount, 64u );
        const size_t ioBytesPerBucket = ioBitsPerBucket / 8; 

        Log::Line( "Minimum IO buffer size required per bucket @ %u buckets: %.2lf MiB", 
            _numBuckets, (double)ioBytesPerBucket BtoMB );
        
        Log::Line( "F1 IO size @ %u buckets: %.2lf MiB", 
            _numBuckets, (double)stack.Remainder() BtoMB );

        FatalIf( stack.Remainder() < ioBitsPerBucket, "Not enough IO reserve size." );

        ioQueue.ResetHeap( stack.Remainder(), stack.Top() );
    }

    //-----------------------------------------------------------
    inline void GenF1()
    {
        using Job = AnonPrefixSumJob<uint64>;
        Job::Run( *_context.threadPool, _context.f1ThreadCount, [=]( Job* self ) {
            
            const uint32 threadCount      = self->JobCount();
            const uint32 id               = self->JobId();
            const uint32 kMinusKExtraBits = _k - kExtraBits;
            const uint32 bucketBits       = bblog2( _numBuckets );
            const uint32 bucketBitShift   = _k - bucketBits;
            const int32  entriesPerBlock  = kF1BlockSizeBits / (int32)_k;
            const size_t entrySizeBits    = Info::YBitSize + _k;    // y + x

            uint32* blocks  = _blocks[id];
            uint64* entries = _entries;

            const int64 trailingEntries = ( 1ll << _k ) - _entriesPerBucket * _numBuckets;
            
            byte key[BB_PLOT_ID_LEN] = { 1 };
            memcpy( key + 1, _context.plotId, BB_PLOT_ID_LEN-1 );

            chacha8_ctx chacha;
            chacha8_keysetup( &chacha, key, 256, NULL );

            auto& bitWriter = _bitWriter;

            for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
            {
                int64  entriesPerThread = _entriesPerThread;
                uint64 x                = (uint64)_entriesPerBucket * bucket + (uint64)entriesPerThread * id;   // Esure this is set before we update entriesPerThread on the last job

                // Last thread grabs trailing entries
                if( self->IsLastThread() )
                {
                    entriesPerThread += _entriesPerBucket - entriesPerThread * (int64)threadCount;

                    if( bucket + 1 == _numBuckets )
                    {
                        // #TODO: Spread across threads instead
                        entriesPerThread += trailingEntries;
                    }
                }

                const uint64 chachaBlock = x / entriesPerBlock;
                const uint64 blockCount  = ( x + entriesPerThread ) / entriesPerBlock - chachaBlock + 1;

                // ChaCha gen
                chacha8_get_keystream( &chacha, chachaBlock, (uint32)blockCount, (byte*)blocks );

                const uint32* yBlocks = blocks + (x - chachaBlock * entriesPerBlock);

                // Distribue to buckets
                uint64 counts     [_numBuckets];
                uint64 pfxSum     [_numBuckets];
                uint64 totalCounts[_numBuckets];

                memset( counts, 0, sizeof( counts ) );
                for( int64 i = 0; i < entriesPerThread; i++ )
                    counts[Swap32( yBlocks[i] ) >> bucketBitShift] ++;

                self->CalculatePrefixSum( _numBuckets, counts, pfxSum, totalCounts );

                if( self->IsControlThread() )
                {
                    self->LockThreads();

                    for( uint32 i = 0; i < _numBuckets; i++ )
                        _context.bucketCounts[0][i] += (uint32)totalCounts[i];

                    // Convert counts to bit sizes
                    for( uint32 i = 0; i < _numBuckets; i++ )
                        totalCounts[i] *= entrySizeBits;
                    
                    _sharedTotalCounts = totalCounts;

                    bitWriter.BeginWriteBuckets( totalCounts );
                    self->ReleaseThreads();
                }
                else
                {
                    self->WaitForRelease();
                }

                const uint32 yBits = Info::YBitSize;
                const uint64 yMask = ( 1ull << yBits ) - 1;

                for( int64 i = 0; i < entriesPerThread; i++ )
                {
                          uint64 y   = Swap32( yBlocks[i] );
                    const uint64 dst = --pfxSum[y >> bucketBitShift];
                    
                    const uint64 xi = ( x + (uint64)i );    // Store bit-compressed already

                    y = ( ( y << kExtraBits ) | ( xi >> kMinusKExtraBits ) ) & yMask;
                    entries[dst] = ( xi << yBits ) | y;
                }


                // Bit-compress each bucket
                uint64 bitsWritten = 0;

                for( uint32 i = 0; i < _numBuckets; i++ )
                {
                    const uint64 offset    = pfxSum[i];
                    const uint64 bitOffset = offset * entrySizeBits - bitsWritten;
                    bitsWritten += _sharedTotalCounts[i];
                    
                    ASSERT( bitOffset + counts[i] * entrySizeBits <= _sharedTotalCounts[i] );

                    BitWriter writer = bitWriter.GetWriter( i, bitOffset );
                    
                    const uint64* entry = entries + offset;
                    const uint64* end   = entry + counts[i];
                    ASSERT( counts[i] >= 2 );

                    // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
                    writer.Write( entry[0], entrySizeBits );
                    writer.Write( entry[1], entrySizeBits );
                    entry += 2;

                    self->SyncThreads();

                    while( entry < end )
                    {
                        writer.Write( *entry, entrySizeBits );
                        entry++;
                    }
                }

                // Write to disk
                if( self->IsControlThread() )
                {
                    self->LockThreads();
                    bitWriter.SubmitAndRelease();
                    self->ReleaseThreads();
                }
                else
                    self->WaitForRelease();
            }

            if( self->IsControlThread() )
                bitWriter.SubmitLeftOvers();
        });

        // Ensure the left-over finish writing
        {
            Fence fence;
            _context.ioQueue->SignalFence( fence );
            _context.ioQueue->CommitCommands();
            fence.Wait();
        }
    }

private:
    DiskPlotContext& _context;
    int64            _entriesPerThread;
    uint64           _blocksPerBucket;
    uint64*          _entries;                      // Work buffer for distributed bucket entries
    uint32*          _blocks[_numBuckets] = { 0 };  // Chacha block buffers for each thread
    FileId           _fileId;
    BitBucketWriter<_numBuckets>  _bitWriter;

    
    const uint64*    _sharedTotalCounts = nullptr;  // Shared across threads
};