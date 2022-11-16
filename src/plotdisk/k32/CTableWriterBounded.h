#pragma once
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/MapWriter.h"
#include "util/Util.h"
#include "util/StackAllocator.h"
#include "threading/MTJob.h"
#include "threading/Fence.h"
#include "algorithm/RadixSort.h"
#include "plotting/TableWriter.h"

template<uint32 _numBuckets>
class CTableWriterBounded
{
    static constexpr uint32 _k                = 32;
    static constexpr uint64 _kEntryCount      = 1ull << _k;
    static constexpr uint64 _entriesPerBucket = (uint64)( _kEntryCount / _numBuckets * BB_DP_XTRA_ENTRIES_PER_BUCKET );
public:

    //-----------------------------------------------------------
    CTableWriterBounded( DiskPlotContext& context )
        : _context    ( context )
        , _ioQueue    ( *context.ioQueue )
        , _readFence  ( context.fencePool->RequireFence() )
        , _writeFence ( context.fencePool->RequireFence() )
        , _threadCount( context.cThreadCount )
    {}

    //-----------------------------------------------------------
    ~CTableWriterBounded()
    {
        _context.fencePool->RestoreAllFences();
    }

    //-----------------------------------------------------------
    void Run( IAllocator& allocator )
    {
        auto& context = _context;

        // #TODO: Make the whole thing parallel?
        uint32 c1NextCheckpoint = 0;  // How many C1 entries to skip until the next checkpoint. If there's any entries here, it means the last bucket wrote a
        uint32 c2NextCheckpoint = 0;  // checkpoint entry and still had entries which did not reach the next checkpoint.
                                        // Ex. Last bucket had 10005 entries, so it wrote a checkpoint at 0 and 10000, then it counted 5 more entries, so
                                        // the next checkpoint would be after 9995 entries.

        // These buffers are small enough on k32 (around 1.6MiB for C1, C2 is negligible), we keep the whole thing in memory,
        // while we write C3 to the actual file
        const uint32 c1Interval       = kCheckpoint1Interval;
        const uint32 c2Interval       = kCheckpoint1Interval * kCheckpoint2Interval;

        const uint64 tableLength      = _context.entryCounts[(int)TableId::Table7];
        const uint32 c1TotalEntries   = (uint32)CDiv( tableLength, (int)c1Interval ) + 1; // +1 because chiapos adds an extra '0' entry at the end
        const uint32 c2TotalEntries   = (uint32)CDiv( tableLength, (int)c2Interval ) + 1; // +1 because we add a short-circuit entry to prevent C2 lookup overflows
                                                                                        // #TODO: Remove the extra c2 entry?

        const size_t c1TableSizeBytes = c1TotalEntries * sizeof( uint32 );
        const size_t c2TableSizeBytes = c2TotalEntries * sizeof( uint32 );


        uint32 c3ParkOverflowCount = 0;                 // Overflow entries from a bucket that did not make it into a C3 park this bucket. Saved for the next bucket.
        uint32 c3ParkOverflow[kCheckpoint1Interval];    // They are then copied to a "prefix region" in the f7 buffer of the next park.

        size_t c3TableSizeBytes = 0;                    // Total size of the C3 table

        // Allocate buffers
        AllocIOBuffers( allocator );
        
        Span<uint64> _mapOut = allocator.CAllocSpan<uint64>( _entriesPerBucket );

        // Use same buffers as map tmp
        uint32* f7Tmp         = _mapOut.template As<uint32>().SliceSize( _entriesPerBucket ).Ptr();
        uint32* idxTmp        = _mapOut.template As<uint32>().Slice( _entriesPerBucket ).Ptr();

        Span<uint32> c1Buffer = allocator.CAllocSpan<uint32>( c1TotalEntries );
        Span<uint32> c2Buffer = allocator.CAllocSpan<uint32>( c2TotalEntries );


        ///
        /// Begin
        ///
        
        // Prepare read bucket files
        _ioQueue.SeekBucket( FileId::FX0   , 0, SeekOrigin::Begin );
        _ioQueue.SeekBucket( FileId::INDEX0, 0, SeekOrigin::Begin );

        // Seek to the start of the C3 table instead of writing garbage data.
        _ioQueue.SeekFile( FileId::PLOT, 0, (int64)(c1TableSizeBytes + c2TableSizeBytes), SeekOrigin::Current );
        _ioQueue.CommitCommands();

        // Load initial bucket
        LoadBucket( 0 );

        uint32* c1Writer = c1Buffer.Ptr();
        uint32* c2Writer = c2Buffer.Ptr();

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            // Load next bucket in the background
            LoadBucket( bucket + 1 );

            // Wait for our bucket data to be available
            auto bucketData = ReadBucket( bucket );

            Span<uint32> f7      = bucketData.first;
            Span<uint32> indices = bucketData.second;
            ASSERT( f7.Length() == indices.Length() );
            ASSERT( f7.Length() == context.bucketCounts[(int)TableId::Table7][bucket] );

            const uint32 bucketLength = (uint32)f7.Length();

            // Sort on f7
            RadixSort256::SortWithKey<BB_DP_MAX_JOBS>( *context.threadPool, _threadCount, f7.Ptr(), f7Tmp, indices.Ptr(), idxTmp, bucketLength );

            // Write the map top disk
            WriteMap( bucket, indices, _mapOut );

            /// Now handle f7 and write them into C tables
            /// We will set the addersses to these tables accordingly.
            
            // Write C1
            {
                ASSERT( bucketLength > c1NextCheckpoint );

                // #TODO: Do C1 multi-threaded? For now jsut single-thread it...
                for( uint32 i = c1NextCheckpoint; i < bucketLength; i += c1Interval )
                    *c1Writer++ = Swap32( f7[i] );
                
                // Track how many entries we covered in the last checkpoint region
                const uint32 c1Length          = bucketLength - c1NextCheckpoint;
                const uint32 c1CheckPointCount = CDiv( c1Length, (int)c1Interval );

                c1NextCheckpoint = c1CheckPointCount * c1Interval - c1Length;
            }

            // Write C2
            {
                // C2 has so few entries on k=32 that there's no sense in doing it multi-threaded
                if( c2NextCheckpoint >= bucketLength )
                    c2NextCheckpoint -= bucketLength;   // No entries to write in this bucket
                else
                {
                    for( uint32 i = c2NextCheckpoint; i < bucketLength; i += c2Interval )
                        *c2Writer++ = Swap32( f7[i] );
                
                    // Track how many entries we covered in the last checkpoint region
                    const uint32 c2Length          = bucketLength - c2NextCheckpoint;
                    const uint32 c2CheckPointCount = CDiv( c2Length, (int)c2Interval );

                    c2NextCheckpoint = c2CheckPointCount * c2Interval - c2Length;
                }
            }

            // Write C3
            {
                const bool isLastBucket = bucket == _numBuckets-1;

                uint32* c3F7           = f7.Ptr();
                uint32  c3BucketLength = bucketLength;

                if( c3ParkOverflowCount )
                {
                    // Copy our overflow to the prefix region of our f7 buffer
                    c3F7           -= c3ParkOverflowCount;
                    c3BucketLength += c3ParkOverflowCount;

                    memcpy( c3F7, c3ParkOverflow, sizeof( uint32 ) * c3ParkOverflowCount );
                    c3ParkOverflowCount = 0;
                }
                
                
            #if _DEBUG
                // #TODO: TEST: Remove this
                // Dump f7's that have the value of 0xFFFFFFFF for now,
                // this is just for compatibility with RAM bladebit
                // for testing plots against it.
                // if( isLastBucket )
                // {
                //     while( c3F7[c3BucketLength-1] == 0xFFFFFFFF )
                //         c3BucketLength--;
                // }
            #endif

                // See TableWriter::GetC3ParkCount for details
                uint32 parkCount       = c3BucketLength / kCheckpoint1Interval;
                uint32 overflowEntries = c3BucketLength - ( parkCount * kCheckpoint1Interval );

                // Greater than 1 because the first entry is excluded as it is written in C1 instead.
                if( isLastBucket && overflowEntries > 1 )
                {
                    overflowEntries = 0;
                    parkCount++;
                }
                else if( overflowEntries && !isLastBucket )
                {
                    // Save any entries that don't fill-up a full park for the next bucket
                    memcpy( c3ParkOverflow, c3F7 + c3BucketLength - overflowEntries, overflowEntries * sizeof( uint32 ) );
                    
                    c3ParkOverflowCount = overflowEntries;
                    c3BucketLength -= overflowEntries;
                }

                const size_t c3BufferSize = CalculateC3Size() * parkCount;
                      byte*  c3Buffer     = GetWriteBuffer( bucket );

                // #NOTE: This function uses re-writes our f7 buffer, so ensure it is done after
                //        that buffer is no longer needed.
                const size_t sizeWritten = TableWriter::WriteC3Parallel<BB_DP_MAX_JOBS>( *_context.threadPool, 
                                                _threadCount, c3BucketLength, c3F7, c3Buffer );
                ASSERT( sizeWritten == c3BufferSize );

                c3TableSizeBytes += sizeWritten;

                // Write the C3 table to the plot file directly
                _ioQueue.WriteFile( FileId::PLOT, 0, c3Buffer, c3BufferSize );
                _ioQueue.SignalFence( _writeFence, bucket );
                _ioQueue.CommitCommands();
            }
        }

        // Seek back to the begining of the C1 table and
        // write C1 and C2 buffers to file, then seek back to the end of the C3 table

        c1Buffer[c1TotalEntries-1] = 0;          // Chiapos adds a trailing 0
        c2Buffer[c2TotalEntries-1] = 0xFFFFFFFF; // C2 overflow protection      // #TODO: Remove?

        _readFence.Reset( 0 );

        _ioQueue.SeekBucket( FileId::PLOT, -(int64)( c1TableSizeBytes + c2TableSizeBytes + c3TableSizeBytes ), SeekOrigin::Current );
        _ioQueue.WriteFile( FileId::PLOT, 0, c1Buffer.Ptr(), c1TableSizeBytes );
        _ioQueue.WriteFile( FileId::PLOT, 0, c2Buffer.Ptr(), c2TableSizeBytes );
        _ioQueue.SeekBucket( FileId::PLOT, (int64)c3TableSizeBytes, SeekOrigin::Current );
        _ioQueue.SignalFence( _readFence, 1 );
        _ioQueue.CommitCommands();

        // Save C table addresses into the plot context.
        // And set the starting address for the table 1 to be written
        const size_t headerSize = _ioQueue.PlotHeaderSize();

        _context.plotTablePointers[7] = headerSize;                                       // C1
        _context.plotTablePointers[8] = _context.plotTablePointers[7] + c1TableSizeBytes; // C2
        _context.plotTablePointers[9] = _context.plotTablePointers[8] + c2TableSizeBytes; // C3
        _context.plotTablePointers[0] = _context.plotTablePointers[9] + c3TableSizeBytes; // T1

        // Save sizes
        _context.plotTableSizes[7] = c1TableSizeBytes;
        _context.plotTableSizes[8] = c2TableSizeBytes;
        _context.plotTableSizes[9] = c3TableSizeBytes;

        // Wait for all commands to finish
        _readFence.Wait( 1 );
    }

    inline Duration IOWait() const { return _tableIOWait; }
    
private:

    //-----------------------------------------------------------
    void AllocIOBuffers( IAllocator& allocator )
    {
        _mapWriter = MapWriter<_numBuckets, false>( _ioQueue, FileId::MAP7, allocator, 
                                                    _entriesPerBucket, _context.tmp1BlockSize, 
                                                    _context.fencePool->RequireFence(), _tableIOWait );

        const size_t blockSize         = _context.tmp2BlockSize;
        const uint32 prefixRegionCount = (uint32)RoundUpToNextBoundaryT( (size_t)kCheckpoint1Interval, blockSize / sizeof( uint32 ) );

        // Allocate and init read buffer views
        uint32* f7[2] = {
            allocator.CAlloc<uint32>( _entriesPerBucket + prefixRegionCount, blockSize ),
            allocator.CAlloc<uint32>( _entriesPerBucket + prefixRegionCount, blockSize )
        };

        // Assign after prefix region
        for( uint32 i = 0; i < _numBuckets; i++ )
            _f7ReadBuffer[i] = Span<uint32>( f7[i&1] + prefixRegionCount, _entriesPerBucket );     // & 1 == % 2

        // Allocate and init index buffer views
        Span<uint32> indices[2] = {
            allocator.CAllocSpan<uint32>( _entriesPerBucket, blockSize ),
            allocator.CAllocSpan<uint32>( _entriesPerBucket, blockSize )
        };

        for( uint32 i = 0; i < _numBuckets; i++ )
            _idxReadBuffer[i] = indices[i&1];

        
        // Allocate write buffers
        _f7WriteBuffer[0] = allocator.CAllocSpan<uint32>( _entriesPerBucket, blockSize );
        _f7WriteBuffer[1] = allocator.CAllocSpan<uint32>( _entriesPerBucket, blockSize );
    }

    //-----------------------------------------------------------
    void WriteMap( const uint32 bucket, Span<uint32> indices, Span<uint64> mapBuffer )
    {
        auto* mapWriter = &_mapWriter;
        
        using Job = AnonPrefixSumJob<uint32>;

        Job::Run( *_context.threadPool, _threadCount, [=]( Job* self ) {
            
            uint64 outMapBucketCounts[_numBuckets];
            uint32 totalCounts       [_numBuckets];

            mapWriter->WriteJob( self, bucket, _mapOffset, indices, mapBuffer,
                                 outMapBucketCounts, totalCounts, _mapBitCounts );

            if( bucket == _numBuckets-1 && self->IsControlThread() )
                mapWriter->SubmitFinalBits();
        });

        _mapOffset += indices.Length();
    }

    //-----------------------------------------------------------
    inline void LoadBucket( const uint32 bucket )
    {
        if( bucket >= _numBuckets )
            return;

        _ioQueue.ReadBucketElementsT( FileId::FX0   , true, _f7ReadBuffer [bucket] );
        _ioQueue.ReadBucketElementsT( FileId::INDEX0, true, _idxReadBuffer[bucket] );
        _ioQueue.SignalFence( _readFence, bucket + 1 );
        _ioQueue.CommitCommands();
    }

    //-----------------------------------------------------------
    inline std::pair<Span<uint32>, Span<uint32>> ReadBucket( const uint32 bucket )
    {
        _readFence.Wait( bucket + 1, _tableIOWait );
        return std::pair<Span<uint32>, Span<uint32>>( _f7ReadBuffer[bucket], _idxReadBuffer[bucket] );
    }

    //-----------------------------------------------------------
    inline byte* GetWriteBuffer( const uint32 bucket )
    {
        if( bucket > 1 )
            _writeFence.Wait( bucket - 2, _tableIOWait );

        return (byte*)_f7WriteBuffer[bucket & 1].Ptr();
    }
    
private:
    DiskPlotContext& _context;
    DiskBufferQueue& _ioQueue;
    Fence&           _readFence;
    Fence&           _writeFence;

    Span<uint32>     _f7ReadBuffer [_numBuckets];
    Span<uint32>     _idxReadBuffer[_numBuckets];
    Span<uint32>     _f7WriteBuffer[2];

    MapWriter<_numBuckets, false> _mapWriter;
    uint64                        _mapOffset = 0;
    uint64                        _mapBitCounts[_numBuckets] = {};

    Duration        _tableIOWait    = Duration::zero();
    uint32          _threadCount    = 0;
};

