#pragma once
#include "util/Util.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotting/PlotTools.h"
#include "util/BitView.h"
#include "pos/chacha8.h"
#include "threading/MTJob.h"


template<uint32 _numBuckets>
class BitBucketWriter
{
    DiskBufferQueue& _queue;
    byte*            _blockBuffers;                             // Left-over block buffers
    uint32           _remainderBitCount[_numBuckets] = { 0 };
    BitWriter        _writers          [_numBuckets];
    FileId           _fileId;

    //-----------------------------------------------------------
    inline BitBucketWriter( DiskBufferQueue& queue, const FileId fileId, byte* blockBuffers[_numBuckets] )
        : _queue       ( queue )
        , _blockBuffers( blockBuffers )
        , _fileId      ( fileId )
    {}

    //-----------------------------------------------------------
    inline void BeginWriteBuckets( const uint64 bucketBitSizes[_numBuckets] )
    {
        const size_t fsBlockSize     = _queue.BlockSize( _fileId );
        const size_t fsBlockSizeBits = fsBlockSize * 8;
              size_t allocSize       = 0;

        ASSERT( fsBlockSize > 0 );
        ASSERT( fsBlockSize / 64 * 64 == fsBlockSizeBits );

        for( uint32 i = 0; i < _numBuckets; i++ )
            allocSize += CDiv( bucketBitSizes[i], fsBlockSizeBits ) * fsBlockSizeBits / 8;

        byte* bucketBuffers = _queue.GetBuffer( allocSize, fsBlockSize, true );

        // Initialize our BitWriters
        byte* fields = bucketBuffers;
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            const size_t bufferBitSize = CDiv( bucketBitSizes[i], fsBlockSizeBits ) * fsBlockSizeBits;

            _writers[i] = BitWriter( (uint64*)fields, bufferBitSize );
            
            const uint32 leftOverBitCount = _remainderBitCount[i];
            if( leftOverBitCount > 0 )
                _writers[i].Write( _remainderFields[i], leftOverBitCount );

            _remainderBitCount[i] = 0;
        }
    }

    //-----------------------------------------------------------
    inline void Write( const uint32 bucket, const uint64 bits, const uint32 bitCount )
    {
        ASSERT( bitCount <= 64 );
        _writers[bucket].Write( bits, bitCount );
    }

    //-----------------------------------------------------------
    inline void Submit()
    {
        const size_t fsBlockSize     = _queue.BlockSize( _fileId );
        const size_t fsBlockSizeBits = fsBlockSize * 8;

        // Save any overflow bits
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            BitWriter& writer = _writers[i];

            const size_t bitsToWrite   = writer.Position() / fsBlockSizeBits * fsBlockSizeBits;
            const size_t remainderBits = writer.Position() - remainderBits;
            const size_t bytesToWrite  = bitsToWrite / 8;

            if( remainderBits )
            {
                // Copy fields needed
                const size_t remainderFields = CDiv( remainderBits, 64 );

                memcpy( _blockBuffers[i], ((byte*)writer.Position()) + bytesToWrite, remainderFields * sizeof( uint64 ) );
                _remainderBitCount[i] = remainderBits;
            }

            _queue.WriteFile( _fileId, i, _blockBuffers[i], bytesToWrite );
        }

        _queue.ReleaseBuffer( _blockBuffers[0] );
        _queue.CommitCommands();
    }

    //-----------------------------------------------------------
    inline void WriteLeftOvers()
    {
        size_t allocSize = 0;
    }
};

template<uint32 _numBuckets>
struct DiskF1
{
    static constexpr uint32 _k = _K;

    //-----------------------------------------------------------
    inline DiskF1( DiskPlotContext& context )
        : _entriesPerBucket( (int64)CDiv( _k, _numBuckets ) )
        , _entriesPerThread( CDiv( _entriesPerBucket, (int32)context.f1ThreadCount ) )
    {
    }

    //-----------------------------------------------------------
    inline void GenF1()
    {
        using Job = AnonPrefixSumJob<uint32>;
        Job::Run( *_context.threadPool, _context.f1ThreadCount, [=]( Job* self ) {

            const uint32 id             = self->JobId();
            const uint32 bucketBitShift = _k - bblog2( _numBuckets );
            
            const int32 entriesPerBlock  = (int32)_k / kF1BlockSizeBits;
            const int64 blocksPerThread  = CDiv( _entriesPerThread, entriesPerBlock );

            uint32* blocks = _context.heapBuffer + ( (size_t)blocksPerThread * kF1BlockSize ) * id;

            int64 tableEntryCount = 1ll << _k;
            
            byte key[BB_PLOT_ID_LEN] = { 1 };
            memcpy( key + 1, _context.plotId, BB_PLOT_ID_LEN-1 );

            chacha8_ctx chacha;
            chacha8_keysetup( &chacha, key, 256, NULL );

            uint64 nextX = 0;

            for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
            {
                const int64 bucketEntryCount = std::min( _entriesPerBucket, tableEntryCount );

                int64 entriesPerThread = bucketEntryCount / (int64)_threadCount;
                
                uint64 x = nextX + (uint64)entriesPerThread * id;   // Esure this is set before we update entriesPerThread on the last job

                if( self->IsLastThread() )
                    entriesPerThread = bucketEntryCount - entriesPerThread * (int64)( _threadCount - 1 );

                const uint64 chachaBlock = x / kF1BlockSize;
                const uint64 blockCount  = (uint64)RoundUpToNextBoundaryT( entriesPerThread, entriesPerBlock );

                // ChaCha gen
                chacha8_get_keystream( &chacha, chachaBlock, blockCount, blocks );


                // Distribue to buckets
                uint32 counts     [_numBuckets];
                uint32 pfxSum     [_numBuckets];
                uint32 totalCounts[_numBuckets];

                memset( counts, 0, sizeof( counts ) );
                for( int64 i = 0; i < entriesPerThread; i++ )
                    counts[Swap32( blocks[i] ) >> bucketBitShift] ++;
                
                self->CalculatePrefixSum( _numBuckets, counts, pfxSum, totalCounts );

                if( self->IsControlThread() )
                {
                    for( uint32 i = 0; i < _numBuckets; i++ )
                        _context.bucketCounts[0][i] += totalCounts[i];
                }

                // Grab a buffer for writing
                if( self->IsControlThread() )
                {
                    // _context.ioQueue->GetBufferForId( _fileId, bucket, 
                }
                else
                {
                    self->WaitForRelease();

                }

                for( int64 i = 0; i < entriesPerThread; i++ )
                {
                    const uint64 y   = Swap32( blocks[i] );
                    const uint32 dst = --pfxSum[y >> bucketBitShift];
                    entries[dst] = ( y << kExtraBits ) | ( ( x + (uint64)i ) >> kMinusKExtraBits );
                }


                // Next bucket
                tableEntryCount -= bucketEntryCount;
                nextX           += bucketEntryCount;
            }
        });
    }

    //-----------------------------------------------------------
    void* GetWriteBuffer()
    {
        
    }

private:
    DiskPlotContext& _context;
    uint32           _threadCount;
    int64            _entriesPerBucket;
    int64            _entriesPerThread;
    uint64           _blocksPerBucket;
    uint32*          _blocks;
    FileId           _fileId;

    void*            _writeBuffers[_numBuckets];
    // uint32*          _writeBuffets[]
};