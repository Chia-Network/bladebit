#pragma once

#include "plotting/Tables.h"
#include "DiskPlotConfig.h"
#include "DiskBufferQueue.h"
#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPairReader.h"
#include "plotdisk/BlockWriter.h"
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

    template<typename T>
    bool LoadRefTable( const char* path, T*& buffer, uint64& outEntryCount );

    template<typename T>
    bool LoadRefTableByName( const char* fileName, T*& buffer, uint64& outEntryCount );

    template<typename T>
    bool LoadRefTableByName( const char* fileName, Span<T>& buffer );

    template<typename T>
    bool LoadRefTableByName( const TableId table, const char* fileName, Span<T>& buffer );

    void LoadYRefTable( const TableId table, Span<uint64>& buffer );

    void LoadRefLinePointTable( const TableId table, uint64*& buffer, uint64& outEntryCount );

    void LoadRefLPIndexTable( const TableId table, uint32*& buffer, uint64& outEntryCount );

    template<uint32 _numBuckets>
    void ValidatePairs( DiskPlotContext& context, const TableId table );

    template<uint32 _numBuckets>
    void ValidateK32Pairs( const TableId table, DiskPlotContext& context );

    template<uint32 _numBuckets, bool _bounded = false>
    void DumpPairs( const TableId table, DiskPlotContext& context );

    void WriteTableCounts( const DiskPlotContext& context );
    bool ReadTableCounts( DiskPlotContext& context );

    void DumpDPUnboundedY( const TableId table, const uint32 bucket, const DiskPlotContext& context, const Span<uint64> y );
    void LoadDPUnboundedY( const TableId table, Span<uint64>& y );
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

//-----------------------------------------------------------
template<typename T>
inline bool Debug::LoadRefTableByName( const char* fileName, T*& buffer, uint64& outEntryCount )
{
    ASSERT( fileName );

    char path[1024];
    sprintf( path, "%s%s", BB_DP_DBG_REF_DIR, fileName );
    Log::Line( " Loading reference table '%s'.", path );
    return LoadRefTable( path, buffer, outEntryCount );
}

//-----------------------------------------------------------
template<typename T>
inline bool Debug::LoadRefTableByName( const char* fileName, Span<T>& buffer )
{
    // T*& buffer = buffer.values;
    uint64 outEntryCount = 0;
    bool r = LoadRefTableByName( fileName, buffer.values, outEntryCount );
    buffer.length = (size_t)outEntryCount;
    return r;
}

//-----------------------------------------------------------
template<typename T>
inline bool Debug::LoadRefTableByName( const TableId table, const char* fileName, Span<T>& buffer )
{
    ASSERT( fileName );

    char fname[1024];
    sprintf( fname, fileName, (int32)table+1 );
    
    return LoadRefTableByName( fname, buffer.values, buffer.length );
}

//-----------------------------------------------------------
template<typename T>
inline bool Debug::LoadRefTable( const char* path, T*& buffer, uint64& outEntryCount )
{
    FileStream file;

    if( !file.Open( path, FileMode::Open, FileAccess::Read, FileFlags::NoBuffering | FileFlags::LargeFile ) )
        return false;

    const size_t blockSize = file.BlockSize();
    
    uint64* block = (uint64*)bbvirtalloc( blockSize );
    ASSERT( block );

    bool success = false;
    if( file.Read( block, blockSize ) )
    {
        outEntryCount = *block;

        if( buffer == nullptr )
            buffer = bbvirtalloc<T>( RoundUpToNextBoundary( outEntryCount * sizeof( T ), (int)blockSize ) );

        int err = 0;
        success = IOJob::ReadFromFile( file, buffer, sizeof( T ) * outEntryCount, block, blockSize, err );
    }

    bbvirtfree( block );
    return success;
}

//-----------------------------------------------------------
inline void Debug::LoadYRefTable( const TableId table, Span<uint64>& buffer )
{
    ASSERT( table < TableId::Table7 && table > TableId::Table1 );

    char path[1024];
    sprintf( path, "%st%d.y.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );
    Log::Line( " Loading reference Y table '%s'.", path );

    FatalIf( !Debug::LoadRefTable<uint64>( path, buffer.values, (uint64&)buffer.length ), "Failed to load reference Y table." );
}

//-----------------------------------------------------------
inline void Debug::LoadRefLinePointTable( const TableId table, uint64*& buffer, uint64& outEntryCount )
{
    ASSERT( table < TableId::Table7 );

    char path[1024];
    sprintf( path, "%slp.t%d.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );
    Log::Line( " Loading reference line point table '%s'.", path );

    FatalIf( !Debug::LoadRefTable<uint64>( path, buffer, outEntryCount ), "Failed to load reference line point table." );
}

//-----------------------------------------------------------
inline void Debug::LoadRefLPIndexTable( const TableId table, uint32*& buffer, uint64& outEntryCount )
{
    ASSERT( table < TableId::Table7 );

    char path[1024];
    sprintf( path, "%slpmap.t%d.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );
    Log::Line( " Loading reference line point index table '%s'.", path );

    FatalIf( !Debug::LoadRefTable( path, buffer, outEntryCount ), "Failed to load reference line index map table." );
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
inline void Debug::ValidatePairs( DiskPlotContext& context, const TableId table )
{
    uint64 refCount = 0;
    Pair*  refPairs = nullptr;

    {
        char path[1024];
        sprintf( path, "%sp1.t%d.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );
        Log::Line( " Loading reference table '%s'.", path );

        FatalIf( !Debug::LoadRefTable( path, refPairs, refCount ), "Failed to load table." );
    }

    ThreadPool& pool = *context.threadPool;

    const uint64 pairCount = context.entryCounts[(int)table];
    uint64* tmpPairs = bbcvirtalloc<uint64>( std::max( refCount, pairCount ) );
    
    // We need to unpack the entries in such a way that we can sort the pairs in increasing order again.
    auto SwapPairs = [&]( Pair* pairs, const uint64 pairCount ) {

        AnonMTJob::Run( pool, [=]( AnonMTJob* self ) {

            uint64 count, offset, endIdx;
            GetThreadOffsets( self, pairCount, count, offset, endIdx );

                  Pair* src = pairs + offset;
            const Pair* end = pairs + endIdx;

            do {
                if( src->left == 16778218 && src->right == 16778555 ) BBDebugBreak();
                std::swap( src->left, src->right );
            } while( ++src < end );
        });
    };

    Log::Line( "Sorting reference pairs.");
    SwapPairs( refPairs, refCount );
    RadixSort256::Sort<BB_MAX_JOBS>( pool, (uint64*)refPairs, tmpPairs, refCount );
    SwapPairs( refPairs, refCount );

    Log::Line( "Loading our pairs." );
    Pair* pairs = bbcvirtalloc<Pair>( pairCount );

    const uint32 savedBits = bblog2( _numBuckets );
    const uint32 pairBits  = _K + 1 - savedBits + 9;

    {
        const size_t blockSize    = context.ioQueue->BlockSize( FileId::T1 );
        const size_t pairReadSize = (size_t)CDiv( pairCount * pairBits, (int)blockSize*8 ) * blockSize;
        const FileId rTableId     = FileId::T1 + (FileId)table;
        
        Fence fence;

        uint64* pairBitBuffer = bbcvirtalloc<uint64>( pairReadSize );
        context.ioQueue->SeekBucket( rTableId, 0, SeekOrigin::Begin );
        context.ioQueue->ReadFile( rTableId, 0, pairBitBuffer, pairReadSize );
        context.ioQueue->SignalFence( fence );
        context.ioQueue->SeekBucket( rTableId, 0, SeekOrigin::Begin );
        context.ioQueue->CommitCommands();

        fence.Wait();

         AnonMTJob::Run( pool, [=]( AnonMTJob* self ) {

            uint64 count, offset, end;
            GetThreadOffsets( self, pairCount, count, offset, end );

            BitReader reader( pairBitBuffer, pairCount * pairBits, offset * pairBits );
            const uint32 lBits = _K - savedBits + 1;
            const uint32 rBits = 9;
            
            for( uint64 i = offset; i < end; i++ )
            {
                const uint32 left  = (uint32)reader.ReadBits64( lBits );
                const uint32 right = left +  (uint32)reader.ReadBits64( rBits );
                pairs[i] = { .left = left, .right = right };
            }

            // Now set them to global coords
            // #TODO: This won't work for overflow entries, for now test like this
            uint32 entryOffset = 0;
            Pair*  pairReader  = pairs;

            for( uint32 b = 0; b < _numBuckets; b++ )
            {
                self->SyncThreads();

                const uint32 bucketCount = context.ptrTableBucketCounts[(int)table][b];

                GetThreadOffsets( self, (uint64)bucketCount, count, offset, end );
                for( uint64 i = offset; i < end; i++ )
                {
                    pairReader[i].left  += entryOffset;
                    pairReader[i].right += entryOffset;
                }

                entryOffset += context.bucketCounts[(int)table-1][b];
                pairReader  += bucketCount;
            }
        });

        bbvirtfree( pairBitBuffer );
    }

    // Log::Line( "Sorting our pairs.");
    SwapPairs( pairs, pairCount );
    RadixSort256::Sort<BB_MAX_JOBS>( pool, (uint64*)pairs, tmpPairs, pairCount );
    SwapPairs( pairs, pairCount );

    Log::Line( "Comparing pairs." );

    // uint32 b           = 0;
    // uint64 bucketStart = 0;
    // uint32 bucketEnd   = context.ptrTableBucketCounts[(int)table][b];
    // uint64 lOffset     = 0;

    ASSERT( refCount == pairCount );
    const uint64 entryCount = std::min( refCount, pairCount );

    AnonMTJob::Run( pool, [=]( AnonMTJob* self ) {

        uint64 count, offset, end;
        GetThreadOffsets( self, entryCount, count, offset, end );

        for( uint64 i = offset; i < end; i++ )
        {
            const Pair p = pairs   [i];
            const Pair r = refPairs[i];

            if( *(uint64*)&p != *(uint64*)&r )
                ASSERT( 0 );

            // if( i == bucketEnd )
            // {
            //     lOffset     += context.bucketCounts[(int)table][b];
            //     bucketStart = bucketEnd;
            //     bucketEnd   += context.ptrTableBucketCounts[(int)table][++b];
            // }
        }
    });

    
    Log::Line( "OK" );
    bbvirtfree( refPairs );
    bbvirtfree( pairs    );
    bbvirtfree( tmpPairs );
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void Debug::ValidateK32Pairs( const TableId table, DiskPlotContext& context )
{
    Log::Line( "[DEBUG] Validating pairs for table %u", table+1 );

    const uint32 _k                   = 32;
    const uint32 _maxEntriesPerBucket = ( ( 1ull << _k ) / _numBuckets ) * 2;
    const bool   hasMap               = table < TableId::Table7;

    const FileId pairsId = FileId::T1 + (FileId)table;
    const FileId mapId   = FileId::MAP2 + (FileId)table-1;
    context.ioQueue->SeekFile( pairsId, 0, 0, SeekOrigin::Begin );
    context.ioQueue->SeekBucket( mapId, 0, SeekOrigin::Begin );
    context.ioQueue->CommitCommands();

    const uint64 entryCount   = context.entryCounts[(int)table];
    Span<Pair>   reference    = bbcvirtallocboundednuma_span<Pair>( entryCount );
    Span<Pair>   bucketPairs  = bbcvirtallocboundednuma_span<Pair>( _maxEntriesPerBucket );
    Span<uint64> referenceMap;
    Span<uint64> bucketMap;

    if( hasMap )
    {
        referenceMap = bbcvirtallocboundednuma_span<uint64>( entryCount );
        bucketMap    = bbcvirtallocboundednuma_span<uint64>( _maxEntriesPerBucket );
    }

    Log::Line( " Reading reference table." );
    {
        void* block      = nullptr;
        size_t blockSize = 0;

        {
        FileStream dbgDir;
        FatalIf( !dbgDir.Open( BB_DP_DBG_REF_DIR, FileMode::Open, FileAccess::Read ), 
            "Failed to open debug directory at '%s'.", BB_DP_DBG_REF_DIR );

            blockSize = dbgDir.BlockSize();
            block     = bbvirtallocbounded( blockSize );
        }
        
        {
            char path[1024];
            sprintf( path, "%st%d.pairs.tmp", BB_DP_DBG_REF_DIR, (int32)table+1 );

        const size_t readSize = sizeof( Pair ) * entryCount;

        int err;
            FatalIf( !IOJob::ReadFromFile( path, reference.Ptr(), readSize, block, blockSize, err ),
                "Failed to read from reference pairs file at '%s' with error: %d", path, err );
        }

        if( hasMap )
        {
            char path[1024];
            sprintf( path, "%st%d.map.tmp", BB_DP_DBG_REF_DIR, (int32)table+1 );

            const size_t readSize = sizeof( uint64 ) * entryCount;

            int err;
            FatalIf( !IOJob::ReadFromFile( path, referenceMap.Ptr(), readSize, block, blockSize, err ),
                "Failed to read from reference map file at '%s' with error: %d", path, err );
        }

        bbvirtfreebounded( block );
    }

    const size_t heapSize = 41ull GB;
    void*        heap     = bbvirtallocboundednuma( heapSize );

    Fence fence;
    StackAllocator allocator( heap, heapSize );
    DiskPairAndMapReader<_numBuckets, true> reader( context, context.fpThreadCount, fence, table, allocator, !hasMap );

    Span<Pair>   refPairs = reference;
    Span<uint64> refMap   = referenceMap;

    const int32 lTable = (int32)table-1;

    uint32 offset  = 0;
    uint64 iGlobal = 0;

    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        Log::Line( " Validating Bucket %u", bucket );

        Span<Pair>   pairs = bucketPairs;
        Span<uint64> map   = bucketMap;

        Duration _;
        reader.LoadNextBucket();
        pairs.length = reader.UnpackBucket( bucket, pairs.Ptr(), map.Ptr(), _ );

        // Validate
        for( uint64 i = 0; i < pairs.Length(); i++ )
        {
            const Pair ref   = refPairs[i];
            const Pair entry = pairs   [i].AddOffset( offset );

            ASSERT( ref.left == entry.left && ref.right == entry.right );
        }

        // Apply offset & go to next bucket
        offset += context.bucketCounts[lTable][bucket];

        refPairs = refPairs.Slice( pairs.Length() );
        iGlobal += pairs.Length();

        if( hasMap )
            refMap = refMap.Slice( pairs.Length() );
    }

    // Cleanup
    bbvirtfreebounded( heap );
    bbvirtfreebounded( reference.Ptr() );
    bbvirtfreebounded( bucketPairs.Ptr() );

    Log::Line( "[DEBUG] Completed" );
}

//-----------------------------------------------------------
template<uint32 _numBuckets, bool _bounded>
void Debug::DumpPairs( const TableId table, DiskPlotContext& context )
{
    ASSERT( table > TableId::Table1 );

    Log::Line( "[DEBUG] Dumping pairs for table %u", table+1 );

    const bool hasMap = table < TableId::Table7;

    const FileId pairsId = FileId::T1   + (FileId)table;
    const FileId mapId   = FileId::MAP2 + (FileId)table-1;

    char pairsPath[1024];
    char mapPath  [1024];
    FileStream pairsFile, mapFile;

    sprintf( pairsPath, "%st%d.pairs.tmp", BB_DP_DBG_REF_DIR, (int32)table+1 );
    FatalIf( !pairsFile.Open( pairsPath, FileMode::OpenOrCreate, FileAccess::Write, FileFlags::NoBuffering | FileFlags::LargeFile ),
        "Failed to open pairs file at '%s'.", pairsPath );

    void*        block     = bbvirtallocbounded( pairsFile.BlockSize() );
    Span<Pair>   pairTable = bbcvirtallocboundednuma_span<Pair>( context.entryCounts[(int)table] );
    Span<uint64> mapTable;

    context.ioQueue->SeekFile( pairsId, 0, 0, SeekOrigin::Begin );

    if( hasMap )
    {
        sprintf( mapPath, "%st%d.map.tmp", BB_DP_DBG_REF_DIR, (int32)table+1 );
        
        FatalIf( !mapFile.Open( mapPath, FileMode::OpenOrCreate, FileAccess::Write, FileFlags::NoBuffering | FileFlags::LargeFile ),
            "Failed to open map file at '%s'.", mapPath );

        context.ioQueue->SeekBucket( mapId, 0, SeekOrigin::Begin );

        mapTable = bbcvirtallocboundednuma_span<uint64>( context.entryCounts[(int)table] );
    }

    // Submit seek commands
    context.ioQueue->CommitCommands();
    

    const size_t heapSize = 4ull GB;
    byte*        heap     = bbvirtallocboundednuma<byte>( heapSize );

    Fence fence;
    StackAllocator allocator( heap, heapSize );
    
    uint64 pairCount = 0;

    // Read from disk
    {
        Span<Pair>   pairs = pairTable;
        Span<uint64> map   = mapTable;

        DiskPairAndMapReader<_numBuckets, _bounded> reader( context, context.fpThreadCount, fence, table, allocator, !hasMap );
        
        const int lTable = (int)table-1;
        uint32 offset = 0;
        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            reader.LoadNextBucket();
            
            Duration _;
            const uint64 pairsRead = reader.UnpackBucket( bucket, pairs.Ptr(), map.Ptr(), _ );
            ASSERT( pairsRead <= pairs.Length() );
            
            // Offset pairs globally
            Span<Pair> bucketPairs = pairs.SliceSize( pairsRead );

            for( uint64 i = 0; i < bucketPairs.Length(); i++ )
            {
                auto p = bucketPairs[i];
                p.left  += offset;
                p.right += offset;

                bucketPairs[i] = p;
            }

            pairCount += pairsRead;
            pairs = pairs.Slice( pairsRead );

            if( hasMap )
                map = map.Slice( pairsRead);

            offset += context.bucketCounts[lTable][bucket];
        }

        ASSERT( pairs.Length() == 0 );
        ASSERT( map.Length() == 0 );
    }

    // Write unpacked
    {
        int err;
        const size_t sizeWrite = sizeof( Pair ) * pairTable.Length();
        
        FatalIf( !IOJob::WriteToFile( pairsFile, pairTable.Ptr(), sizeWrite, block, pairsFile.BlockSize(), err ),
            "Failed to write to pairs file '%s' with error: %d", pairsPath, err );

        if( hasMap )
        {
            const size_t mapSizeWrite = sizeof( uint64 ) * mapTable.Length();
            FatalIf( !IOJob::WriteToFile( mapFile, mapTable.Ptr(), mapSizeWrite, block, mapFile.BlockSize(), err ),
                "Failed to write to map file '%s' with error: %d", mapPath, err );
        }
    }

    // Cleanup
    bbvirtfreebounded( block );
    bbvirtfreebounded( pairTable.Ptr() );
    bbvirtfreebounded( heap );
    if( hasMap ) 
        bbvirtfreebounded( mapTable.Ptr() );

    Log::Line( "[DEBUG] Completed." );
}

