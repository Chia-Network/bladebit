//#include "TestUtil.h"
//#include "io/BucketStream.h"
//#include "io/HybridStream.h"
//#include "util/Util.h"
//#include "threading/MTJob.h"
//#include "pos/chacha8.h"
//#include "ChiaConsts.h"
//#include "plotdisk/DiskBufferQueue.h"
//#include "algorithm/RadixSort.h"
//#include "util/CliParser.h"
//
//const uint32 k          = 24;
//const uint32 entryCount = 1ull << k;
//
//Span<uint32> entriesRef;        // Reference entries that we will test again. Generated all at once, then sorted.
//Span<uint32> entriesTest;       // Full set of reference entries, but bucket-sorted first. This is to test against reference entries, excluding I/O issues.
//Span<uint32> entriesTmp;
//Span<uint32> entriesBuffer;
//
//const char*      tmpDir  = "/home/harold/plot/tmp";
//const FileId     fileId  = FileId::FX0;
//DiskBufferQueue* ioQueue = nullptr;
//Fence fence;
//
//using Job = AnonPrefixSumJob<uint32>;
//
//auto seed = HexStringToBytes( "c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835" );
//
//static void AllocateBuffers( const uint32 blockSize );
//static void InitIOQueue( const uint32 numBuckets );
//static void GenerateRandomEntries( ThreadPool& pool );
//
//template<uint32 _numBuckets>
//static void RunForBuckets( ThreadPool& pool, const uint32 threadCount );
//
//template<uint32 _numBuckets>
//static void ValidateTestEntries( ThreadPool& pool, uint32 bucketCounts[_numBuckets] );
//
//
////-----------------------------------------------------------
//TEST_CASE( "bucket-slice-write", "[unit-core]" )
//{
//    SysHost::InstallCrashHandler();
//
//    uint32 maxThreads  = SysHost::GetLogicalCPUCount();
//    uint32 threadStart = maxThreads;
//    uint32 threadEnd   = maxThreads;
//
//    // Parse environment vars
//    std::vector<const char*> args;
//    {
//        const char* e;
//
//        if( (e = std::getenv( "bbtest_thread_count" ) ) )
//        {
//            args.push_back( "--thread_count" );
//            args.push_back( e );
//        }
//        if( (e = std::getenv( "bbtest_end_thread" ) ) )
//        {
//            args.push_back( "--end_thread" );
//            args.push_back( e );
//        }
//        if( std::getenv( "bbtest_all_threads" ) )
//            args.push_back( "--all_threads" );
//    }
//
//
//    if( args.size() > 0 )
//    {
//        CliParser cli( (int)args.size(), args.data() );
//
//        while( cli.HasArgs() )
//        {
//            if( cli.ReadU32( threadStart, "--thread_count" ) ) continue;
//            if( cli.ReadU32( threadEnd, "--end_thread" ) ) continue;
//            if( cli.ArgConsume( "--all_threads" ) )
//            {
//                threadStart = maxThreads;
//                threadEnd   = 1;
//            }
//        }
//
//        FatalIf( threadStart == 0, "threadStart == 0" );
//        // FatalIf( threadEnd > threadStart, "threadEnd > threadStart: %u > %u", threadEnd , threadStart );
//
//        if( threadStart > maxThreads )
//            threadStart = maxThreads;
//        if( threadEnd > threadStart )
//            threadEnd = threadStart;
//    }
//
//    ThreadPool pool( maxThreads );
//
//    for( uint32 i = threadStart; i >= threadEnd; i-- )
//    {
//        Log::Line( "[Threads: %u]", i );
//        RunForBuckets<64> ( pool, i );
//        RunForBuckets<128>( pool, i );
//        RunForBuckets<256>( pool, i );
//        RunForBuckets<512>( pool, i );
//    }
//}
//
////-----------------------------------------------------------
//void AllocateBuffers( const uint32 numBuckets, const uint32 blockSize )
//{
//    const uint32 entriesPerBucket        = entryCount / numBuckets;
//    const uint32 maxEntriesPerSlice      = (uint32)((entriesPerBucket / numBuckets) * BB_DP_ENTRY_SLICE_MULTIPLIER);
//    const uint32 entriesPerSliceAligned  = RoundUpToNextBoundaryT( maxEntriesPerSlice, blockSize ) + blockSize / sizeof( uint32 ); // Need an extra block for when we offset the entries
//    const uint32 entriesPerBucketAligned = entriesPerSliceAligned * numBuckets;
//
//    if( entriesRef.Ptr() == nullptr )
//    {
//        entriesRef  = bbcvirtallocboundednuma_span<uint32>( entryCount );
//        entriesTest = bbcvirtallocboundednuma_span<uint32>( entryCount );
//        entriesTmp  = bbcvirtallocboundednuma_span<uint32>( entryCount );
//    }
//
//    if( entriesBuffer.Ptr() )
//    {
//        bbvirtfreebounded( entriesBuffer.Ptr() );
//        entriesBuffer = Span<uint32>();
//    }
//
//    entriesBuffer = bbcvirtallocboundednuma_span<uint32>( (size_t)( entriesPerBucketAligned ) );
//}
//
////-----------------------------------------------------------
//template<uint32 _numBuckets>
//void RunForBuckets( ThreadPool& pool, const uint32 threadCount )
//{
//    Log::Line( " Testing buckets: %u", _numBuckets );
//    InitIOQueue( _numBuckets );
//    AllocateBuffers( _numBuckets, (uint32)ioQueue->BlockSize( fileId ) );
//    GenerateRandomEntries( pool );
//
//    // Fence* fence = &_fence;
//    fence.Reset( 0 );
//
//    uint32 bucketCounts[_numBuckets] = {};
//    uint32* pBucketCounts = bucketCounts;
//
//    Job::Run( pool, threadCount, [=, &fence]( Job* self ) {
//
//        const uint32 entriesPerBucket = entryCount / _numBuckets;
//        const uint32 bucketBits       = bblog2( _numBuckets );
//        const uint32 bucketShift      = k - bucketBits;
//        const size_t blockSize        = ioQueue->BlockSize( fileId );
//
//        Span<uint32> srcEntries  = entriesRef.Slice();
//        Span<uint32> testEntries = entriesTest.Slice();
//
//
//        uint32 count, offset, end;
//        GetThreadOffsets( self, entriesPerBucket, count, offset, end );
//
//        uint32 pfxSum            [_numBuckets];
//        uint32 offsets           [_numBuckets] = {};
//        uint32 totalCounts       [_numBuckets];
//        uint32 alignedWriteCounts[_numBuckets];
//        uint32 prevOffsets       [_numBuckets] = {};
//
//        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
//        {
//            uint32 counts[_numBuckets] = {};
//
//            // Count entries / bucket
//            for( uint32 i = offset; i < end; i++ )
//                counts[srcEntries[i] >> bucketShift]++;
//
//
//            // Prefix sum
//            memcpy( prevOffsets, offsets, sizeof( offsets ) );
//
//            self->CalculateBlockAlignedPrefixSum<uint32>( _numBuckets, blockSize, counts, pfxSum, totalCounts, offsets, alignedWriteCounts );
//
//            // Distribute entries to buckets
//            for( uint32 i = offset; i < end; i++ )
//            {
//                const uint32 e   = srcEntries[i];
//                const uint32 dst = --pfxSum[e >> bucketShift];
//
//                entriesBuffer[dst] = e;
//            }
//
//            // Caluclate non-aligned test values
//            self->CalculatePrefixSum( _numBuckets, counts, pfxSum, nullptr );
//
//            // Distribute test entries to buckets
//            for( uint32 i = offset; i < end; i++ )
//            {
//                const uint32 e   = srcEntries[i];
//                const uint32 dst = --pfxSum[e >> bucketShift];
//
//                testEntries[dst] = e;
//            }
//
//            // Write to disk
//            if( self->BeginLockBlock() )
//            {
//                ioQueue->WriteBucketElementsT<uint32>( fileId, entriesBuffer.Ptr(), alignedWriteCounts, totalCounts );
//                ioQueue->SignalFence( fence );
//                ioQueue->CommitCommands();
//                fence.Wait();
//
//                auto sliceSizes = ioQueue->SliceSizes( fileId );
//                for( uint32 i = 0; i < _numBuckets; i++ )
//                {
//                    const size_t sliceSize = sliceSizes[bucket][i] / sizeof( uint32 );
//                    ENSURE( sliceSize == totalCounts[i] );
//                }
//
//                // Update total bucket coutns
//                for( uint32 i = 0; i < _numBuckets; i++ )
//                    pBucketCounts[i] += totalCounts[i];
//
//                size_t alignedBucketLength = 0;
//                size_t bucketLength        = 0;
//
//                for( uint32 i = 0; i < _numBuckets; i++ )
//                {
//                    bucketLength        += totalCounts[i];
//                    alignedBucketLength += alignedWriteCounts[i];
//                }
//                ENSURE( entriesPerBucket == bucketLength );
//                ENSURE( alignedBucketLength >= bucketLength );
//
//                // Validate entries against test entries
//                auto test   = testEntries.Slice( 0, bucketLength );
//                auto target = entriesBuffer.Slice( 0, alignedBucketLength );
//
//                for( uint32 i = 0; i < _numBuckets; i++ )
//                {
//                    auto testSlice   = test  .Slice( 0, totalCounts[i] );
//                    auto targetSlice = target.Slice( prevOffsets[i], testSlice.Length() );
//
//                    ENSURE( testSlice.EqualElements( targetSlice ) );
//
//                    test   = test.Slice( totalCounts[i] );
//                    target = target.Slice( alignedWriteCounts[i] );
//                }
//            }
//            self->EndLockBlock();
//
//            testEntries = testEntries.Slice( entriesPerBucket );
//
//            // Write next bucket slices
//            srcEntries = srcEntries.Slice( entriesPerBucket );
//        }
//    });
//
//    // Sort the reference entries
//    RadixSort256::Sort<BB_DP_MAX_JOBS>( pool, entriesRef.Ptr(), entriesTmp.Ptr(), entriesRef.Length() );
//
//    ioQueue->SeekBucket( fileId, 0, SeekOrigin::Begin );
//    ioQueue->CommitCommands();
//    fence.Reset();
//
//    // Total I/O queue bucket length must match our bucket length
//    {
//        auto sliceSizes = ioQueue->SliceSizes( fileId );
//
//        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
//        {
//            size_t bucketLength = 0;
//
//            for( uint32 slice = 0; slice < _numBuckets; slice++ )
//                bucketLength += sliceSizes[slice][bucket];
//
//            ENSURE( bucketLength / sizeof( uint32 ) * sizeof( uint32 ) == bucketLength );
//
//            bucketLength /= sizeof( uint32 );
//            ENSURE( bucketLength == bucketCounts[bucket] );
//        }
//    }
//
//    // Validate against test non-I/O entries first
//    ValidateTestEntries<_numBuckets>( pool, bucketCounts );
//
//    // Read back entries and validate them
//    auto refSlice = entriesRef.Slice();
//
//    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
//    {
//              auto   readBuffer = entriesBuffer.Slice();
//        const size_t capacity   = readBuffer.Length();
//
//        ioQueue->ReadBucketElementsT( fileId, readBuffer );
//        ioQueue->SignalFence( fence );
//        ioQueue->CommitCommands();
//        fence.Wait();
//
//        ENSURE( readBuffer.Length() <= capacity );
//        ENSURE( readBuffer.Length() == bucketCounts[bucket] );
//
//        RadixSort256::Sort<BB_DP_MAX_JOBS>( pool, readBuffer.Ptr(), entriesTmp.Ptr(), readBuffer.Length() );
//
//        if( !readBuffer.EqualElements( refSlice, readBuffer.Length() ) )
//        {
//            // Find first failure
//            for( uint32 i = 0; i < readBuffer.Length(); i++ )
//            {
//                ENSURE( readBuffer[i] == refSlice[i] );
//            }
//        }
//
//        refSlice = refSlice.Slice( readBuffer.Length() );
//    }
//
//    // Delete the file
//    // fence.Reset();
//    // ioQueue->DeleteBucket( FileId::FX0 );
//    // ioQueue->SignalFence( fence );
//    // ioQueue->CommitCommands();
//    // fence.Wait();
//}
//
////-----------------------------------------------------------
//template<uint32 _numBuckets>
//void ValidateTestEntries( ThreadPool& pool, uint32 bucketCounts[_numBuckets] )
//{
//    const uint32 bucketBits  = bblog2( _numBuckets );
//    const uint32 bucketShift = k - bucketBits;
//
//    auto sliceSizes = ioQueue->SliceSizes( fileId );
//
//    auto refEntries = entriesRef;
//
//    // Test the whole buffer first
//    {
//        auto entries = bbcvirtallocboundednuma_span<uint32>( entryCount );
//        entriesTest.CopyTo( entries );
//        ENSURE( entries.EqualElements( entriesTest ) );
//
//        RadixSort256::Sort<BB_DP_MAX_JOBS>( pool, entries.Ptr(), entriesTmp.Ptr(), entries.Length() );
//        ENSURE( entries.EqualElements( entriesRef ) );
//
//        bbvirtfreebounded( entries.Ptr() );
//    }
//
//    uint32 offsets[_numBuckets] = {};
//
//    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
//    {
//        const size_t bucketLength   = bucketCounts[bucket];
//              size_t ioBucketLength = 0;
//
//        auto testBucket = entriesBuffer.Slice( 0, bucketLength );
//        auto writer     = testBucket;
//        auto reader     = entriesTest;
//
//        // Read slices from all buckets
//        for( uint32 slice = 0; slice < _numBuckets; slice++ )
//        {
//            const size_t sliceSize = sliceSizes[slice][bucket] / sizeof( uint32 );
//
//            auto readSlice = reader.Slice( offsets[slice], sliceSize );
//            readSlice.CopyTo( writer );
//
//            for( uint32 i = 0; i < sliceSize; i++ )
//            {
//                ENSURE( writer[i] >> bucketShift == bucket );
//            }
//
//            if( slice + 1 < _numBuckets )
//            {
//                size_t readOffset = 0;;
//                for( uint32 i = 0; i < _numBuckets; i++ )
//                    readOffset += sliceSizes[slice][i] / sizeof( uint32 );
//
//                writer = writer.Slice( sliceSize  );
//                reader = reader.Slice( readOffset );
//            }
//
//            offsets[slice] += sliceSize;
//            ioBucketLength += sliceSize;
//        }
//
//        ENSURE( ioBucketLength == bucketLength );
//        // if( bucket == 7 )
//        // {
//        //     for( uint32 i = 0; i < ioBucketLength; i++ )
//        //         if( testBucket[i] == 1835013 ) BBDebugBreak();
//        // }
//
//        // Sort the bucket
//        RadixSort256::Sort<BB_DP_MAX_JOBS>( pool, testBucket.Ptr(), entriesTmp.Ptr(), testBucket.Length() );
//
//        // Validate it
//        auto refBucket = refEntries.Slice( 0, bucketLength );
//        ENSURE( testBucket.EqualElements( refBucket ) );
//
//        refEntries = refEntries.Slice( bucketLength );
//    }
//}
//
////-----------------------------------------------------------
//void GenerateRandomEntries( ThreadPool& pool )
//{
//    ASSERT( seed.size() == 32 );
//
//    Job::Run( pool, [=]( Job* self ){
//
//        const uint32 entriesPerBlock = kF1BlockSize / sizeof( uint32 );
//        const uint32 blockCount      = entryCount / entriesPerBlock;
//        ASSERT( blockCount * entriesPerBlock == entryCount );
//
//        uint32 count, offset, end;
//        GetThreadOffsets( self, blockCount, count, offset, end );
//
//        byte key[32] = { 1 };
//        memcpy( key+1, seed.data()+1, 31 );
//
//        chacha8_ctx chacha;
//        chacha8_keysetup( &chacha, key, 256, nullptr );
//        chacha8_get_keystream( &chacha, offset, count, (byte*)entriesRef.Ptr() + offset * kF1BlockSize );
//
//        const uint32 mask = ( 1u << k ) - 1;
//        offset *= entriesPerBlock;
//        end    *= entriesPerBlock;
//        for( uint32 i = offset; i < end; i++ )
//            entriesRef[i] &= mask;
//    });
//}
//
//
////-----------------------------------------------------------
//void InitIOQueue( const uint32 numBuckets )
//{
//    // #NOTE: We let it leak after, as we still don't properly terminate
//    // #TODO: Delete the I/O queue after fixing its termination
//
//    const FileSetOptions flags = FileSetOptions::DirectIO | FileSetOptions::Cachable | FileSetOptions::Interleaved;
//    FileSetInitData opts;
//    opts.cacheSize = (size_t)( sizeof( uint32 ) * entryCount * 1.25);
//    opts.cache     = bbvirtallocboundednuma( opts.cacheSize );
//
//    char* nameBuf = new char[64];
//    sprintf( nameBuf, "test-slices-%u", numBuckets );
//
//    byte dummyHeap = 1;
//    ioQueue = new DiskBufferQueue( tmpDir, tmpDir, tmpDir, &dummyHeap, dummyHeap, 1 );
//    ioQueue->InitFileSet( fileId, nameBuf, numBuckets, flags, &opts );
//}
//
