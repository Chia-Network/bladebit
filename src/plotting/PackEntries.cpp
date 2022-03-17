#include "PackEntries.h"
#include "util/BitView.h"

//-----------------------------------------------------------
void EntryPacker::Run()
{
    ASSERT( entryCount >= 2 );

    if( this->unpack )
    {
        DerializeEntries( input, output, entryCount, offset, bitsPerEntry );
    }
    else
    {
        // 2 passes to ensure 2 threads don't write to the same field at the same time
        SerializeEntries( input, output, entryCount - 2, offset, bitsPerEntry );
        this->SyncThreads();
        SerializeEntries( input, output, 2, offset + entryCount - 2, bitsPerEntry );
    }
}


//-----------------------------------------------------------
void EntryPacker::Serialize( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry )
{
    DoJob( pool, threadCount, input, output, entryCount, bitsPerEntry, false );
}

//-----------------------------------------------------------
void EntryPacker::Deserialize( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry )
{
    DoJob( pool, threadCount, input, output, entryCount, bitsPerEntry, true );
}

//-----------------------------------------------------------
void EntryPacker::DoJob( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry,
        bool          unpack )
{
    ASSERT( threadCount <= pool.ThreadCount() );
    MTJobRunner<EntryPacker> jobs( pool );

    const uint64 entriesPerThread = entryCount / threadCount;
    ASSERT( entriesPerThread >= 2 );

    for( uint64 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];
        job.input        = input;
        job.output       = output;
        job.entryCount   = entriesPerThread;
        job.offset       = i * entriesPerThread;
        job.bitsPerEntry = bitsPerEntry;
        job.unpack       = unpack;
    }

    const uint64 trailingEntries = entryCount - entriesPerThread * threadCount;
    jobs[threadCount-1].entryCount += trailingEntries;

    jobs.Run( threadCount );
}


//-----------------------------------------------------------
void EntryPacker::SerializeEntries( const uint64* entries, uint64* bits, const int64 entryCount, const uint64 entryOffset, const size_t bitsPerEntry )
{
    const uint64 bitOffset = entryOffset * bitsPerEntry;
    entries += entryOffset;

    BitWriter writer( bits, (uint64)entryCount * bitsPerEntry, bitOffset );

    for( int64 i = 0; i < entryCount; i++ )
        writer.Write( entries[i], bitsPerEntry );
}

//-----------------------------------------------------------
void EntryPacker::DerializeEntries( const uint64* bits,    uint64* entries, const int64 entryCount, const uint64 entryOffset, const size_t bitsPerEntry )
{
    const uint64 bitOffset      = entryOffset * bitsPerEntry;
    // const uint64 fieldOffset    = bitOffset / 64;
    // const uint64 fieldBitOffset = bitOffset - fieldOffset * 64;
    // const size_t capacity       = RoundUpToNextBoundary( entryCount * bitsPerEntry, 64 );

    entries += entryOffset;
    // bits    += fieldOffset;

    // BitReader reader( bits, capacity, fieldBitOffset );
    BitReader reader( bits, entryCount * bitsPerEntry, bitOffset );

    for( int64 i = 0; i < entryCount; i++ )
        entries[i] = reader.ReadBits64( bitsPerEntry );
}