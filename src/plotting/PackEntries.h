#pragma once
#include "threading/MTJob.h"

struct EntryPacker : MTJob<EntryPacker>
{
    const uint64* input;
    uint64*       output;
    int64         entryCount;
    uint64        offset;
    size_t        bitsPerEntry;
    bool          unpack;

    static void Serialize( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry );

    static void Deserialize( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry );

    void Run() override;

    static void SerializeEntries( const uint64* entries, uint64* bits   , const int64 entryCount, const uint64 entryOffset, const size_t bitsPerEntry );
    static void DerializeEntries( const uint64* bits,    uint64* entries, const int64 entryCount, const uint64 entryOffset, const size_t bitsPerEntry );

private:
    static void DoJob( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry,
        bool          unpack );
};

