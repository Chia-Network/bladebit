#pragma once
#include "PlotContext.h"
#include "plotting/CTables.h"

class MemPhase4
{
    friend class MemPlotter;
public:
    
    MemPhase4( MemPlotContext& context );

    void Run();

    void WriteP7();
    void WriteC1();
    void WriteC2();
    void WriteC3();

private:
    MemPlotContext& _context;
};

struct P7Job
{
    uint64        parkCount;
    const uint32* indices;
    byte*         parkBuffer;
};

struct C12Job
{
    uint64        length;
    const uint32* f7Entries;
    uint32*       writeBuffer;

    #if DEBUG
        uint32 jobIndex;
    #endif
};

struct C3Job
{
    uint64  parkCount;
    uint32* f7Entries;
    byte*   writeBuffer;
};

// P7
template<uint MAX_JOBS>
size_t WriteP7Parallel( ThreadPool& pool, const uint64 length, 
                        const uint32* indices, byte* parkBuffer );

void WriteP7Parks( const uint64 parkCount, const uint32* indices, byte* parkBuffer );
void WriteP7Entries( const uint64 length, const uint32* indices, byte* parkBuffer );


// C1 & C2 tables
size_t WriteC12Parallel( ThreadPool& pool, const uint64 length, 
                         const uint32* f7Entries, uint32* parkBuffer );

template<uint CInterval>
void WriteC12Entries( const uint64 length, const uint32* f7Entries, uint32* c1Buffer );

// C3 parks
uint64 GetC3ParkCount( const uint64 length );
uint64 GetC3ParkCount( const uint64 length, uint64& outLastParkRemainder );

template<uint MAX_JOBS>
size_t WriteC3Parallel( ThreadPool& pool, const uint64 length, uint32* f7Entries, byte* c3Buffer );

void WriteC3Parks( const uint64 parkCount, uint32* f7Entries, byte* writeBuffer );
void WriteC3Park( const uint64 length, uint32* f7Entries, byte* parkBuffer );


///
/// P7
///

//-----------------------------------------------------------
inline void WriteP7Thread( P7Job* job )
{
    WriteP7Parks( job->parkCount, job->indices, job->parkBuffer );
}

//-----------------------------------------------------------
template<uint MAX_JOBS>
inline size_t WriteP7Parallel( ThreadPool& pool, const uint64 length, const uint32* indices, byte* parkBuffer )
{
    const uint32 threadCount     = std::min( pool.ThreadCount(), MAX_JOBS );

    const uint64 parkCount       = length / kEntriesPerPark;           // Number of parks that are completely filled with entries
    const uint64 parksPerThread  = parkCount / threadCount;
    
    uint64 trailingParks = parkCount - ( parksPerThread * threadCount );

    const uint64 trailingEntries   = length - ( parkCount * kEntriesPerPark );
    const uint64 totalParksWritten = parkCount + ( trailingEntries ? 1 : 0 );

    /**
     * #NOTE: 8-byte (uint64s) fields fit perfectly into the 
     *        park size buffer, so we don't have to worry about
     *        race conditions where a thread might write to its last field
     *        which is shared with the first thread's field as well.
     *          33 (K+1) * kEntriesPerPark (2048)
     *          = 67584 / 8
     *          = 8448 / 8
     *          = 1056 64-bit fields
     */
    const size_t parkSize = CDiv( (_K + 1) * kEntriesPerPark, 8 );
    static_assert( parkSize / 8 == 1056 );
    
    P7Job jobs[MAX_JOBS];

    const uint32* threadIndices    = indices;
    byte*         threadParkBuffer = parkBuffer;

    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.parkCount  = parksPerThread;
        job.indices    = threadIndices;
        job.parkBuffer = threadParkBuffer;

        // Assign trailing parks accross threads
        if( trailingParks )
        {
            job.parkCount ++;
            trailingParks --;
        }

        threadIndices    += job.parkCount * kEntriesPerPark;
        threadParkBuffer += job.parkCount * parkSize;
    }

    // Run jobs
    pool.RunJob( WriteP7Thread, jobs, threadCount );

    // Write trailing entries into a park, if we have any
    if( trailingEntries )
    {
        memset( threadParkBuffer, 0, parkSize );
        WriteP7Entries( trailingEntries, threadIndices, threadParkBuffer );
    }

    return totalParksWritten * parkSize;
}

//-----------------------------------------------------------
inline void WriteP7Parks( const uint64 parkCount, const uint32* indices, byte* parkBuffer )
{
    const size_t parkSize = CDiv( (_K + 1) * kEntriesPerPark, 8 );

    for( uint64 i = 0; i < parkCount; i++ )
    {
        WriteP7Entries( kEntriesPerPark, indices, parkBuffer );
        indices    += kEntriesPerPark;
        parkBuffer += parkSize;
    }
}

//-----------------------------------------------------------
inline void WriteP7Entries( const uint64 length, const uint32* indices, byte* parkBuffer )
{
    uint64* fieldWriter = (uint64*)parkBuffer;
    
    // chiapos requires this to have an extra bit for some odd reason.
    // Otherwise we could have copied the buffer as-is.
    const uint32 bitsPerEntry = _K + 1;

    uint64 field = 0;
    uint32 bits  = 0;

    // Like when serializing stubs in the LinePoint parks,
    // we always store it all the way to the MSbits 
    // (shift all the way to the left)
    for( uint64 i = 0; i < length; i++ )
    {
        const uint64 index    = indices[i];
        const uint   freeBits = 64 - bits;

        // Filled a field?
        if( freeBits <= bitsPerEntry )
        {
            // Update the next field bits to what the stub bits that were not written into the current field
            bits = bitsPerEntry - freeBits;
            
            // Write what we can (which may be nothing) into the free bits of the current field
            field |= index >> bits;

            // Store field
            *fieldWriter++ = Swap64( field );

            // Write the remaining bits to the next field
            // if( bits )
            //     field = f7 << ( 64 - bits );
            // else
            //     field = 0;

            // Non-branching mask-based method
            const uint remainder = 64 - bits;
            uint64 mask = ( ( 1ull << bits ) - 1 ) << (remainder & 63);
            field = ( index << remainder ) & mask;
        }
        else
        {
            // The entry completely fits into the current field with room to spare
            field |= index << ( freeBits - bitsPerEntry );
            bits += bitsPerEntry;
        }
    }

    // Write any trailing fields
    if( bits > 0 )
        *fieldWriter++ = Swap64( field );
}


///
/// C1 & C2
///
//-----------------------------------------------------------
template<uint CInterval>
inline void WriteC12Thread( C12Job* job )
{
    WriteC12Entries<CInterval>( job->length, job->f7Entries, job->writeBuffer );
}

//-----------------------------------------------------------
template<uint MAX_JOBS, uint CInterval>
inline size_t WriteC12Parallel( ThreadPool& pool, const uint64 length, 
                                const uint32* f7Entries, uint32* parkBuffer )
{
    const uint32 threadCount      = std::min( pool.ThreadCount(), MAX_JOBS );

    const uint64 parkEntries      = CDiv( length, (int) CInterval );
    const uint64 entriesPerThread = parkEntries / threadCount;
    const uint64 trailingEntries  = parkEntries - (entriesPerThread * threadCount);

    C12Job jobs[MAX_JOBS];

    const uint32* threadf7Entries = f7Entries;
    uint32*       parkWriter      = parkBuffer;

    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];
        
        job.length      = entriesPerThread;
        job.f7Entries   = threadf7Entries;
        job.writeBuffer = parkWriter;

        #if DEBUG
            job.jobIndex = i;
        #endif
        
        threadf7Entries += entriesPerThread * CInterval;
        parkWriter      += entriesPerThread;
    }

    pool.RunJob( WriteC12Thread<CInterval>, jobs, threadCount );

    // Write trailing entries, if any
    if( trailingEntries )
        WriteC12Entries<CInterval>( trailingEntries, threadf7Entries, parkWriter );


    if constexpr ( CInterval == kCheckpoint1Interval * kCheckpoint2Interval )
    {
        // #NOTE: Unfortunately, chiapos infers the size of the C2 table by substracting
        //  the C3 pointer by the C2 pointer. This does not work for us
        //  because since we do block-aligned writes we, our C2 size disk-occupied size
        //  will most likely be greater than the actual C2 size. 
        //  To work around this, we can add a trailing entry with the maximum k32 value size.
        //  This will force chiapos to stop at that point as the f7 is lesser than max k32 value.
        //  #IMPORTANT: This means that we can't have any f7's that are 0xFFFFFFFF!.
        parkWriter[trailingEntries] = 0xFFFFFFFF;
    }
    else
    {
        
        // Write an empty one at the end (compatibility with chiapos)
        parkWriter[trailingEntries] = 0;
    }

    return (parkEntries + 1) * sizeof( uint32 );
}

//-----------------------------------------------------------
template<uint CInterval>
inline void WriteC12Entries( const uint64 length, const uint32* f7Entries, uint32* c1Buffer )
{
    uint64 f7Src = 0;
    for( uint64 i = 0; i < length; i++, f7Src += CInterval )
        c1Buffer[i] = Swap32( f7Entries[f7Src] );
}


///
/// C3
///

//-----------------------------------------------------------
inline uint64 GetC3ParkCount( const uint64 length, uint64& outLastParkRemainder )
{
    const uint64 c3ParkCount       = length / kCheckpoint1Interval;
    const uint64 lastParkRemainder = length - ( c3ParkCount * kCheckpoint1Interval );

    // First 1 is stored in the C1 park, so we ignore it here.
    // Must have at least 1 delta
    if( lastParkRemainder > 1 )
    {
        outLastParkRemainder = lastParkRemainder - 1;
        return c3ParkCount + 1;
    }

    outLastParkRemainder = 0;
    return c3ParkCount;
}

//-----------------------------------------------------------
inline uint64 GetC3ParkCount( const uint64 length )
{
    uint64 remainder;
    return GetC3ParkCount( length, remainder );
}

//-----------------------------------------------------------
inline void WriteC3Thread( C3Job* job )
{
    WriteC3Parks( job->parkCount, job->f7Entries, job->writeBuffer );
}

//-----------------------------------------------------------
template<uint MAX_JOBS>
inline size_t WriteC3Parallel( ThreadPool& pool, const uint64 length, uint32* f7Entries, byte* c3Buffer )
{
    const uint32 threadCount       = std::min( pool.ThreadCount(), MAX_JOBS );

    const uint64 parkCount         = length / kCheckpoint1Interval;
    const uint64 parksPerThread    = parkCount / threadCount;
    
    uint64 trailingParks           = parkCount - ( parksPerThread * threadCount );
    
    const uint64 trailingEntries    = length - ( parkCount * kCheckpoint1Interval );
    
    // We need to check trailingEntries > 1 because the first entry is stored in C1.
    // Therefore, we need to have at least 1 delta to write an extra C3 park.
    const bool   hasTrailingEntries = trailingEntries > 1;
    const uint64 totalParksWritten  = parkCount + ( hasTrailingEntries ? 1 : 0 );
    
    const size_t c3Size = CalculateC3Size();

    C3Job jobs[MAX_JOBS];

    uint32* threadF7Entries = f7Entries;
    byte*   threadC3Buffer  = c3Buffer;

    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.parkCount   = parksPerThread;
        job.f7Entries   = threadF7Entries;
        job.writeBuffer = threadC3Buffer;

        // Distribute trailing parks accross threads
        if( trailingParks )
        {
            job.parkCount ++;
            trailingParks --;
        }

        threadF7Entries += job.parkCount * kCheckpoint1Interval;
        threadC3Buffer  += job.parkCount * c3Size;
    }

    // Run jobs
    pool.RunJob( WriteC3Thread, jobs, threadCount );

    // Write any trailing entries to a park
    if( hasTrailingEntries )
        WriteC3Park( trailingEntries-1, threadF7Entries, threadC3Buffer );

    return totalParksWritten * c3Size;
}

//-----------------------------------------------------------
inline void WriteC3Parks( const uint64 parkCount, uint32* f7Entries, byte* writeBuffer )
{
    const size_t c3Size = CalculateC3Size();

    for( uint64 i = 0; i < parkCount; i++ )
    {
        WriteC3Park( kCheckpoint1Interval-1, f7Entries, writeBuffer );

        f7Entries   += kCheckpoint1Interval;
        writeBuffer += c3Size;
    }
}

//-----------------------------------------------------------
inline void WriteC3Park( const uint64 length, uint32* f7Entries, byte* parkBuffer )
{
    const size_t c3Size = CalculateC3Size();

    // Re-use f7Entries as the delta buffer. 
    // We won't use f7 entries after this, so we can re-write it.
    byte* deltaWriter = (byte*)f7Entries;

    // f7Entries must always start at an interval of kCheckpoint1Interval
    // Therefore its first entry is a C1 entry, and not written as a delta.
    uint32 prevF7 = *f7Entries;
    
    // Convert to deltas
    for( uint64 i = 1; i <= length; i++ )
    {
        const uint32 f7    = f7Entries[i];
        const uint32 delta = f7 - prevF7;
        prevF7 = f7;

        ASSERT( delta < 255 );
        *deltaWriter++ = (byte)delta;
    }

    ASSERT( (uint64)(deltaWriter - (byte*)f7Entries) == length );

    // Serialize them into the C3 park buffer
    const size_t compressedSize = FSE_compress_usingCTable(
        parkBuffer+2, c3Size, (byte*)f7Entries, 
        length, (const FSE_CTable*)CTable_C3
    );
    ASSERT( (compressedSize+2) < c3Size );
    
    // Store size in the first 2 bytes
    *((uint16*)parkBuffer) = Swap16( (uint16)compressedSize );

    // Zero-out remainder (not necessary, though...)
    const size_t remainder = c3Size - (compressedSize + 2);
    if( remainder )
        memset( parkBuffer + compressedSize + 2, 0, remainder );
}


