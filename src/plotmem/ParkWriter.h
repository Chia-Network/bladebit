#pragma once
#include "plotting/CTables.h"
#include "ChiaConsts.h"
#include "threading/ThreadPool.h"

struct WriteParkJob
{
    size_t  parkSize;       // #TODO: This should be a compile-time constant?
    uint64  parkCount;      // How many parks to write
    uint64* linePoints;     // Sorted line points to write to the park
    byte*   parkBuffer;     // Buffer into which the parks will be written
    TableId tableId;        // What table are we writing this park to?
};

// Write parks in parallel
// Returns the total size written
template<uint MaxJobs>
size_t WriteParks( ThreadPool& pool, const uint64 length, uint64* linePoints, byte* parkBuffer, TableId tableId );

// Write a single park.
// Returns the offset to the next park buffer
void WritePark( const size_t parkSize, const uint64 count, uint64* linePoints, byte* parkBuffer, TableId tableId );

void WriteParkThread( WriteParkJob* job );

//-----------------------------------------------------------
template<uint MaxJobs>
inline size_t WriteParks( ThreadPool& pool, const uint64 length, uint64* linePoints, byte* parkBuffer, TableId tableId )
{
    const uint   threadCount    = MaxJobs > pool.ThreadCount() ? pool.ThreadCount() : MaxJobs;
    const size_t parkSize       = CalculateParkSize( tableId );
    const uint64 parkCount      = length / kEntriesPerPark;
    const uint64 parksPerThread = parkCount / threadCount;

    uint64 trailingParks        = parkCount - ( parksPerThread * threadCount );
    ASSERT( trailingParks < threadCount );

    const uint64 parkEntriesWritten = parkCount * kEntriesPerPark;
    const uint64 trailingEntries    = length - parkEntriesWritten;
    ASSERT( trailingEntries <= kEntriesPerPark );

    WriteParkJob jobs[MaxJobs];

    uint64* threadLinePoints = linePoints;
    byte*   threadParkBuffer = parkBuffer;

    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.parkSize   = parkSize;
        job.parkCount  = parksPerThread;
        job.linePoints = threadLinePoints;
        job.parkBuffer = threadParkBuffer;
        job.tableId    = tableId;

        // Assign trailer parks accross threads. hehe
        if( trailingParks )
        {
            job.parkCount ++;
            trailingParks --;
        }

        threadLinePoints += job.parkCount * kEntriesPerPark;
        threadParkBuffer += job.parkCount * parkSize;
    }

    ASSERT( !trailingParks );
    pool.RunJob( WriteParkThread, jobs, threadCount );

    // Write trailing entries if any
    if( trailingEntries )
        WritePark( parkSize, trailingEntries, threadLinePoints, threadParkBuffer, tableId );


    const size_t sizeWritten = parkSize * ( parkCount + (trailingEntries ? 1 : 0) );

    return sizeWritten;
}

//-----------------------------------------------------------
inline void WritePark( const size_t parkSize, const uint64 count, uint64* linePoints, byte* parkBuffer, TableId tableId )
{
    ASSERT( count <= kEntriesPerPark );

    uint64* writer = (uint64*)parkBuffer;

    // Write the first LinePoint as a full LinePoint
    uint64 prevLinePoint = linePoints[0];

    *writer++ = Swap64( prevLinePoint );

    // Convert to deltas
    for( uint64 i = 1; i < count; i++ )
    {
        uint64 linePoint = linePoints[i];
        linePoints[i]    = linePoint - prevLinePoint;
        
        prevLinePoint = linePoint;
    }

    // Grab the writing location after the stubs
    const uint64 stubBitSize      = (_K - kStubMinusBits);       // For us, it is 29 bits since K = 32
    const size_t stubSectionBytes = CDiv( (kEntriesPerPark - 1) * stubBitSize, 8 );

    byte* deltaBytesWriter = ((byte*)writer) + stubSectionBytes;

    // Write stubs
    {
        const uint64 stubMask = ((1ULL << stubBitSize) - 1);

        uint64 field = 0;   // Current field to write
        uint   bits  = 0;   // Bits occupying the current field (always shifted to the leftmost bits)

        for( uint64 i = 1; i < count; i++ )
        {
            const uint64 lpDelta = linePoints[i];
            const uint64 stub    = lpDelta & stubMask;

            // Serialize into bits, one uint64 field at a time
            // Always store it all the way to the MSbits
            const uint freeBits = 64 - bits;
            if( freeBits <= stubBitSize )
            {
                // Update the next field bits to what the stub bits that were not written into the current field
                bits = stubBitSize - freeBits;

                // Write what we can (which may be nothing) into the free bits of the current field
                field |= stub >> bits;

                // Store field
                *writer++ = Swap64( field );

                // Write the remaining stub bits (which may be none) into the next field
                // if( bits )
                //     field = stub << ( 64 - bits );
                // else
                //     field = 0;

                // #NOTE: I benchmarked both branching and non-branching
                //        versions and they pretty much run the same in arm CPUs.
                //        Perhaps we should go with the branching method with no undefined behavior.

                // Add a mask to avoid branching here.
                // If the bits is 0, we will shift by 64 bits, which
                // is undefined behavior. Therefore, we mask it so that in the
                // case that bits is 0, we will mask it out to 0.
                const uint remainder = 64 - bits;
                uint64 mask = ( ( 1ull << bits ) - 1 ) << (remainder & 63);
                field = ( stub << remainder ) & mask;
            }
            else
            {
                // The stub completely fits into the current field with room to spare
                field |= stub << (freeBits - stubBitSize);
                bits += stubBitSize;
            }
        }

        // Write any trailing fields
        if( bits > 0 )
            *writer++ = Swap64( field );

        // Zero-out any remaining unused bytes
        const size_t stubUsedBytes  = CDiv( (count - 1) * stubBitSize, 8 );
        const size_t remainderBytes = stubSectionBytes - stubUsedBytes;
        
        memset( deltaBytesWriter - remainderBytes, 0, remainderBytes );
    }
    
    
    // Convert to small deltas
    constexpr uint64 smallDeltaShift = (_K - kStubMinusBits);
    byte* smallDeltas = (byte*)&linePoints[1];
    
    #if DEBUG
        uint64 deltaBitsAccum = 0;
    #endif
    for( uint64 i = 1; i < count; i++ )
    {
        // We re-write the same buffer, but since our
        // byte writer is always behind the read (uint64 size fields),
        // we don't have to worry about corrupting our current data
        smallDeltas[i-1] = (byte)(linePoints[i] >> smallDeltaShift);
        ASSERT( smallDeltas[i-1] < 256 );

        #if DEBUG
            uint sm = smallDeltas[i-1];
            for( int j = 7; j >= 0; j-- )
            {
                uint mask = (1u << j) & sm;
                if( sm & mask )
                    deltaBitsAccum += j+1;
            }
        #endif
    }
    
    #if DEBUG
        const double averageDeltaBits = deltaBitsAccum / (double)2047;
        ASSERT( averageDeltaBits <= kMaxAverageDeltaTable1 );
    #endif

    // Write small deltas
    {
        uint16* deltaSizeWriter = (uint16*)deltaBytesWriter;
        deltaBytesWriter += 2;

        const FSE_CTable* ct = CTables[(int)tableId];

        size_t deltasSize = FSE_compress_usingCTable( 
                                deltaBytesWriter, (count-1) * 8,
                                smallDeltas, count-1, ct );

        if( !deltasSize )
        {
            // Deltas were NOT compressed, we have to copy them raw
            deltasSize = (count-1);
            *deltaSizeWriter = (uint16)(deltasSize | 0x8000);
            memcpy( deltaBytesWriter, smallDeltas, count-1 );
        }
        else
        {
            // Deltas were compressed
            *deltaSizeWriter = (uint16)deltasSize;
        }

        deltaBytesWriter += deltasSize;

        const size_t parkSizeWritten = deltaBytesWriter - parkBuffer;

        if( parkSizeWritten > parkSize )
            Fatal( "Overran park buffer for table %d.", (int)tableId + 1 );
        
        // Zero-out any remaining bytes in the deltas section
        const size_t parkSizeRemainder = parkSize - parkSizeWritten;

        memset( deltaBytesWriter, 0, parkSizeRemainder );
    }
}

//-----------------------------------------------------------
inline void WriteParkThread( WriteParkJob* job )
{
    const size_t  parkSize  = job->parkSize;
    const uint64  parkCount = job->parkCount;
    const TableId tableId   = job->tableId;

    uint64* linePoints = job->linePoints;
    byte*   parkBuffer = job->parkBuffer;

    for( uint64 i = 0; i < parkCount; i++ )
    {
        WritePark( parkSize, kEntriesPerPark, linePoints, parkBuffer, tableId );
        
        linePoints += kEntriesPerPark;
        parkBuffer += parkSize;
    }
}

