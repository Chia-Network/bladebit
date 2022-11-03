#pragma once
#include <atomic>
#include "PlotContext.h"

///
/// Job structs
///

struct SyncedJob
{   
    uint               _threadId;
    uint               _threadCount;
    std::atomic<uint>* _readyCount;
    std::atomic<uint>* _releaseLock;
    
    inline void Init( const uint threadId, const uint threadCount, std::atomic<uint>& signal, std::atomic<uint>& releaseLock );
    inline void WaitForThreads();
};

struct LPJob : public SyncedJob
{
    uint32* lTable;             // Left table (x for Table 1, or lookup table to new indices otherwise)
    
    uint64  length;             // R Table length
    uint64  offset;             // Offset in R table to our entries
    Pair*   rTable;             // R table
    uint64* lpBuffer;           // Where to store the pruned Pairs as line points

    LPJob*  jobs;               // All threads participating in this job

    const byte* markedEntries;  // Marked entries that will not be pruned
    
    uint32* map;
};

struct BackPtr
{
    uint64 x, y;
};

template<bool PruneTable>
void ProcessTableThread( LPJob* job );

void PruneAndMapThread( LPJob* job );
void ConverToLinePointThread( LPJob* job );
void WriteLookupTableThread( LPJob* job );

// Calculates x * (x-1) / 2. Division is done before multiplication.
inline uint64 GetXEnc( uint64 x );
inline uint64 SquareToLinePoint( uint64 x, uint64 y );
inline uint128 GetXEnc128( uint64 x );

inline BackPtr LinePointToSquare( uint128 index );
inline BackPtr LinePointToSquare64( uint64 index );


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
/// #NOTE: From chiapos:
// Calculates x * (x-1) / 2. Division is done before multiplication.
//-----------------------------------------------------------
FORCE_INLINE uint64 GetXEnc( uint64 x )
{
    ASSERT( x );
    uint64 a = x, b = x - 1;

    // if( a % 2 == 0 )
    if( (a & 1) == 0 )
        a >>= 1; // a /= 2;
    else
        b >>= 1; // b /= 2;

    const uint64 r = a * b;
    ASSERT( r >= a && r >= b );

    return r;
}

//-----------------------------------------------------------
FORCE_INLINE uint128 GetXEnc128( uint64 x )
{
    ASSERT( x );
    uint64 a = x, b = x - 1;

    // if( a % 2 == 0 )
    if( (a & 1) == 0 )
        a >>= 1; // a /= 2;
    else
        b >>= 1; // b /= 2;

    const uint128 r = (uint128)a * b;
    ASSERT( r >= a && r >= b );

    return r;
}

/// #NOTE: From chiapos:
// Encodes two max k bit values into one max 2k bit value. This can be thought of
// mapping points in a two dimensional space into a one dimensional space. The benefits
// of this are that we can store these line points efficiently, by sorting them, and only
// storing the differences between them. Representing numbers as pairs in two
// dimensions limits the compression strategies that can be used.
// The x and y here represent table positions in previous tables.
FORCE_INLINE uint64 SquareToLinePoint( uint64 x, uint64 y )
{
    // Always makes y < x, which maps the random x, y  points from a square into a
    // triangle. This means less data is needed to represent y, since we know it's less
    // than x.
    if( y > x )
        return GetXEnc( y ) + x;

    return GetXEnc( x ) + y;
}

FORCE_INLINE BackPtr LinePointToSquare( uint128 index )
{
    // Performs a square root, without the use of doubles, 
    // to use the precision of the uint128_t.
    uint64 x = 0;
    for( int i = 63; i >= 0; i-- ) 
    {
        uint64 new_x = x + ((uint64)1 << i);
        if( GetXEnc128( new_x ) <= index )
            x = new_x;
    }

    return { x, (uint64)(uint64_t)( index - GetXEnc128( x ) ) };
}

FORCE_INLINE BackPtr LinePointToSquare64( uint64 index )
{
    // Performs a square root, without the use of doubles, 
    // to use the precision of the uint128_t.
    uint64 x = 0;
    for( int i = 63; i >= 0; i-- ) 
    {
        uint64 new_x = x + ((uint64)1 << i);
        if( GetXEnc( new_x ) <= index )
            x = new_x;
    }

    return { x, ( index - GetXEnc( x ) ) };
}

#pragma GCC diagnostic pop


//-----------------------------------------------------------
inline void SyncedJob::Init( const uint threadId, const uint threadCount, std::atomic<uint>& signal, std::atomic<uint>& releaseLock )
{
    _threadId    = threadId;
    _threadCount = threadCount;
    _readyCount  = &signal;
    _releaseLock = &releaseLock;
}

//-----------------------------------------------------------
inline void SyncedJob::WaitForThreads()
{
    std::atomic<uint>& readyCount  = *_readyCount;
    std::atomic<uint>& releaseLock = *_releaseLock;

    const uint targetCount = _threadCount - 1;

    if( _threadId == 0 )
    {
        // Wait for all threads
        while( readyCount.load( std::memory_order_relaxed ) != targetCount );
        
        // Signal all threads
        releaseLock .store( 0, std::memory_order_release );
        readyCount  .store( 0, std::memory_order_release );
    }
    else
    {
        // Signal control thread
        uint count = readyCount.load( std::memory_order_acquire );

        while( !readyCount.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );

        // Wait for control thread to signal us
        while( readyCount.load( std::memory_order_relaxed ) != 0 );

        // Ensure all threads have been released
        count = releaseLock.load( std::memory_order_acquire );
        while( !releaseLock.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );
        while( releaseLock.load( std::memory_order_relaxed ) != targetCount );
    }
}





