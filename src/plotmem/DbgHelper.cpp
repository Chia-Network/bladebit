#include "DbgHelper.h"
#include "b3/blake3.h"

//-----------------------------------------------------------
void DumpTestProofs( const MemPlotContext& cx, const uint64 f7Index )
{
    uint32 proof[64];

    const Pair* tables[6] = {
        cx.t7LRBuffer,
        cx.t6LRBuffer,
        cx.t5LRBuffer,
        cx.t4LRBuffer,
        cx.t3LRBuffer,
        cx.t2LRBuffer
    };

    const uint32* t1xTable = cx.t1XBuffer;

    const uint32 f7     = cx.t7YBuffer [f7Index];
    const Pair&  f7Pair = cx.t7LRBuffer[f7Index];

    Log::Line( "T7 [%-2llu] f7  : %llu : 0x%08lx", f7Index, f7, f7 );
    Log::Line( "T7 [%-2llu] L/R : %-8lu | %-8lu", f7Index, f7Pair.left, f7Pair.right );

    const Pair* rPairs[16]; // R table pairs
    const Pair* lPairs[32]; // L table pairs
    memset( rPairs, 0, sizeof( rPairs ) );
    memset( lPairs, 0, sizeof( lPairs ) );

    rPairs[0] = &f7Pair;

    // Get all pairs up to the 2nd table
    for( uint i = 1; i < 6; i++ )
    {
        const uint32 rCount = 1ul << (i-1);
        const uint32 lCount = 1ul << i;

        const Pair* table = tables[i];
        Log::Line( "Table %d", 7-i );

        for( uint r = 0, l = 0; r < rCount; r++ )
        {
            const Pair* rPair = rPairs[r];
            
            Log::Line( "T%d [%-2lu] L/R: %-10lu | %-10lu", 7-i, r, 
                        rPair->left, rPair->right );

            lPairs[l++] = &table[rPair->left ];
            lPairs[l++] = &table[rPair->right];
        }

        // Copy pairs to rTable
        memcpy( rPairs, lPairs, sizeof( Pair* ) * lCount );
    }

    // Grab all x values pointed by the pairs
    for( uint i = 0, p = 0; i < 32; i++ )
    {
        const Pair* pair = lPairs[i];

        proof[p++] = t1xTable[pair->left ];
        proof[p++] = t1xTable[pair->right];
    }

    Log::Line( "Proof x's:" );
    for( uint i = 0; i < 64; i++ )
    {
        const uint32 x = proof[i];
        
        Log::Line( "[%-2d] : %-10lu : 0x%08lx", i, x, x );
    }
}

//-----------------------------------------------------------
void WritePhaseTableFiles( MemPlotContext& cx )
{
    DbgWriteTableToFile( *cx.threadPool, DBG_P1_TABLE1_FNAME  , cx.entryCount[0], cx.t1XBuffer  );
    DbgWriteTableToFile( *cx.threadPool, DBG_P1_TABLE2_FNAME  , cx.entryCount[1], cx.t2LRBuffer );
    DbgWriteTableToFile( *cx.threadPool, DBG_P1_TABLE3_FNAME  , cx.entryCount[2], cx.t3LRBuffer );
    DbgWriteTableToFile( *cx.threadPool, DBG_P1_TABLE4_FNAME  , cx.entryCount[3], cx.t4LRBuffer );
    DbgWriteTableToFile( *cx.threadPool, DBG_P1_TABLE5_FNAME  , cx.entryCount[4], cx.t5LRBuffer );
    DbgWriteTableToFile( *cx.threadPool, DBG_P1_TABLE6_FNAME  , cx.entryCount[5], cx.t6LRBuffer );
    DbgWriteTableToFile( *cx.threadPool, DBG_P1_TABLE7_FNAME  , cx.entryCount[6], cx.t7LRBuffer );
    DbgWriteTableToFile( *cx.threadPool, DBG_P1_TABLE7_Y_FNAME, cx.entryCount[6], cx.t7YBuffer  );
}

//-----------------------------------------------------------
FILE* CreateHashFile( const char* fileName )
{
    char filePath[512];
    snprintf( filePath, sizeof( filePath ), DBG_TABLES_PATH "hash_%s.txt", fileName );

    FILE* file = fopen( filePath, "wb" );
    return file;
}

//-----------------------------------------------------------
// Hash some bytes with B3 and print a 256 bit hash of the output to a file
void PrintHash( FILE* file, uint64 index, const void* input, size_t inputSize )
{
    blake3_hasher hasher;

    byte hash[32];
    blake3_hasher_init( &hasher );
    blake3_hasher_update( &hasher, input, inputSize );
    blake3_hasher_finalize( &hasher, (uint8_t*)hash, sizeof( hash ) );
    
    fprintf( file, "[%-12llu] 0x", (llu)index );
    for( uint64 i = 0; i < sizeof( hash ); i+=16 )
    {
        fprintf( file, 
        "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x", 
        hash[i],   hash[i+1],  hash[i+2],  hash[i+3],
        hash[i+4], hash[i+5],  hash[i+6],  hash[i+7],
        hash[i+8], hash[i+9],  hash[i+10], hash[i+11], 
        hash[i+12],hash[i+13], hash[i+14], hash[i+15]);
    } 
    fputc( '\n', file );
    // fflush( file );
}
