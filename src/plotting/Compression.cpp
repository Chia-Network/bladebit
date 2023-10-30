#include "Compression.h"
#include "plotting/FSETableGenerator.h"
#include "util/Util.h"
#include <mutex>
#include <algorithm>

// Caches for C and D tables
static std::atomic<FSE_CTable*> _cTableCache[32] = {};
static std::atomic<FSE_DTable*> _dTableCache[32] = {};

static std::mutex _cCacheLock;
static std::mutex _dCacheLock;

void* CreateCompressionCTableForCLevel( size_t* outTableSize, const uint32 compressionLevel, const double rValue, const bool compress )
{
    if( compress )
    {
        FSE_CTable* cTable = _cTableCache[compressionLevel].load( std::memory_order_acquire );
        if( !cTable )
        {
            _cCacheLock.lock();
            cTable = _cTableCache[compressionLevel].load( std::memory_order_acquire );
            if( !cTable )
            {
                // Cache it
                cTable = FSETableGenerator::GenCompressionTable( rValue, outTableSize );
                _cTableCache[compressionLevel].store( cTable, std::memory_order_release );
            }
            _cCacheLock.unlock();
        }

        return cTable;
    }
    else
    {
        FSE_DTable* dTable = _dTableCache[compressionLevel].load( std::memory_order_acquire );
        if( !dTable )
        {
            _dCacheLock.lock();
            dTable = _dTableCache[compressionLevel].load( std::memory_order_acquire );
            if( !dTable )
            {
                // Cache it
                dTable = FSETableGenerator::GenDecompressionTable( rValue, outTableSize );
                _dTableCache[compressionLevel].store( dTable, std::memory_order_release );
            }
            _dCacheLock.unlock();
        }

        return dTable;
    }

    return nullptr;
}

template<uint32 level>
void* CreateCompressionCTable( size_t* outTableSize, const bool compress )
{
    return CreateCompressionCTableForCLevel( outTableSize, level, CompressionLevelInfo<level>::ANS_R_VALUE, compress );
}

template<uint32 level>
void GetCompressionInfoForLevel( CompressionInfo& info )
{
    info.entrySizeBits = CompressionLevelInfo<level>::ENTRY_SIZE;
    info.stubSizeBits  = CompressionLevelInfo<level>::STUB_BIT_SIZE;
    info.tableParkSize = CompressionLevelInfo<level>::TABLE_PARK_SIZE;
    info.ansRValue     = CompressionLevelInfo<level>::ANS_R_VALUE;
}

CompressionInfo GetCompressionInfoForLevel( const uint32 compressionLevel )
{
    CompressionInfo info = {};

    switch ( compressionLevel )
    {
        case 1: return CreateCompressionCTable<1>( outTableSize, compress );
        case 2: return CreateCompressionCTable<2>( outTableSize, compress );
        case 3: return CreateCompressionCTable<3>( outTableSize, compress );
        case 4: return CreateCompressionCTable<4>( outTableSize, compress );
        case 5: return CreateCompressionCTable<5>( outTableSize, compress );
        case 6: return CreateCompressionCTable<6>( outTableSize, compress );
        case 7: return CreateCompressionCTable<7>( outTableSize, compress );
        case 8: return CreateCompressionCTable<8>( outTableSize, compress );
        case 9: return CreateCompressionCTable<9>( outTableSize, compress );
        case 10: return CreateCompressionCTable<10>( outTableSize, compress );
        case 11: return CreateCompressionCTable<11>( outTableSize, compress );
        case 12: return CreateCompressionCTable<12>( outTableSize, compress );
        case 13: return CreateCompressionCTable<13>( outTableSize, compress );
        case 14: return CreateCompressionCTable<14>( outTableSize, compress );
        case 15: return CreateCompressionCTable<15>( outTableSize, compress );
        case 16: return CreateCompressionCTable<16>( outTableSize, compress );
        case 17: return CreateCompressionCTable<17>( outTableSize, compress );
        case 18: return CreateCompressionCTable<18>( outTableSize, compress );
        case 19: return CreateCompressionCTable<19>( outTableSize, compress );
        case 20: return CreateCompressionCTable<20>( outTableSize, compress );
        case 21: return CreateCompressionCTable<21>( outTableSize, compress );
        case 22: return CreateCompressionCTable<22>( outTableSize, compress );
        case 23: return CreateCompressionCTable<23>( outTableSize, compress );
        case 24: return CreateCompressionCTable<24>( outTableSize, compress );
        case 25: return CreateCompressionCTable<25>( outTableSize, compress );
        case 26: return CreateCompressionCTable<26>( outTableSize, compress );
        case 27: return CreateCompressionCTable<27>( outTableSize, compress );
        case 28: return CreateCompressionCTable<28>( outTableSize, compress );
        case 29: return CreateCompressionCTable<29>( outTableSize, compress );
        case 30: return CreateCompressionCTable<30>( outTableSize, compress );
        case 31: return CreateCompressionCTable<31>( outTableSize, compress );
        case 32: return CreateCompressionCTable<32>( outTableSize, compress );
        case 33: return CreateCompressionCTable<33>( outTableSize, compress );
        case 34: return CreateCompressionCTable<34>( outTableSize, compress );
        case 35: return CreateCompressionCTable<35>( outTableSize, compress );
        case 36: return CreateCompressionCTable<36>( outTableSize, compress );
        case 37: return CreateCompressionCTable<37>( outTableSize, compress );
        case 38: return CreateCompressionCTable<38>( outTableSize, compress );
        case 39: return CreateCompressionCTable<39>( outTableSize, compress );       
    default:
        Fatal( "Invalid compression level %u.", compressionLevel );
        break;
    }

    return info;
}

void* CreateCompressionTable( const uint32 compressionLevel, size_t* outTableSize, const bool compress )
{
    switch ( compressionLevel )
    {

        case 1: return CreateCompressionCTable<1>( outTableSize, compress );
        case 2: return CreateCompressionCTable<2>( outTableSize, compress );
        case 3: return CreateCompressionCTable<3>( outTableSize, compress );
        case 4: return CreateCompressionCTable<4>( outTableSize, compress );
        case 5: return CreateCompressionCTable<5>( outTableSize, compress );
        case 6: return CreateCompressionCTable<6>( outTableSize, compress );
        case 7: return CreateCompressionCTable<7>( outTableSize, compress );
        case 8: return CreateCompressionCTable<8>( outTableSize, compress );
        case 9: return CreateCompressionCTable<9>( outTableSize, compress );
        case 10: return CreateCompressionCTable<10>( outTableSize, compress );
        case 11: return CreateCompressionCTable<11>( outTableSize, compress );
        case 12: return CreateCompressionCTable<12>( outTableSize, compress );
        case 13: return CreateCompressionCTable<13>( outTableSize, compress );
        case 14: return CreateCompressionCTable<14>( outTableSize, compress );
        case 15: return CreateCompressionCTable<15>( outTableSize, compress );
        case 16: return CreateCompressionCTable<16>( outTableSize, compress );
        case 17: return CreateCompressionCTable<17>( outTableSize, compress );
        case 18: return CreateCompressionCTable<18>( outTableSize, compress );
        case 19: return CreateCompressionCTable<19>( outTableSize, compress );
        case 20: return CreateCompressionCTable<20>( outTableSize, compress );
        case 21: return CreateCompressionCTable<21>( outTableSize, compress );
        case 22: return CreateCompressionCTable<22>( outTableSize, compress );
        case 23: return CreateCompressionCTable<23>( outTableSize, compress );
        case 24: return CreateCompressionCTable<24>( outTableSize, compress );
        case 25: return CreateCompressionCTable<25>( outTableSize, compress );
        case 26: return CreateCompressionCTable<26>( outTableSize, compress );
        case 27: return CreateCompressionCTable<27>( outTableSize, compress );
        case 28: return CreateCompressionCTable<28>( outTableSize, compress );
        case 29: return CreateCompressionCTable<29>( outTableSize, compress );
        case 30: return CreateCompressionCTable<30>( outTableSize, compress );
        case 31: return CreateCompressionCTable<31>( outTableSize, compress );
        case 32: return CreateCompressionCTable<32>( outTableSize, compress );
        case 33: return CreateCompressionCTable<33>( outTableSize, compress );
        case 34: return CreateCompressionCTable<34>( outTableSize, compress );
        case 35: return CreateCompressionCTable<35>( outTableSize, compress );
        case 36: return CreateCompressionCTable<36>( outTableSize, compress );
        case 37: return CreateCompressionCTable<37>( outTableSize, compress );
        case 38: return CreateCompressionCTable<38>( outTableSize, compress );
        case 39: return CreateCompressionCTable<39>( outTableSize, compress );
    
        default:
        break;
    }

    Fatal( "Invalid compression level %u.", compressionLevel );
    return nullptr;
}

FSE_CTable* CreateCompressionCTable( const uint32 compressionLevel, size_t* outTableSize )
{
    return (FSE_CTable*)CreateCompressionTable( compressionLevel, outTableSize, true );
}

FSE_DTable* CreateCompressionDTable( const uint32 compressionLevel, size_t* outTableSize )
{
    return (FSE_DTable*)CreateCompressionTable( compressionLevel, outTableSize, false );
}

uint32 GetCompressedLPBitCount( const uint32 compressionLevel )
{
    // #TODO: Don't support this? Or rather, support K size
    if( compressionLevel == 0 )
        return 64;

    const uint32 nDroppedTables = compressionLevel < 9 ? 1 :
                                   compressionLevel < 13 ? 2 : 3;

    auto info = GetCompressionInfoForLevel( compressionLevel );

    uint32 lpBitSize = info.entrySizeBits * 2 * nDroppedTables - 1;

    // for( uint32 i = 0; i < nDroppedTables; i++ )
    //     lpBitSize = lpBitSize * 2 - 1;

    return lpBitSize * 2 - 1;
}

size_t GetLargestCompressedParkSize()
{
    return std::max( {
        GetCompressionInfoForLevel( 1 ).tableParkSize,
        GetCompressionInfoForLevel( 2 ).tableParkSize,
        GetCompressionInfoForLevel( 3 ).tableParkSize,
        GetCompressionInfoForLevel( 4 ).tableParkSize,
        GetCompressionInfoForLevel( 5 ).tableParkSize,
        GetCompressionInfoForLevel( 6 ).tableParkSize,
        GetCompressionInfoForLevel( 7 ).tableParkSize,
        GetCompressionInfoForLevel( 8 ).tableParkSize,
        GetCompressionInfoForLevel( 9 ).tableParkSize }
    );
}