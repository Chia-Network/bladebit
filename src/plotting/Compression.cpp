#include "Compression.h"
#include "plotting/FSETableGenerator.h"
#include "util/Util.h"
#include <mutex>

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
    info.subtSizeBits  = CompressionLevelInfo<level>::STUB_BIT_SIZE;
    info.tableParkSize = CompressionLevelInfo<level>::TABLE_PARK_SIZE;
    info.ansRValue     = CompressionLevelInfo<level>::ANS_R_VALUE;
}

CompressionInfo GetCompressionInfoForLevel( const uint32 compressionLevel )
{
    CompressionInfo info = {};

    switch ( compressionLevel )
    {
        case 1: GetCompressionInfoForLevel<1>( info ); break;
        case 2: GetCompressionInfoForLevel<2>( info ); break;
        case 3: GetCompressionInfoForLevel<3>( info ); break;
        case 4: GetCompressionInfoForLevel<4>( info ); break;
        case 5: GetCompressionInfoForLevel<5>( info ); break;
        case 6: GetCompressionInfoForLevel<6>( info ); break;
        case 7: GetCompressionInfoForLevel<7>( info ); break;
        case 8: GetCompressionInfoForLevel<8>( info ); break;
        case 9: GetCompressionInfoForLevel<9>( info ); break;
    
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