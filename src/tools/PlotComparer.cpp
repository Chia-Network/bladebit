#include "io/FileStream.h"
#include "ChiaConsts.h"
#include "Util.h"
#include "util/Log.h"
#include "plotshared/PlotTools.h"
#include <vector>

inline bool MemCmp( const void* a, const void* b, size_t size )
{
    return memcmp( a, b, size ) == 0;
}

class PlotInfo;

void TestTable( TableId table, PlotInfo& ref, PlotInfo& tgt );
void TestC3Table( PlotInfo& ref, PlotInfo& tgt );
void TestTable( PlotInfo& ref, PlotInfo& tgt, TableId table );

Span<uint> ReadC1Table( PlotInfo& plot );

class PlotInfo
{
public:
    PlotInfo() {}

    ~PlotInfo()
    {

    }

    void Open( const char* path )
    {
        _path = path;
        FatalIf( IsOpen(), "Plot is already open." );

        // FileFlags::NoBuffering | FileFlags::NoBuffering ),   // #TODO: allow unbuffered reading with our own buffer... For now just use like this
        FatalIf( !_plotFile.Open( path, FileMode::Open, FileAccess::Read, FileFlags::None ), 
            "Failed to open plot '%s' with error %d.", path, _plotFile.GetError() );

        const size_t blockSize = _plotFile.BlockSize();
        _blockBuffer = bbvirtalloc<byte>( blockSize );

        ///
        /// Read header
        ///

        // Magic
        {
            char magic[sizeof( kPOSMagic )-1] = { 0 };
            Read( sizeof( magic ), magic );
            FatalIf( !MemCmp( magic, kPOSMagic, sizeof( magic ) ), "Invalid plot magic." );
        }
        
        // Plot Id
        {
            Read( sizeof( _id ), _id );

            char str[65] = { 0 };
            size_t numEncoded = 0;
            BytesToHexStr( _id, sizeof( _id ), str, sizeof( str ), numEncoded );
            ASSERT( numEncoded == sizeof( _id ) );
            _idString = str;
        }

        // K
        {
            byte k = 0;
            Read( 1, &k );
            _k = k;
        }

        // Format Descritption
        {
            const uint formatDescSize =  ReadUInt16();
            FatalIf( formatDescSize != sizeof( kFormatDescription ) - 1, "Invalid format description size." );

            char desc[sizeof( kFormatDescription )-1] = { 0 };
            Read( sizeof( desc ), desc );
            FatalIf( !MemCmp( desc, kFormatDescription, sizeof( desc ) ), "Invalid format description." );
        }
        
        // Memo
        {
            uint memoSize = ReadUInt16();
            FatalIf( memoSize > sizeof( _memo ), "Invalid memo." );
            _memoLength = memoSize;

            Read( memoSize, _memo );

            char str[BB_PLOT_MEMO_MAX_SIZE*2+1] = { 0 };
            size_t numEncoded = 0;
            BytesToHexStr( _memo, memoSize, str, sizeof( str ), numEncoded );
            
            _memoString = str;
        }

        // Table pointers
        Read( sizeof( _tablePtrs ), _tablePtrs );
        for( int i = 0; i < 10; i++ )
            _tablePtrs[i] = Swap64( _tablePtrs[i] );

        // What follows is table data
    }

public:
    const bool IsOpen() const { return _plotFile.IsOpen(); }

    const byte* PlotId() const { return _id; }

    const std::string& PlotIdStr() const { return _idString; }

    uint PlotMemoSize() const { return _memoLength; }

    const byte* PlotMemo() const { return _memo; }

    const std::string& PlotMemoStr() const { return _memoString; }

    uint K() const { return _k; }

    uint64 TableAddress( TableId table ) const
    {
        ASSERT( table >= TableId::Table1 && table <= TableId::Table7 );
        return _tablePtrs[(int)table];
    }

    uint64 CTableAddress( int c )
    {
        ASSERT( c >= 1 && c <= 3 );

        return _tablePtrs[c+6];
    }

    size_t TableSize( int tableIndex )
    {
        ASSERT( tableIndex >= 0 && tableIndex < 10 );

        const uint64 address = _tablePtrs[tableIndex];
        uint64 endAddress = _plotFile.Size();

        // Check all table entris where we find and address that is 
        // greater than ours and less than the current end address
        for( int i = 0; i < 10; i++ )
        {
            const uint64 a = _tablePtrs[i];
            if( a > address && a < endAddress )
                endAddress = a;
        }

        return (size_t)( endAddress - address );
    }

    ssize_t Read( size_t size, void* buffer )
    {
        ASSERT( buffer );
        if( size == 0 )
            return 0;

        const size_t blockSize  = _plotFile.BlockSize();

        // Read-in any data already left-over in the block buffer
        if( _blockRemainder )
        {
            const size_t copySize = std::min( _blockRemainder, size );
            memcpy( buffer, _blockBuffer + _blockOffset, copySize );

            _blockOffset    += copySize;
            _blockRemainder -= copySize;

            buffer = (void*)((byte*)buffer + copySize);
            size -= copySize;

            if( size == 0 )
                return copySize;
        }

        const size_t blockCount = size / blockSize;

        size_t blockSizeToRead = blockCount * blockSize;
        const size_t remainder  = size - blockSizeToRead;

        byte* reader = (byte*)buffer;
        ssize_t sizeRead = 0;

        while( blockSizeToRead )
        {
            ssize_t read = _plotFile.Read( buffer, blockSizeToRead );
            FatalIf( read < 0 , "Plot %s failed to read with error: %d.", _path.c_str(), _plotFile.GetError() );
            
            reader   += read;
            sizeRead += read;
            blockSizeToRead -= (size_t)read;
        }

        if( remainder )
        {
            ssize_t read = _plotFile.Read( _blockBuffer, blockSize );
            ASSERT( read == (ssize_t)remainder || read == (ssize_t)blockSize );       

            FatalIf( read < (ssize_t)remainder, "Failed to read a full block on plot %s.", _path.c_str() );

            memcpy( reader, _blockBuffer, remainder );
            sizeRead += read;

            // Save any left over data in the block buffer
            _blockOffset    = remainder;
            _blockRemainder = blockSize - remainder;
        }

        return sizeRead;
    }

    uint16 ReadUInt16()
    {
        uint16 value = 0;
        Read( sizeof( value ), &value );
        return Swap16( value );
    }

    void ReadTable( int tableIndex, void* buffer )
    {
        const size_t size = TableSize( tableIndex );
        
        _blockRemainder = 0;
        FatalIf( !_plotFile.Seek( (int64)_tablePtrs[tableIndex], SeekOrigin::Begin ),
            "Failed to seek to table %u.", tableIndex+1 );

        Read( size, buffer );
    }

    void DumpHeader()
    {
        Log::Line( "Plot %s", _path.c_str() );
        Log::Line( "-----------------------------------------" );
        Log::Line( "Id       : %s", _idString.c_str() );
        Log::Line( "Memo     : %s", _memoString.c_str() );
        Log::Line( "K        : %u", _k );

        for( int i = 0; i <= (int)TableId::Table7; i++ )
        {
            const size_t size = TableSize( i );

            Log::Line( "Table %u  : %16lu ( 0x%016lx ) : %8llu MiB ( %.2lf GiB )", 
                i+1, _tablePtrs[i], _tablePtrs[i],
                size BtoMB, (double)size BtoGB );

        }

        for( int i = (int)TableId::Table7+1; i < 10; i++ )
        {
            const size_t size = TableSize( i );

            Log::Line( "C%u       : %16lu ( 0x%016lx ) : %8llu MiB ( %.2lf GiB )",
                i-6, _tablePtrs[i], _tablePtrs[i],
                size BtoMB, (double)size BtoGB );
        }

    }

private:
    FileStream  _plotFile;
    byte        _id[BB_PLOT_ID_LEN]          = { 0 };
    byte        _memo[BB_PLOT_MEMO_MAX_SIZE] = { 0 };
    uint        _memoLength     = 0;
    std::string _idString       = "";
    std::string _memoString     = "";
    uint        _k              = 0;
    std::string _path           = "";
    uint64      _tablePtrs[10]  = { 0 };
    byte*       _blockBuffer    = nullptr;
    size_t      _blockRemainder = 0;
    size_t      _blockOffset    = 0;

    // size_t      _readBufferSize = 32 MB;
    // byte*       _readBuffer     = nullptr;

};

/// Compares 2 plot's tables
//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    PlotInfo refPlot; // Reference
    PlotInfo tgtPlot; // Target

    {
        const char* refPath = "/mnt/p5510a/plots-ref/plot-k32-2022-02-05-17-07-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
        const char* tgtPath = "/mnt/p5510a/disk_tmp/plot-k32-2022-02-05-21-16-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot.tmp";

        refPlot.Open( refPath );
        tgtPlot.Open( tgtPath );

        refPlot.DumpHeader();
        Log::Line( "" );
        tgtPlot.DumpHeader();
    }

    FatalIf( !MemCmp( refPlot.PlotId(), tgtPlot.PlotId(), BB_PLOT_ID_LEN ), "Plot id mismatch." );
    FatalIf( !MemCmp( refPlot.PlotMemo(), tgtPlot.PlotMemo(), std::min( refPlot.PlotMemoSize(), tgtPlot.PlotMemoSize() ) ), "Plot memo mismatch." );
    FatalIf( refPlot.K() != tgtPlot.K(), "K value mismatch." );

    for( TableId table = TableId::Table1; table <= TableId::Table7; table++ )
        TestTable( refPlot, tgtPlot, table );

    TestC3Table( refPlot, tgtPlot );


    return 0;
}

//-----------------------------------------------------------
Span<uint> ReadC1Table( PlotInfo& plot )
{
    const size_t tableSize  = plot.TableSize( 7 );
    const uint32 entryCount = (uint)( tableSize / sizeof( uint32 ) );
    
    uint32* c1 = bbvirtalloc<uint32>( tableSize );
    plot.ReadTable( 7, c1 );

    for( uint i = 0; i < entryCount; i++ )
        c1[i] = Swap32( c1[i] );

    return Span<uint>( c1, entryCount );
}

//-----------------------------------------------------------
Span<uint> ReadC2Table( PlotInfo& plot )
{
    const size_t tableSize  = plot.TableSize( 8 );
    const uint32 entryCount = (uint)( tableSize / sizeof( uint32 ) );
    
    uint32* c2 = bbvirtalloc<uint32>( tableSize );
    plot.ReadTable( 8, c2 );

    for( uint i = 0; i < entryCount; i++ )
        c2[i] = Swap32( c2[i] );

    return Span<uint>( c2, entryCount );
}

//-----------------------------------------------------------
void TestC3Table( PlotInfo& ref, PlotInfo& tgt )
{
    Log::Line( "Reading C tables..." );
    // const size_t refSize = ref.TableSize( 9 );
    // const size_t tgtSize = tgt.TableSize( 9 );

    // const size_t c3Size = std::min( refSize, tgtSize );

    // Read C1 so that we know how many parks we got
    Span<uint> refC1 = ReadC1Table( ref );
    Span<uint> tgtC1 = ReadC1Table( tgt );

    // Compare C1
    const uint c1Length = std::min( refC1.length, tgtC1.length );
    {
        Log::Line( "Validating C1 table... " );
        for( uint i = 0; i < c1Length; i++ )
        {
            FatalIf( refC1[i] != tgtC1[i], "C1 table mismatch @ index %u. Ref: %u Tgt: %u", i, refC1[i], tgtC1[i] );
        }
        Log::Line( "Success!" );
    }

    // Check C2
    Span<uint> refC2 = ReadC2Table( ref );
    Span<uint> tgtC2 = ReadC2Table( tgt );

    const uint c2Length = std::min( refC2.length, tgtC2.length );
    {
        Log::Line( "Validating C2 table... " );
        for( uint i = 0; i < c2Length; i++ )
        {
            FatalIf( refC2[i] != tgtC2[i], "C2 table mismatch @ index %u. Ref: %u Tgt: %u", i, refC1[i], tgtC1[i] );
        }
        Log::Line( "Success!" );
    }

    const uint64 f7Count = c1Length * (uint64)kCheckpoint1Interval;
    Log::Line( "F7 Count: ~%llu", f7Count );

    Log::Line( "Validating C3 table..." );
    {
        const size_t refC3Size = ref.TableSize( 9 );
        const size_t tgtC3Size = tgt.TableSize( 9 );

        byte* refC3 = bbvirtalloc<byte>( refC3Size );
        byte* tgtC3 = bbvirtalloc<byte>( tgtC3Size );

        ref.ReadTable( 9, refC3 );
        tgt.ReadTable( 9, tgtC3 );

        // const size_t c3Size = std::min( refC3Size, tgtC3Size );

        const int64  parkCount  = (int64)(c1Length - 1);
        const size_t c3ParkSize = CalculateC3Size();

        const byte* refC3Reader = refC3;
        const byte* tgtC3Reader = tgtC3;

        std::vector<int64> failures;

        for( int64 i = 0; i < parkCount; i++ )
        {
            if( !MemCmp( refC3Reader, tgtC3Reader, c3ParkSize ) )
            {
                Log::Line( " C3 park %lld failed.", i );
                failures.push_back( i );
            }
            // FatalIf( !MemCmp( refC3Reader, tgtC3Reader, c3ParkSize ), 
            //     "C3 park mismatch @ park %lld / %lld", i, parkCount );

            refC3Reader += c3ParkSize;
            tgtC3Reader += c3ParkSize;
        }

        if( failures.size() < 1 )
            Log::Line( "Success!" );
        else
            Log::Line( "%llu / %lld C3 parks failed!", failures.size(), parkCount );
        
        SysHost::VirtualFree( refC3 );
        SysHost::VirtualFree( tgtC3 );
    }

    SysHost::VirtualFree( refC2.values );
    SysHost::VirtualFree( tgtC2.values );
    SysHost::VirtualFree( refC1.values );
    SysHost::VirtualFree( tgtC1.values );
}

//-----------------------------------------------------------
void TestTable( PlotInfo& ref, PlotInfo& tgt, TableId table )
{
    Log::Line( "Reading Table %u...", table+1 );

    const size_t parkSize = table < TableId::Table7 ? CalculateParkSize( table ) : CalculatePark7Size( ref.K() );

    const size_t sizeRef = ref.TableSize( (int)table );
    const size_t sizeTgt = tgt.TableSize( (int)table );

    byte* p7Ref = bbvirtalloc<byte>( sizeRef );
    byte* p7Tgt = bbvirtalloc<byte>( sizeTgt );

    ref.ReadTable( (int)table, p7Ref );
    tgt.ReadTable( (int)table, p7Tgt );

    const size_t tableSize = std::min( sizeRef, sizeTgt );
    const int64  parkCount = (int64)( tableSize / parkSize );

    const byte* parkRef = p7Ref;
    const byte* parkTgt = p7Tgt;
    
    Log::Line( "Validating Table %u...", table+1 );
    std::vector<int64> failures;

    for( int64 i = 0; i < parkCount; i++ )
    {
        if( !MemCmp( parkRef, parkTgt, parkSize ) )
        {
            Log::Line( " T%u park %lld failed.", table+1, i );
            failures.push_back( i );
        }

        parkRef += parkSize;
        parkTgt += parkSize;
    }

    if( failures.size() < 1 )
        Log::Line( "Success!" );
    else
        Log::Line( "%llu / %lld Table %u parks failed.", failures.size(), parkCount, table+1 );

    SysHost::VirtualFree( p7Ref );
    SysHost::VirtualFree( p7Tgt );
}


