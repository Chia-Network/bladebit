#include "io/FileStream.h"
#include "ChiaConsts.h"
#include "util/Util.h"
#include "util/Log.h"
#include "util/CliParser.h"
#include "util/BitView.h"
#include "plotting/PlotTools.h"
#include "plotting/GlobalPlotConfig.h"
#include "plotdisk/jobs/IOJob.h"
#include "threading/ThreadPool.h"
#include <vector>
#include <algorithm>

class PlotInfo;

void TestTable( TableId table, PlotInfo& ref, PlotInfo& tgt );
void TestC3Table( PlotInfo& ref, PlotInfo& tgt );
void TestTable( PlotInfo& ref, PlotInfo& tgt, TableId table );

void UnpackPark7( const byte* srcBits, uint64* dstEntries );

void DumpP7( PlotInfo& plot, const char* path );

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

    FileStream& PlotFile() { return _plotFile; }

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
        // if( _blockRemainder )
        // {
        //     const size_t copySize = std::min( _blockRemainder, size );
        //     memcpy( buffer, _blockBuffer + _blockOffset, copySize );

        //     _blockOffset    += copySize;
        //     _blockRemainder -= copySize;

        //     buffer = (void*)((byte*)buffer + copySize);
        //     size -= copySize;

        //     if( size == 0 )
        //         return copySize;
        // }

        const size_t blockCount = size / blockSize;

        size_t blockSizeToRead = blockCount * blockSize;
        const size_t remainder  = size - blockSizeToRead;

        byte* reader = (byte*)buffer;
        ssize_t sizeRead = 0;

        while( blockSizeToRead )
        {
            ssize_t read = _plotFile.Read( reader, blockSizeToRead );
            FatalIf( read < 0 , "Plot %s failed to read with error: %d.", _path.c_str(), _plotFile.GetError() );
            
            reader   += read;
            sizeRead += read;
            blockSizeToRead -= (size_t)read;
        }

        if( remainder )
        {
            ssize_t read = _plotFile.Read( reader, remainder );
            
            // ssize_t read = _plotFile.Read( _blockBuffer, blockSize );
            ASSERT( read == (ssize_t)remainder || read == (ssize_t)blockSize );       

            // FatalIf( read < (ssize_t)remainder, "Failed to read a full block on plot %s.", _path.c_str() );

            // memcpy( reader, _blockBuffer, remainder );
            sizeRead += read;

            // // Save any left over data in the block buffer
            // _blockOffset    = remainder;
            // _blockRemainder = blockSize - remainder;
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



//-----------------------------------------------------------
const char USAGE[] = R"(plotcmp <plot_a_path> <plot_b_path>

Compares 2 plots for matching tables.

[ARGUMENTS]
<plot_*_path> : Path to the plot files to be compared.
 -h, --help   : Print this help message and exit.
)";

//-----------------------------------------------------------
void PlotCompareMainPrintUsage()
{
    Log::Line( "" );
    Log::Flush();
}


/// Compares 2 plot's tables
//-----------------------------------------------------------
struct PlotCompareOptions
{
    const char* plotAPath = "";
    const char* plotBPath = "";
};

//-----------------------------------------------------------
void PlotCompareMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    PlotCompareOptions opts;

    while( cli.HasArgs() )
    {
        if( cli.ArgConsume( "-h", "--help" ) )
        {
            PlotCompareMainPrintUsage();
            exit( 0 );
        }
        else
            break;
    }

    opts.plotAPath = cli.ArgConsume();
    opts.plotBPath = cli.ArgConsume();

    PlotInfo refPlot; // Reference
    PlotInfo tgtPlot; // Target

    {
        const char* refPath = opts.plotAPath;
        const char* tgtPath = opts.plotBPath;

        refPlot.Open( refPath );
        tgtPlot.Open( tgtPath );

        refPlot.DumpHeader();
        Log::Line( "" );
        tgtPlot.DumpHeader();
    }

    FatalIf( refPlot.K() != 32, "Plot A is k%u. Only k32 plots are currently supported.", refPlot.K() );
    FatalIf( tgtPlot.K() != 32, "Plot B is k%u. Only k32 plots are currently supported.", tgtPlot.K() );

    FatalIf( !MemCmp( refPlot.PlotId(), tgtPlot.PlotId(), BB_PLOT_ID_LEN ), "Plot id mismatch." );
    FatalIf( !MemCmp( refPlot.PlotMemo(), tgtPlot.PlotMemo(), std::min( refPlot.PlotMemoSize(), tgtPlot.PlotMemoSize() ) ), "Plot memo mismatch." );
    FatalIf( refPlot.K() != tgtPlot.K(), "K value mismatch." );

    // Test P7, dump it
    // DumpP7( refPlot, "/mnt/p5510a/reference/p7.tmp" );

    // TestTable( refPlot, tgtPlot, TableId::Table7 );
    // TestTable( refPlot, tgtPlot, TableId::Table3 );

    TestC3Table( refPlot, tgtPlot );

    for( TableId table = TableId::Table1; table <= TableId::Table7; table++ )
        TestTable( refPlot, tgtPlot, table );

    // TestC3Table( refPlot, tgtPlot );
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
    const uint64 c1Length = (uint64)std::min( refC1.length, tgtC1.length );
    {
        Log::Line( "Validating C1 table... " );
        uint64 failCount = 0;
        for( uint i = 0; i < c1Length; i++ )
        {
            if( refC1[i] != tgtC1[i] )
            {
                if( failCount == 0 )
                    Log::Line( "C1 table mismatch @ index %u. Ref: %u Tgt: %u", i, refC1[i], tgtC1[i] );

                failCount++;
            }
        }

        if( failCount == 0 )
            Log::Line( "Success!" );
        else
            Log::Line( "C1 table mismatch: %llu entries failed.", failCount );
    }

    // Check C2
    Span<uint> refC2 = ReadC2Table( ref );
    Span<uint> tgtC2 = ReadC2Table( tgt );

    const uint64 c2Length = (uint64)std::min( refC2.length, tgtC2.length );
    {
        Log::Line( "Validating C2 table... " );

        uint64 failCount = 0;
        for( uint i = 0; i < c2Length; i++ )
        {
            if( refC2[i] != tgtC2[i] )
            {
                if( failCount == 0 )
                    Log::Line( "C2 table mismatch @ index %u. Ref: %u Tgt: %u", i, refC2[i], tgtC2[i] );

                failCount++;
            }
        }

        if( failCount == 0 )
            Log::Line( "Success!" );
        else
            Log::Line( "C2 table mismatch: %llu entries failed.", failCount );
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
uint64 CompareP7( PlotInfo& ref, PlotInfo& tgt, const byte* p7RefBytes, const byte* p7TgtBytes, const int64 parkCount )
{
    // Double-buffer parks at a time so that we can compare entries across parks
    uint64 refParks[2][kEntriesPerPark];
    uint64 tgtParks[2][kEntriesPerPark];

    uint64 refMatch[kEntriesPerPark];
    uint64 tgtMatch[kEntriesPerPark];

    // Unpack first park
    ASSERT( parkCount );

    const size_t parkSize = CalculatePark7Size( ref.K() );

    UnpackPark7( p7RefBytes, refParks[0] );
    UnpackPark7( p7TgtBytes, tgtParks[0] );
    p7RefBytes += parkSize;
    p7TgtBytes += parkSize;

    uint64 parkFailCount = 0;

    for( int64 p = 0; p < parkCount; p++ )
    {
        const bool isLastPark = p + 1 == parkCount;

        // Load the next park, if we can
        if( !isLastPark )
        {
            UnpackPark7( p7RefBytes, refParks[1] );
            UnpackPark7( p7TgtBytes, tgtParks[1] );
            p7RefBytes += parkSize;
            p7TgtBytes += parkSize;
        }

        const uint64* ref = refParks[0];
        const uint64* tgt = tgtParks[0];

        const int64 entriesEnd = isLastPark ? kEntriesPerPark : kEntriesPerPark * 2;

        for( int64 i = 0; i < kEntriesPerPark; i++ )
        {
            if( ref[i] == tgt[i] )
                continue;

            // Potential out-of-order entries
            // Try sorting and matching entry pairs until we run out of entries
            int64 j = i + 1;

            bool matched = false;
            for( ; j < entriesEnd; j++ )
            {
                const size_t matchCount = (size_t)( j - i ) + 1;

                bbmemcpy_t( refMatch, ref+i, matchCount );
                bbmemcpy_t( tgtMatch, tgt+i, matchCount );
                
                std::sort( refMatch, refMatch+matchCount );
                std::sort( tgtMatch, tgtMatch+matchCount );

                if( memcmp( refMatch, tgtMatch, matchCount * sizeof( uint64 ) ) == 0 )
                {
                    matched = true;
                    break;
                }
                else
                    continue;
            }

            if( !matched )
            {
                Log::Line( "T7 park %lld failed. First entry failed at index %lld", p, i );
                parkFailCount++;
            }
        
            i = j;
        }

        // Set next park as current park
        memcpy( refParks[0], refParks[1], sizeof( uint64 ) * kEntriesPerPark );
        memcpy( tgtParks[0], tgtParks[1], sizeof( uint64 ) * kEntriesPerPark );
    }
    
    return parkFailCount;
}

//-----------------------------------------------------------
void TestTable( PlotInfo& ref, PlotInfo& tgt, TableId table )
{
    Log::Line( "Reading Table %u...", table+1 );

    const size_t parkSize = table < TableId::Table7 ? CalculateParkSize( table ) : CalculatePark7Size( ref.K() );

    const size_t sizeRef = ref.TableSize( (int)table );
    const size_t sizeTgt = tgt.TableSize( (int)table );

    byte* tableParksRef = bbvirtalloc<byte>( sizeRef );
    byte* tableParksTgt = bbvirtalloc<byte>( sizeTgt );

    ref.ReadTable( (int)table, tableParksRef );
    tgt.ReadTable( (int)table, tableParksTgt );

    const size_t tableSize = std::min( sizeRef, sizeTgt );
    const int64  parkCount = (int64)( tableSize / parkSize );

    const byte* parkRef = tableParksRef;
    const byte* parkTgt = tableParksTgt;
    
    Log::Line( "Validating Table %u...", table+1 );

    uint64 failureCount = 0;
    if( table == TableId::Table7 )
    {
        // Because entries can be found in different orders in P7,
        // we compare them in a different manner than the other parks.
        // (this can happen because these entries are sorted on f7,
        //  and if there's multiple entries with the same value
        //  there's no guarantee an implementation will sort it
        //  into the same index as another )
        failureCount = CompareP7( ref, tgt, tableParksRef, tableParksTgt, parkCount );
    }
    else
    {
        for( int64 i = 0; i < parkCount; i++ )
        {
            if( !MemCmp( parkRef, parkTgt, parkSize ) )
            {
                bool failed = true;

                if( failed )
                {
                    Log::Line( " T%u park %lld failed.", table+1, i );
                    failureCount++;
                }

                // {
                //     const byte* refBytes = parkRef;
                //     const byte* tgtBytes = parkTgt;

                //     // if( table != TableId::Table7 )
                //     // {
                //     //     refBytes += 8;
                //     // }

                //     for( uint64 b = 0; b < parkSize; b++ )
                //     {
                //         if( refBytes[b] != tgtBytes[b] )
                //         {
                //             Log::Line( "Byte mismatch @ byte %llu: 0x%02x (%3d) != 0x%02x (%3d)", b,
                //              (int)refBytes[b], (int)refBytes[b], (int)tgtBytes[b], (int)tgtBytes[b] );
                //             break;
                //         }
                //     }
                // }
            }

            parkRef += parkSize;
            parkTgt += parkSize;
        }
    }


    if( failureCount < 1 )
        Log::Line( "Success!" );
    else
        Log::Line( "%llu / %lld Table %u parks failed.", failureCount, parkCount, table+1 );

    SysHost::VirtualFree( tableParksRef );
    SysHost::VirtualFree( tableParksTgt );
}

// Unpack a single park 7,
// ensure srcBits is algined to uint64
//-----------------------------------------------------------
void UnpackPark7( const byte* srcBits, uint64* dstEntries )
{
    ASSERT( ((uintptr_t)srcBits & 7 ) == 0 );
    const uint32 _k = _K;

    const uint32 bitsPerEntry = _k + 1;
    CPBitReader reader( srcBits, CalculatePark7Size( _k ) * 8, 0  );

    for( int32 i = 0; i < kEntriesPerPark; i++ )
        dstEntries[i] = reader.Read64Aligned( bitsPerEntry );
}

//-----------------------------------------------------------
void DumpP7( PlotInfo& plot, const char* path )
{
    FileStream file;
    FatalIf( !file.Open( path, FileMode::Create, FileAccess::Write, FileFlags::LargeFile | FileFlags::NoBuffering ),
        "Failed to open file at '%s' with error: %d.", path, file.GetError() );

    const size_t parkSize = CalculatePark7Size( plot.K() );

    const size_t tableSize  = plot.TableSize( (int)TableId::Table7 );
    const int64  parkCount  = (int64)( tableSize / parkSize );
    const uint64 numEntries = (uint64)parkCount * kEntriesPerPark;

    byte* p7Bytes = bbvirtalloc<byte>( tableSize );

    Log::Line( "Reading Table7..." );
    plot.ReadTable( (int)TableId::Table7, p7Bytes );

    Log::Line( "Unpacking Table 7..." );
    uint64* entries = bbvirtalloc<uint64>( RoundUpToNextBoundaryT( (size_t)numEntries* sizeof( uint64 ), file.BlockSize() ) );

    const byte*   parkReader  = p7Bytes;
          uint64* entryWriter = entries;
    for( int64 i = 0; i < parkCount; i++ )
    {
        UnpackPark7( p7Bytes, entryWriter );

        parkReader  += parkSize;
        entryWriter += kEntriesPerPark;
    }

    Log::Line( "Writing to disk..." );
    const size_t blockSize = file.BlockSize();
    
    uint64* block = bbvirtalloc<uint64>( blockSize );   
    
    int err;
    FatalIf( !IOJob::WriteToFile( file, entries, numEntries * sizeof( uint64 ), block, blockSize, err ),
        "Entry write failed with error: %d.", err );

    Log::Line( "All done." );
    bbvirtfree( p7Bytes );
    bbvirtfree( entries );
    bbvirtfree( block   );
}


