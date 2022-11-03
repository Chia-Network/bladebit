#pragma once
#include "plotting/PlotTools.h"
#include "plotting/PlotTypes.h"
#include "io/FileStream.h"
#include "util/Util.h"
#include <vector>

class CPBitReader;

struct PlotHeader
{
    byte   id  [BB_PLOT_ID_LEN]        = { 0 };
    byte   memo[BB_PLOT_MEMO_MAX_SIZE] = { 0 };
    uint   memoLength                  = 0;
    uint32 k                           = 0;
    uint64 tablePtrs[10]               = { 0 };
};

// Base Abstract class for read-only plot files
class IPlotFile
{
public:

    inline uint K() const { return _header.k; }

    inline const byte* PlotId() const { return _header.id; }

    inline uint PlotMemoSize() const { return _header.memoLength; }

    inline const byte* PlotMemo() const { return _header.memo; }

    inline uint64 TableAddress( PlotTable table ) const
    {
        ASSERT( table >= PlotTable::Table1 && table <= PlotTable::C3 );
        return _header.tablePtrs[(int)table];
    }

    inline size_t TableSize( PlotTable table )
    {
        ASSERT( table >= PlotTable::Table1 && table <= PlotTable::C3 );

        const uint64 address    = _header.tablePtrs[(int)table];
        uint64       endAddress = PlotSize();

        // Check all table entris where we find and address that is 
        // greater than ours and less than the current end address
        for( int i = 0; i < 10; i++ )
        {
            const uint64 a = _header.tablePtrs[i];
            if( a > address && a < endAddress )
                endAddress = a;
        }

        return (size_t)( endAddress - address );
    }

    // #NOTE: User must check for read errors!
    inline uint16 ReadUInt16()
    {
        uint16 value = 0;
        const ssize_t read = Read( sizeof( value ), &value );
        if( read < 0 )
            return 0;
        
        ASSERT( read == sizeof( uint16 ) );
        if( read != sizeof( uint16 ) )
        {
            // #TODO: Set proper error
            return 0;
        }

        return Swap16( value );
    }

    // Abstract Interface
public:
    virtual bool Open( const char* path ) = 0;
    virtual bool IsOpen() const = 0;

    // Plot size in bytes
    virtual size_t PlotSize() const = 0;
    
    // Read data from the plot file
    virtual ssize_t Read( size_t size, void* buffer ) = 0;

    // Seek to the specific location on the plot stream,
    // whatever the underlying implementation may be.
    virtual bool Seek( SeekOrigin origin, int64 offset ) = 0;

    // Get last error ocurred
    virtual int GetError() = 0;

protected:

    // Implementors can call this to load the header
    bool ReadHeader( int& error );

protected:
    PlotHeader _header;
};

class MemoryPlot : public IPlotFile
{
public:
    MemoryPlot();
    MemoryPlot( const MemoryPlot& plotFile );
    ~MemoryPlot();

    bool Open( const char* path ) override;
    bool IsOpen() const override;

    size_t PlotSize() const override;
    
    ssize_t Read( size_t size, void* buffer ) override;

    bool Seek( SeekOrigin origin, int64 offset ) override;

    int GetError() override;

private:
    Span<byte>  _bytes;  // Plot bytes
    int         _err      = 0;
    ssize_t     _position = 0;
    std::string _plotPath = "";
};

class FilePlot : public IPlotFile
{
public:
    FilePlot();
    FilePlot( const FilePlot& file );
    ~FilePlot();

    bool Open( const char* path ) override;
    bool IsOpen() const override;

    size_t PlotSize() const override;

    ssize_t Read( size_t size, void* buffer ) override;

    bool Seek( SeekOrigin origin, int64 offset ) override;

    int GetError() override;

private:
    FileStream  _file;
    std::string _plotPath = "";
};

class PlotReader
{
public:
    PlotReader( IPlotFile& plot );
    ~PlotReader();

    uint64 GetC3ParkCount() const;

    // Get the maximum potential F7 count.
    // This may be more than the actual number of F7s that we have,
    // since the last park is likely not full.
    uint64 GetMaxF7EntryCount() const;

    size_t GetTableParkCount( const PlotTable table ) const;

    uint64 GetMaximumC1Entries() const;

    // Returns the number of apparent C1 entries (same as C3 park count),
    // excluding any empty space appearing after the table for alignment,
    // This may not be accurate if the C1 entries are malformed and 
    // not sorted in ascending order.
    // This method performs various reads until it find a seemingly valid C1 entries
    bool GetActualC1EntryCount( uint64& outC1Count );

    // Read a whole C3 park into f7Buffer.
    // f7Buffer must hold at least as many as the amount of entries
    // required per C3Park. (kCheckpoint1Interval).
    // The return value should be the entries in the park.
    // It should never be 0.
    // If the return value is negative, there was an error reading the park.
    int64 ReadC3Park( uint64 parkIndex, uint64* f7Buffer );

    bool ReadP7Entries( uint64 parkIndex, uint64* p7Indices );

    uint64 GetFullProofForF7Index( uint64 f7Index, byte* fullProof );

    bool FetchProof( uint64 t6LPIndex, uint64 fullProofXs[BB_PLOT_PROOF_X_COUNT] );

    // void   FindF7ParkIndices( uintt64 f7, std::vector<uint64> indices );
    bool ReadLPPark( TableId table, uint64 parkIndex, uint128 linePoints[kEntriesPerPark], uint64& outEntryCount );

    bool ReadLP( TableId table, uint64 index, uint128& outLinePoint );

    bool FetchProofFromP7Entry( uint64 p7Entry, uint64 proof[32] );

    inline IPlotFile& PlotFile() const { return _plot; }

    Span<uint64> GetP7IndicesForF7( const uint64 f7, Span<uint64> indices );
    
private:

    bool ReadLPParkComponents( TableId table, uint64 parkIndex, 
                               CPBitReader& outStubs, byte*& outDeltas, 
                               uint128& outBaseLinePoint, uint64& outDeltaCounts );

    bool LoadC2Entries();
private:
    IPlotFile& _plot;
    uint32     _version;

    // size_t  _parkBufferSize;
    uint64*      _parkBuffer    ;        // Buffer for loading compressed park data.
    byte*        _deltasBuffer  ;        // Buffer for decompressing deltas in parks that have delta. 

    Span<uint64> _c2Entries;
    byte*        _c1Buffer = nullptr;
    Span<uint64> _c3Buffer;
};

