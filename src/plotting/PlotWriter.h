#pragma once

#include "util/SPCQueue.h"
#include "plotting/PlotTypes.h"
#include "plotting/PlotHeader.h"
#include "tools/PlotChecker.h"
#include "io/FileStream.h"
#include "threading/Thread.h"
#include "threading/AutoResetSignal.h"
#include "threading/Fence.h"
#include <functional>
#include <mutex>
#include <queue>

/**
 * Handles writing the final plot data to disk asynchronously.
 *
 * The tables may appear in any order, but the [C2_Table] MUST appear immediately before [C3_Table_Parks].
 * This is to keep in step with how chiapos loads the C2 table currently.
 * 
 * Final file format is as follows:
 * 
 * VERSION 1.0:
 * [Header]
 *   - "Proof of Space Plot" (utf-8) : 19 bytes
 *   - unique plot id                : 32 bytes
 *   - k                             : 1 byte  
 *   - format description length     : 2 bytes 
 *   - format description            : * bytes 
 *   - memo length                   : 2 bytes 
 *   - memo                          : * bytes 
 *   - table pointers                : 80 bytes (8 * 10)
 * [(Optional_Padding)]
 * [Table1_Parks]
 * [Table2_Parks]
 * [Table3_Parks]
 * [Table4_Parks]
 * [Table5_Parks]
 * [Table6_Parks]
 * [Table7_Parks]
 * [C1_Table]
 * [C2_Table]
 * [C3_Table_Parks]
 * 
 * 
 * VERSION 2.0:
 * 
 * [Header] is modified in the following way:
 * 
 * [V2_Header]
 *   - magic number (0x544F4C50 'PLOT') : 4  bytes
 *   - version number                   : 4  bytes
 *   - unique plot id                   : 32 bytes
 *   - k                                : 1  byte
 *   - memo length                      : 2  bytes
 *   - memo                             : *  bytes
 *   - plot flags ([Plot_Flags])        : 4 bytes
 *   - (if compression flag is set)
 *    - compression level (1-9)         : 1 byte
 *   - table pointers                   : 80 bytes (8 * 10)
 *   - table sizes                      : 80 bytes (8 * 10)
 * 
 * The rest of the file is the normal payload tables, but [Table1_Parks] and [Table2_Parks]
 * may be missing depending of the compression level. In this case the
 * table pointers of the missing tables should be set to the 
 * start of the next available table, and their sizes set ot 0.
 * 
 */

class FileStream;
class Fence;
class Thread;
class DiskBufferQueue;

class PlotWriter
{
    friend class DiskBufferQueue;

    struct Command;
    enum class CommandType : uint32;

    static constexpr size_t BUFFER_ALLOC_SIZE = 32 MiB;
public:

    PlotWriter();
    PlotWriter( bool useDirectIO );
    PlotWriter( DiskBufferQueue& ownerQueue );
    virtual ~PlotWriter();
    
    void EnablePlotChecking( PlotChecker& checker );

    // Begins writing a new plot. Any previous plot must have finished before calling this
    bool BeginPlot( PlotVersion version, 
        const char* plotFileDir, const char* plotFileName, const byte plotId[32],
        const byte* plotMemo, const uint16 plotMemoSize, uint32 compressionLevel = 0 );

    // bool BeginCompressedPlot( PlotVersion version, 
    //     const char* plotFileDir, const char* plotFileName, const byte plotId[32],
    //     const byte* plotMemo, const uint16 plotMemoSize, uint32 compressionLevel );

    void EndPlot( bool rename = true );

    void WaitForPlotToComplete();

    void DumpTables();

    // Begin plotting a table
    void BeginTable( const PlotTable table );

    // End current table
    void EndTable();

    // Don't actually write table data, but reserve the size for it and seek
    // to the offset that would be written
    void ReserveTableSize( const PlotTable table, const size_t size );

    // Write data to the currently active table
    void WriteTableData( const void* data, const size_t size );

    // Write all data to a reserved table.
    // The whole buffer must be specified
    void WriteReservedTable( const PlotTable table, const void* data );

    void SignalFence( Fence& fence );
    void SignalFence( Fence& fence, uint32 sequence );

    // Dispatch a callback from the writer thread
    void CallBack( std::function<void()> func );
    
    void CompleteTable();

    inline int32 GetError() { return _stream.GetError(); }

    inline size_t BlockSize() const { return _stream.BlockSize(); }

    inline const uint64* GetTablePointers() const { return _tablePointers; }

    inline const uint64* GetTableSizes() const { return _tableSizes; }

    inline const char* GetLastPlotFileName() const { return _plotFinalPathName ? _plotFinalPathName : ""; }

    size_t BlockAlign( size_t size ) const;

    template<typename T>
    inline T* BlockAlignPtr( void* ptr ) const
    {
        return (T*)(uintptr_t)BlockAlign( (size_t)(uintptr_t)ptr );
    }

    inline void EnableDummyMode()
    {
        ASSERT( !_haveTable );
        ASSERT( _headerSize == 0 );
        ASSERT( _position == 0 );
        _dummyMode = true;
    }

private:

    bool BeginPlotInternal( PlotVersion version,
        const char* plotFileDir, const char* plotFileName, const byte plotId[32],
        const byte* plotMemo, const uint16 plotMemoSize,
        int32 compressionLevel );

    bool CheckPlot();

    Command& GetCommand( CommandType type );
    void SubmitCommands();
    void SubmitCommand( const Command cmd );
    
    void SeekToLocation( size_t location );

    void ExitWriterThread();

    static void WriterThreadEntry( PlotWriter* self );
    void WriterThreadMain();
    void ExecuteCommand( const Command& cmd );
    void FlushRetainedBytes();

    void WriteData( const byte* data, size_t size );


private:
    void CmdBeginTable( const Command& cmd );
    void CmdEndTable( const Command& cmd );
    void CmdWriteTable( const Command& cmd );
    void CmdReserveTable( const Command& cmd );
    void CmdWriteReservedTable( const Command& cmd );
    void CmdSignalFence( const Command& cmd );
    void CmdEndPlot( const Command& cmd );
    void CmdCallBack( const Command& cmd );

private:
    enum class CommandType : uint32
    {
        None = 0,
        Exit,
        BeginTable,
        EndTable,
        WriteTable,
        ReserveTable,
        WriteReservedTable,
        SignalFence,
        EndPlot,
        CallBack,
    };

    struct Command
    {
        CommandType type;

        union {
            struct
            {
                PlotTable table;
            } beginTable;

            struct
            {
                PlotTable table;
                size_t size;
            } reserveTable;

            // Also used for WriteReservedTable
            struct 
            {
                const byte* buffer;
                size_t      size;
            } writeTable;

            struct 
            {
                PlotTable   table;
                const byte* buffer;
            } writeReservedTable;

            struct
            {
                PlotTable table;
            } endTable;

            // Also used for EndPlot
            struct
            {
                Fence* fence;
                int64  sequence;
            } signalFence;

            struct
            {
                Fence* fence;
                bool   rename;
            } endPlot;

            struct
            {
                std::function<void()>* func;
            } callback;
        };
    };


private:
    class DiskBufferQueue* _owner               = nullptr;  // This instance might be own by an IOQueue, which will 
                                                            // dispatch our ocmmands in its own threads.

    FileStream             _stream;
    bool                   _directIO;
    bool                   _dummyMode           = false;    // In this mode we don't actually write anything
    PlotVersion            _plotVersion         = PlotVersion::v2_0;
    Span<char>             _plotPathBuffer      = {};
    char*                  _plotFinalPathName   = {};
    Thread*                _writerThread        = nullptr;
    Fence                  _completedFence;             // Signal plot completed
    AutoResetSignal        _cmdReadySignal;
    AutoResetSignal        _cmdConsumedSignal;
    AutoResetSignal        _readyToPlotSignal;          // Set when the writer is ready to start the next plot.
    Span<byte>             _writeBuffer         = {};
    size_t                 _bufferBytes         = 0;    // Current number of bytes in the buffer
    size_t                 _headerSize          = 0;
    bool                   _haveTable           = false;
    PlotTable              _currentTable        = PlotTable::Table1;
    size_t                 _position            = 0;    // Current read/write location, relative to the start of the file
    size_t                 _unalignedFileSize   = 0;    // Current total file size, including headers, but excluding any extra alignment bytes
    size_t                 _alignedFileSize     = 0;    // Current actual size of data we've written to disk.
                                                        //  This is different than _unalignedFileSize because the latter might 
                                                        //  have data accounted in the buffer, but not written to disk. In which
                                                        //  case the _alignedFileSize is lesser than _unalignedFileSize.
                                                        
    // size_t                 _tablesBeginAddress  = 0;    // Start location for tables section
    size_t                 _tableStart          = 0;    // Current table start location
    uint64                 _tablePointers[10]   = {};
    uint64                 _tableSizes   [10]   = {};
    // SPCQueue<Command, 512> _queue;

    std::queue<Command>     _queue;
    std::mutex              _queueLock;
    // std::mutex              _pushLock;

    PlotChecker* _plotChecker              = nullptr;    // User responsible for ownership of checker. Must live until this PlotWriter's lifetime neds.
};

