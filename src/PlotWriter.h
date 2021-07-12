#pragma once
#include "io/FileStream.h"
#include "threading/Thread.h"
#include "threading/Semaphore.h"

/**
 * Handles writing the final plot to disk
 *
 * Final file format is as follows:
 * [Header]
 *   - "Proof of Space Plot" (utf-8) : 19 bytes
 *   - unique plot id                : 32 bytes
 *   - k                             : 1 byte  
 *   - format description length     : 2 bytes 
 *   - format description            : * bytes 
 *   - memo length                   : 2 bytes 
 *   - memo                          : * bytes 
 *   - table pointers                : 80 bytes (8 * 10 )
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
 */
class DiskPlotWriter
{
public:
    DiskPlotWriter();
    ~DiskPlotWriter();

    // Begins writing a new plot. Any previous plot must have finished before calling this
    bool BeginPlot( const char* plotFilePath, FileStream& file, const byte plotId[32],
                    const byte* plotMemo, const uint16 plotMemoSize );

    // Submits and signals the writing thread to write a table
    bool WriteTable( const void* buffer, size_t size );

    // Submits the table for writing, but does not actually write it to disk yet
    bool SubmitTable( const void* buffer, size_t size );

    // Flush pending tables to write
    // bool FlushTables();

    // Returns true if there's no errors.
    // If there are any errors, call GetError() to obtain the file write error.
    bool WaitUntilFinishedWriting();

    // Returns true if the plotter has finished writing the last queued plot.
    // (We nullify our reference when the file has been closed and finished.)
    inline bool HasFinishedWriting() { return _file == nullptr; }

    // If gotten after flush, they will be in big-endian format
    inline const uint64* GetTablePointers() { return _tablePointers; }

    template<typename T>
    inline T* AlignPointerToBlockSize( void* pointer )
    {
        return (T*)(uintptr_t)AlignToBlockSize( (uintptr_t)pointer );
    }

    inline int  GetError() { return _error; }

    inline const std::string& FilePath() { return _filePath; }

    // Number of tables written
    inline uint TablesWritten() { return _lastTableIndexWritten.load( std::memory_order_acquire ); }

private:
    static void WriterMain( void* data );
    void WriterThread();
    size_t AlignToBlockSize( size_t size );

    struct TableBuffer
    {
        const byte*  buffer;
        size_t size;
    };

private:
    FileStream* _file              = nullptr;
    std::string _filePath;
    size_t      _headerSize        = 0;
    byte*       _headerBuffer      = nullptr;
    size_t      _position          = 0;             // Current write position
    uint64      _tablePointers[10] = { 0 };         // Pointers to the table begin position
    TableBuffer _tablebuffers [10];                 // Table buffers passed to us for writing.

    std::atomic<uint> _tableIndex             = 0;  // Next table index to write
    std::atomic<uint> _lastTableIndexWritten  = 10; // Index of the latest table that was fully written to disk. (Owned by writer thread.)
                                                    //  That is, index-1 is the index of the last table written.
                                                    //  We start it at 10 (all tables written), to denote that we have
                                                    //  no tables pending to write.

    int               _error              = 0;      // Set if there was an error writing the plot file

    Thread            _writerThread;
    Semaphore         _writeSignal;                 // Main thread signals writer thread to write a new table
    Semaphore         _plotFinishedSignal;          // Writer thread signals that it's finished writing a plot
    std::atomic<bool> _terminateSignal    = false;  // Main thread signals us to exit
};

