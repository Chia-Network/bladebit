#include "PlotWriter.h"
#include "ChiaConsts.h"
#include "SysHost.h"
#include "Config.h"

//-----------------------------------------------------------
DiskPlotWriter::DiskPlotWriter()
    : _writeSignal       ( 0 )
    , _plotFinishedSignal( 0 )
{
    #if BB_BENCHMARK_MODE
        return;
    #endif
    // Start writer thread
    _writerThread.Run( WriterMain, this );
}

//-----------------------------------------------------------
DiskPlotWriter::~DiskPlotWriter()
{
    #if BB_BENCHMARK_MODE
        return;
    #endif

    // Signal writer thread to exit, if it hasn't already
    _terminateSignal.store( true, std::memory_order_release );
    _writeSignal.Release();

    // Wait for thread to exit
    _plotFinishedSignal.Wait();

    if( _headerBuffer )
        SysHost::VirtualFree( _headerBuffer );

    if( _file )
        delete _file;
}

//-----------------------------------------------------------
bool DiskPlotWriter::BeginPlot( const char* plotFilePath, FileStream& file, const byte plotId[32], const byte* plotMemo, const uint16 plotMemoSize )
{
    #if BB_BENCHMARK_MODE
        _filePath = plotFilePath;
        return true;
    #endif

    ASSERT( plotMemo     );
    ASSERT( plotMemoSize );

    // Make sure we're not still writing a plot
    if( _file || _error || !file.IsOpen() )
        return false;

    const size_t headerSize =
        ( sizeof( kPOSMagic ) - 1 ) +
        32 +            // plot id
        1  +            // k
        2  +            // kFormatDescription length
        ( sizeof( kFormatDescription ) - 1 ) +
        2  +            // Memo length
        plotMemoSize +  // Memo
        80              // Table pointers
    ;

    ASSERT( plotFilePath );
    _filePath = plotFilePath;


    const size_t paddedHeaderSize = RoundUpToNextBoundary( headerSize, (int)file.BlockSize() );
    
    byte* header = _headerBuffer;
    
    // Do we need to realloc?
    if( _headerSize )
    {
        ASSERT( _headerBuffer );

        const size_t currentPaddedHeaderSize = RoundUpToNextBoundary( _headerSize, (int)file.BlockSize() );

        if( currentPaddedHeaderSize != paddedHeaderSize )
        {
            SysHost::VirtualFree( header );
            header = nullptr;
        }
    }

    // Alloc the header buffer if we need to
    if( !header )
    {
        header = (byte*)SysHost::VirtualAlloc( paddedHeaderSize );

        // Zero-out padded portion and table pointers
        memset( header+headerSize-80, 0, paddedHeaderSize-headerSize-80 );
    }


    // Write the header
    {
        // Magic
        byte* headerWriter = header;
        memcpy( headerWriter, kPOSMagic, sizeof( kPOSMagic ) - 1 );
        headerWriter += sizeof( kPOSMagic ) - 1;

        // Plot Id
        memcpy( headerWriter, plotId, 32 );
        headerWriter += 32;

        // K
        *headerWriter++ = (byte)_K;

        // Format description
        *((uint16*)headerWriter) = Swap16( (uint16)(sizeof( kFormatDescription ) - 1) );
        headerWriter += 2;
        memcpy( headerWriter, kFormatDescription, sizeof( kFormatDescription ) - 1 );
        headerWriter += sizeof( kFormatDescription ) - 1;

        // Memo
        *((uint16*)headerWriter) = Swap16( plotMemoSize );
        headerWriter += 2;
        memcpy( headerWriter, plotMemo, plotMemoSize );
        headerWriter += plotMemoSize;

        // Tables will be copied at the end.
    }

    // Seek to the aligned position after the header
    if( !file.Seek( (int64)paddedHeaderSize, SeekOrigin::Begin ) )
    {
        _error = file.GetError();
        return false;
    }
    
    // Save header so that we can actually write it to the end of the file
    _headerBuffer = header;
    _headerSize   = headerSize;

    // Store initial table offset
    _tablePointers[0] = paddedHeaderSize;
    _position         = paddedHeaderSize;

    // Give ownership of the file to the writer thread and signal it
    _tableIndex = 0;
    _file       = &file;
    _writeSignal.Release();

    return true;
}

//-----------------------------------------------------------
bool DiskPlotWriter::WriteTable( const void* buffer, size_t size )
{
    #if BB_BENCHMARK_MODE
        return true;
    #endif

    if( !SubmitTable( buffer, size ) )
        return false;

    // Signal the writer thread that there is a new table to write
    _writeSignal.Release();

    return true;
}

//-----------------------------------------------------------
bool DiskPlotWriter::SubmitTable( const void* buffer, size_t size )
{
    #if BB_BENCHMARK_MODE
        return true;
    #endif

    // Make sure the thread has already started.
    // We must have no errors
    if( !_file || _error )
        return false;

    const uint tableIndex = _tableIndex.load( std::memory_order_relaxed );
    ASSERT( tableIndex < 10 );

    // Can't overflow tables
    if( tableIndex >= 10 )
        return false;

    ASSERT( buffer );
    ASSERT( size   );

    _tablebuffers[tableIndex].buffer = (byte*)buffer;
    _tablebuffers[tableIndex].size   = size;

    // Store the value
    _tableIndex.store( tableIndex + 1, std::memory_order_release );

    return true;
}

// //-----------------------------------------------------------
// bool DiskPlotWriter::FlushTables()
// {
//     // Make sure the thread has already started.
//     if( _headerSize < 1 )
//         return false;

//     // We must have no errors
//     if( _error )
//         return false;
    
//     // Signal the writer thread for each table we have
//     const uint tableIndex = _tableIndex.load( std::memory_order_relaxed );

//     for( uint i = 0; i < tableIndex; i++ )
//         _writeSignal.Release();

//     return true;
// }

//-----------------------------------------------------------
bool DiskPlotWriter::WaitUntilFinishedWriting()
{
#if BB_BENCHMARK_MODE
    return true;
#else

    // For re-entry checks
    if( !_file && _plotFinishedSignal.GetCount() == 0 )
        return true;

    do {
        _plotFinishedSignal.Wait();
    } while( _file );

    ASSERT( _file == nullptr );

    return _error == 0;
#endif
}


///
/// Writer thread
///
//-----------------------------------------------------------
void DiskPlotWriter::WriterMain( void* data )
{
    SysHost::SetCurrentThreadAffinityCpuId( 0 );
    
    ASSERT( data );
    DiskPlotWriter* self = reinterpret_cast<DiskPlotWriter*>( data );
    
    self->WriterThread();
}

//-----------------------------------------------------------
void DiskPlotWriter::WriterThread()
{
    if( _terminateSignal.load( std::memory_order_relaxed ) )
        return;

    FileStream* file = nullptr;

    uint tableIndex = 0;    // Local table index

    // Buffer for writing 
    size_t blockBufferSize = 0;
    byte*  blockBuffer     = nullptr;
    size_t blockSize       = 0;

    for( ;; )
    {
        // Wait to be signalled by the main thread
        _writeSignal.Wait();

        if( _terminateSignal.load( std::memory_order_relaxed ) )
            break;

        // Started writing a new table, ensure we have a file
        if( file == nullptr )
        {
            file = _file;

            // We may have been signalled multiple times to write tables 
            // and we grabbed the tables without waiting for the signal,
            // so it might occurr that we don't have a file yet.
            if( !file )
                continue;

            // Reset table index
            tableIndex = 0;
            _lastTableIndexWritten.store( 0, std::memory_order_release );

            // Allocate a new block buffer, if we need to
            blockSize = file->BlockSize();
            if( blockSize > blockBufferSize )
            {
                if( blockBuffer )
                    SysHost::VirtualFree( blockBuffer );

                blockBufferSize = blockSize;
                blockBuffer     = (byte*)SysHost::VirtualAlloc( blockBufferSize );

                if( !blockBuffer )
                    Fatal( "Failed to allocate buffer for writing to disk." );
            }
        }

        // See if we have a new table to write (should always be the case when we're signaled)
        uint newIndex = _tableIndex.load( std::memory_order_acquire );

        while( tableIndex < newIndex )
        {
            TableBuffer& table       = _tablebuffers[tableIndex];

            const byte*  writeBuffer = table.buffer;

            // Write as many blocks as we can, 
            // then write the remainder by copying it to our own block-aligned buffer
            const size_t blockCount  = table.size / blockSize;
            size_t       sizeToWrite = blockCount * blockSize;

            const size_t remainder   = table.size - sizeToWrite;

            while( sizeToWrite )
            {
                ssize_t sizeWritten = file->Write( writeBuffer, sizeToWrite );
                if( sizeWritten < 1 )
                {
                    // Error occurred, stop writing.
                    _error = file->GetError();
                    break;
                }
                ASSERT( (size_t)sizeWritten <= sizeToWrite );

                sizeToWrite -= (size_t)sizeWritten;
                writeBuffer += sizeWritten;
            };

            // Break out if we got a write error
            if( _error )
                break;

            // Write remainder, if we have any
            if( remainder )
            {
                memset( blockBuffer, 0, blockSize );
                memcpy( blockBuffer, writeBuffer, remainder );

                ssize_t sizeWritten = file->Write( blockBuffer, blockSize );
                if( sizeWritten < 1 )
                {
                    _error = file->GetError();
                    break;   
                }

                ASSERT( (size_t)sizeWritten == blockSize );
            }

            if( !file->Flush() )
            {
                _error = file->GetError();
                break;
            }

            // Go to the next table
            tableIndex ++;
            _lastTableIndexWritten.store( tableIndex, std::memory_order_release );

            _position += RoundUpToNextBoundary( table.size, (int)blockSize );

            // Save the table pointer
            _tablePointers[tableIndex] = _position;
        }
        
        if( _error )
            break;
        
        // Have we finished writing the last table?
        if( tableIndex >= 10 )
        {
            ASSERT( tableIndex == 10 );

            // We now need to seek to the beginning so that we can write the header
            // with the table pointers set
            if( file->Seek( 0, SeekOrigin::Begin ) )
            {
                const size_t alignedHeaderSize = _tablePointers[0];

                // Convert to BE
                for( uint i = 0; i < 10; i++ )
                    _tablePointers[i] = Swap64( _tablePointers[i] );

                memcpy( _headerBuffer + (_headerSize-80), _tablePointers, 80 );

                if( file->Write( _headerBuffer, alignedHeaderSize ) != (ssize_t)alignedHeaderSize )
                    _error = file->GetError();
            }
            else
                _error = file->GetError();

            // Plot data cleanup
            if( !file->Flush() )
                _error = file->GetError();
            
            file->Close();
            delete file;
            file = nullptr;
            _file = nullptr;

            _tableIndex = 0;

            // Signal that we've finished writing the plot
            _plotFinishedSignal.Release();
        }
    }


    ///
    /// Cleanup
    ///
    if( blockBuffer )
        SysHost::VirtualFree( blockBuffer );

    // Close the file in case we had an error
    if( file )
    {
        file->Close();
        delete file;
    }

    _file = nullptr;

    // Signal that this thread is finished
    _plotFinishedSignal.Release();
}

//-----------------------------------------------------------
size_t DiskPlotWriter::AlignToBlockSize( size_t size )
{
    #if BB_BENCHMARK_MODE
        return RoundUpToNextBoundary( size, 4096 );
    #endif

    ASSERT( _file );

    // Shoud never be called without a file
    if( !_file )
        return size;

    return RoundUpToNextBoundary( size, (int)_file->BlockSize() );
}


