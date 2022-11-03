#include "MemPhase4.h"
#include "plotting/CTables.h"
#include "util/Log.h"

//-----------------------------------------------------------
MemPhase4::MemPhase4( MemPlotContext& context )
    : _context( context )
{}

//-----------------------------------------------------------
void MemPhase4::Run()
{
    // Use meta0 to write the final tables to disk
    MemPlotContext& cx = _context;
    
    // The first 32 GiB of meta0 are used by phase 3 to write the table 6 park,
    // so we need to offset here to write the rest.
    cx.p4WriteBuffer = ((byte*)cx.metaBuffer0) + 32ull GB;
    cx.p4WriteBufferWriter = cx.p4WriteBuffer;

    WriteP7();
    WriteC1();
    WriteC2();
    WriteC3();
}

//-----------------------------------------------------------
void MemPhase4::WriteP7()
{
    // Write P7 (Table 7 park), which are indices into
    // the previous table's LinePoints (which are parked as well).

    MemPlotContext& cx = _context;

    const uint32* lTable     = cx.t1XBuffer;          // L table is passed around in the t1XBuffer
    const uint64  entryCount = cx.entryCount[(int)TableId::Table7];
    byte*         p7Buffer   = cx.plotWriter->AlignPointerToBlockSize<byte>( cx.p4WriteBufferWriter ); // This buffer is big enough to hold the whole park
    
    Log::Line( "  Writing P7." );
    auto timer = TimerBegin();

    const size_t sizeWritten = WriteP7Parallel<MAX_THREADS>( *cx.threadPool, entryCount, lTable, p7Buffer );
    
    cx.p4WriteBufferWriter = ((byte*)p7Buffer) + sizeWritten;
    
    if( !cx.plotWriter->WriteTable( p7Buffer, sizeWritten ) )
        Fatal( "Failed to write P7 to disk." );

    double elapsed = TimerEnd( timer );
    Log::Line( "  Finished writing P7 in %.2lf seconds.", elapsed );
}

//-----------------------------------------------------------
void MemPhase4::WriteC1()
{
    MemPlotContext& cx = _context;
 
    const uint64 entryCount  = cx.entryCount[(int)TableId::Table7];
    uint32*      writeBuffer = cx.plotWriter->AlignPointerToBlockSize<uint32>( cx.p4WriteBufferWriter );

    Log::Line( "  Writing C1 table." );
    auto timer = TimerBegin();

    const size_t sizeWritten = WriteC12Parallel<MAX_THREADS, kCheckpoint1Interval>( 
        *cx.threadPool, entryCount, cx.t7YBuffer, writeBuffer );

    cx.p4WriteBufferWriter = ((byte*)writeBuffer) + sizeWritten;

    if( !cx.plotWriter->WriteTable( writeBuffer, sizeWritten ) )
        Fatal( "Failed to write C1 to disk." );

    double elapsed = TimerEnd( timer );
    Log::Line( "  Finished writing C1 table in %.2lf seconds.", elapsed );
}

//-----------------------------------------------------------
void MemPhase4::WriteC2()
{
    MemPlotContext& cx = _context;
 
    const uint64 entryCount  = cx.entryCount[(int)TableId::Table7];
    uint32*      writeBuffer = cx.plotWriter->AlignPointerToBlockSize<uint32>( cx.p4WriteBufferWriter );

    Log::Line( "  Writing C2 table." );
    auto timer = TimerBegin();

    const size_t sizeWritten = WriteC12Parallel<MAX_THREADS, kCheckpoint1Interval*kCheckpoint2Interval>( 
        *cx.threadPool, entryCount, cx.t7YBuffer, writeBuffer );

    cx.p4WriteBufferWriter = ((byte*)writeBuffer) + sizeWritten;

    if( !cx.plotWriter->WriteTable( writeBuffer, sizeWritten ) )
        Fatal( "Failed to write C2 to disk." );

    double elapsed = TimerEnd( timer );
    Log::Line( "  Finished writing C2 table in %.2lf seconds.", elapsed );
}

//-----------------------------------------------------------
void MemPhase4::WriteC3()
{
    MemPlotContext& cx = _context;
 
    const uint64 entryCount  = cx.entryCount[(int)TableId::Table7];
    byte*        writeBuffer = cx.plotWriter->AlignPointerToBlockSize<byte>( cx.p4WriteBufferWriter );

    Log::Line( "  Writing C3 table." );
    auto timer = TimerBegin();

    const size_t sizeWritten = WriteC3Parallel<MAX_THREADS>( 
         *cx.threadPool, entryCount, cx.t7YBuffer, writeBuffer );

    cx.p4WriteBufferWriter = ((byte*)writeBuffer) + sizeWritten;

    if( !cx.plotWriter->WriteTable( writeBuffer, sizeWritten ) )
        Fatal( "Failed to write C3 to disk." );

    double elapsed = TimerEnd( timer );
    Log::Line( "  Finished writing C3 table in %.2lf seconds.", elapsed );
}


