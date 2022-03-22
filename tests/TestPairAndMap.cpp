#include "TestUtil.h"
#include "util/jobs/MemJobs.h"
#include "plotdisk/DiskFp.h"
#include "plotdisk/DiskPlotter.h"

#define WORK_TMP_PATH "/mnt/p5510a/disk_tmp/"

//-----------------------------------------------------------
TEST_CASE( "PairsAndMap", "[sandbox]" )
{
    ThreadPool pool( SysHost::GetLogicalCPUCount() );
    
    DiskBufferQueue queue( WORK_TMP_PATH, nullptr, 0, 1 );
}