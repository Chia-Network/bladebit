#include <vector>
#include <cmath>
#include <queue>
#include "ChiaConsts.h"

// Need to define this for FSE_CTABLE_SIZE/FSE_DTABLE_SIZE
#define FSE_STATIC_LINKING_ONLY
#include "fse/fse.h"


///
/// Offline generate tables required for ANS encoding/decoding
///
const char USAGE[] = "fsegen [OPTIONS] [<output_path>]\n"
R"(
output_path: Output file path. If nothing is specified, STDOUT is used.

OPTIONS:
 -d, --decompress: Generate decompression table.
)";

static std::vector<short> CreateNormalizedCount(double R);
static void DumpFSETables( FILE* file, bool compression );


//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    argc--;
    argv++;

    bool compress = true;

    const char* outFilePath = nullptr;
    
    for( int i = 0; i < argc; i++ )
    {
        const char* arg = argv[i];
        if( strcmp( "-d", arg ) == 0 || strcmp( "--decompress", arg ) == 0 )
            compress = false;
        else if( i < argc-1 )
        {
            Fatal( "Unknown argument '%s'", arg );
        }
        else
            outFilePath = arg;
    }

    FILE* file = nullptr;
    if( outFilePath )
    {
        file = fopen( outFilePath, "wb" );
        FatalIf( !file, "Failed to open file for writing at '%s'.", outFilePath );
    }
    else
    {
        file = stdout;
    }

    DumpFSETables( file, compress );
    return 0;
}

//-----------------------------------------------------------
void DumpFSETables( FILE* file, bool compression )
{
    ASSERT( file );
    const char* prefix = compression ? "C" : "D";

    fprintf( file, "#pragma once\n" );
    fprintf( file, "#include \"fse/fse.h\"\n\n" );


    for( int i = 0; i < 7; i++ )
    {
        double R = i < 6 ? kRValues[i] : kC3R;  // Last one is for C3 entries

        std::vector<short> nCount = CreateNormalizedCount(R);
        unsigned maxSymbolValue = (unsigned)nCount.size() - 1;
        unsigned tableLog       = 14;

        FatalIf( maxSymbolValue > 255, "maxSymbolValue > 255" );
        
        size_t      err = 0;
        FSE_CTable* ct  = nullptr;

        if( compression )
        {
            ct  = FSE_createCTable( maxSymbolValue, tableLog );
            err = FSE_buildCTable( ct, nCount.data(), maxSymbolValue, tableLog );
        }
        else
        {
            // FSE_CTable and FSE_DTable are just uint typedefs, so we can use FSE_CTable here
            ct  = FSE_createDTable( tableLog );
            err = FSE_buildDTable( ct, nCount.data(), maxSymbolValue, tableLog );
        }

        FatalIf( FSE_isError( err ), "Failed to build %sTable for table %d: %s", 
            prefix, i+1, FSE_getErrorName( err ) );
        
        const size_t tableSizeBytes = compression ? 
            FSE_CTABLE_SIZE( tableLog, maxSymbolValue ) : FSE_DTABLE_SIZE( tableLog );

        const size_t rowSize        = 32;
        const size_t nRows          = tableSizeBytes / rowSize;
        const size_t lastRowCount   = tableSizeBytes - ( nRows * rowSize );

        const uint8_t* bytes = (uint8_t*)ct;

        if( i < 6 )
            fprintf( file, "const byte %sTable_%d[%llu] = {\n", prefix, i, tableSizeBytes );
        else
            fprintf( file, "const byte %sTable_C3[%llu] = {\n", prefix, tableSizeBytes );
        
        for( size_t j = 0; j < nRows; j++ )
        {
            fprintf( file, "  " );
            const uint8_t* row = bytes + j * rowSize;
            for( size_t r = 0; r < rowSize; r++ )
                fprintf( file, "0x%02hhx, ",  row[r] );

            fprintf( file, "\n" );
        }
        
        if( lastRowCount )
        {
            fprintf( file, "  " );
            for( size_t j = 0; j < lastRowCount; j++ )
            {
                fprintf( file, "0x%02hhx",  bytes[rowSize*nRows+j] );
                
                if( j+1 < lastRowCount )
                    fprintf( file, ", " );
            }

            fputc( '\n', file );
        }

        fprintf( file, "};\n\n" );
        fflush( file );
    }

    fprintf( file, "const FSE_%sTable* const %sTables[6] = {\n", prefix, prefix );
    for( int i = 0; i < 6; i++ )
    {
        fprintf( file, "  (const FSE_%sTable*)%sTable_%d,\n", prefix, prefix, i );
    }
    fprintf( file, "};\n\n" );
    fclose( file );
}


///
/// Takem from chiapos
///
std::vector<short> CreateNormalizedCount(double R)
{
    std::vector<double> dpdf;
    int N = 0;
    double E = 2.718281828459;
    double MIN_PRB_THRESHOLD = 1e-50;
    int TOTAL_QUANTA = 1 << 14;
    double p = 1 - pow((E - 1) / E, 1.0 / R);

    while (p > MIN_PRB_THRESHOLD && N < 255) {
        dpdf.push_back(p);
        N++;
        p = (pow(E, 1.0 / R) - 1) * pow(E - 1, 1.0 / R);
        p /= pow(E, ((N + 1) / R));
    }

    std::vector<short> ans(N, 1);
    auto cmp = [&dpdf, &ans](int i, int j) {
        return dpdf[i] * (log2(ans[i] + 1) - log2(ans[i])) <
                dpdf[j] * (log2(ans[j] + 1) - log2(ans[j]));
    };

    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
    for (int i = 0; i < N; ++i) pq.push(i);

    for (int todo = 0; todo < TOTAL_QUANTA - N; ++todo) {
        int i = pq.top();
        pq.pop();
        ans[i]++;
        pq.push(i);
    }

    for (int i = 0; i < N; ++i) {
        if (ans[i] == 1) {
            ans[i] = (short)-1;
        }
    }
    return ans;
}