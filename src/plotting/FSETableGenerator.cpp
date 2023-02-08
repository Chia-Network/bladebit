#define FSE_STATIC_LINKING_ONLY
#include "fse/fse.h"

#include <vector>
#include <cmath>
#include <queue>
#include "ChiaConsts.h"
#include "plotting/PlotTools.h"

static std::vector<short> CreateNormalizedCount(double R);

//-----------------------------------------------------------
void* GenFSETable( const double rValue, size_t* outTableSize, const bool compress )
{
    std::vector<short> nCount         = CreateNormalizedCount( rValue );
    unsigned           maxSymbolValue = (unsigned)nCount.size() - 1;
    unsigned           tableLog       = 14;

    FatalIf( maxSymbolValue > 255, "maxSymbolValue > 255" );
    
    size_t      err = 0;
    FSE_CTable* ct  = nullptr;
    FSE_CTable* dt  = nullptr;

    if( compress )
    {
        ct  = FSE_createCTable( maxSymbolValue, tableLog );
        err = FSE_buildCTable( ct, nCount.data(), maxSymbolValue, tableLog );
    }
    else
    {
        ct  = FSE_createDTable( tableLog );
        err = FSE_buildDTable( ct, nCount.data(), maxSymbolValue, tableLog );
    }
    
    FatalIf( FSE_isError( err ), "Failed to generate FSE compression table with error: %s", FSE_getErrorName( err ) );
    
    // const size_t tableSizeBytes = FSE_CTABLE_SIZE( tableLog, maxSymbolValue );

    if( outTableSize )
    {
        if( compress )
            *outTableSize = FSE_CTABLE_SIZE( tableLog, maxSymbolValue );
        else
            *outTableSize = FSE_DTABLE_SIZE( tableLog );
    }

    return ct;
}

//-----------------------------------------------------------
FSE_CTable* PlotTools::GenFSECompressionTable( const double rValue, size_t* outTableSize )
{
    return (FSE_CTable*)GenFSETable( rValue, outTableSize, true );
}

//-----------------------------------------------------------
FSE_DTable* PlotTools::GenFSEDecompressionTable( double rValue, size_t* outTableSize )
{
    return (FSE_CTable*)GenFSETable( rValue, outTableSize, false );
}

///
/// Taken from chiapos
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