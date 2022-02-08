#include <vector>
#include <cmath>
#include <queue>
#include "ChiaConsts.h"

static std::vector<short> CreateNormalizedCount(double R);

/// Offline generate tables required for ANS encoding/decoding

//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    /// Generate for C3
    
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