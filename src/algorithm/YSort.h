#pragma once


class ThreadPool;


class YSorter
{
public:
    YSorter( ThreadPool& pool );
    ~YSorter();
    
    // template<uint MaxJobs>
    void Sort( 
        uint64 length, 
        uint64* yBuffer, uint64* yTmp );

    void Sort( 
        uint64 length, 
        uint64* yBuffer, uint64* yTmp,
        uint32* sortKey, uint32* sortKeyTmp );

private:
    void DoSort( bool useSortKey, uint64 length, 
                uint64* yBuffer, uint64* yTmp,
                uint32* sortKey, uint32* sortKeyTmp );
private:
    ThreadPool& _pool;
    // byte*       _pageCounts;
};




