#pragma once


enum class FileId
{
    None = 0,

    // Phase 1 fx, key, metadata
    FX0, FX1,

    // Bounded Phase 1:
    META0, META1,
    INDEX0, INDEX1,

    // Table 7 fx values
    F7,

    // Back pointers
    T1, // 1  : X values
    T2, // 2-7: Back pointers
    T3,
    T4,
    T5,
    T6,
    T7,

    // Maps from match order to y-sorted order
    MAP2,
    MAP3,
    MAP4,
    MAP5,
    MAP6,
    MAP7,
    
    // Marked entries for prunning
    MARKED_ENTRIES_2,
    MARKED_ENTRIES_3,
    MARKED_ENTRIES_4,
    MARKED_ENTRIES_5,
    MARKED_ENTRIES_6,

    // Line points
    LP,
    LP_MAP_0,
    LP_MAP_1,

    PLOT

    ,_COUNT
}; ImplementArithmeticOps( FileId );


//-----------------------------------------------------------
inline FileId TableIdToSortKeyId( const TableId table )
{
    // switch( table )
    // {
    //     case TableId::Table2: return FileId::SORT_KEY2;
    //     case TableId::Table3: return FileId::SORT_KEY3;
    //     case TableId::Table4: return FileId::SORT_KEY4;
    //     case TableId::Table5: return FileId::SORT_KEY5;
    //     case TableId::Table6: return FileId::SORT_KEY6;
    //     case TableId::Table7: return FileId::SORT_KEY7;
    
    //     default:
    //         ASSERT( 0 );
    //         break;
    // }
    
    ASSERT( 0 );
    return FileId::None;
}

//-----------------------------------------------------------
inline FileId TableIdToBackPointerFileId( const TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::T2;
        case TableId::Table3: return FileId::T3;
        case TableId::Table4: return FileId::T4;
        case TableId::Table5: return FileId::T5;
        case TableId::Table6: return FileId::T6;
        case TableId::Table7: return FileId::T7;

        default:
            ASSERT( 0 );
            break;
    }
    
    ASSERT( 0 );
    return FileId::None;
}

//-----------------------------------------------------------
inline FileId TableIdToMapFileId( const TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::MAP2;
        case TableId::Table3: return FileId::MAP3;
        case TableId::Table4: return FileId::MAP4;
        case TableId::Table5: return FileId::MAP5;
        case TableId::Table6: return FileId::MAP6;
        case TableId::Table7: return FileId::MAP7;

        default:
            ASSERT( 0 );
            break;
    }
    
    ASSERT( 0 );
    return FileId::None;
}

//-----------------------------------------------------------
inline FileId TableIdToMarkedEntriesFileId( const TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::MARKED_ENTRIES_2;
        case TableId::Table3: return FileId::MARKED_ENTRIES_3;
        case TableId::Table4: return FileId::MARKED_ENTRIES_4;
        case TableId::Table5: return FileId::MARKED_ENTRIES_5;
        case TableId::Table6: return FileId::MARKED_ENTRIES_6;

        default:
            ASSERT( 0 );
            break;
    }
    
    ASSERT( 0 );
    return FileId::None;
}
