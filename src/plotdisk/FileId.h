#pragma once


enum class FileId
{
    None = 0,

    // Phase 1 fx, key, metadata
    FX0, FX1,

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
    LP_2,
    LP_3,
    LP_4,
    LP_5,
    LP_6,
    LP_7,

    // Line point key (y sorted order key)
    LP_KEY_2,
    LP_KEY_3,
    LP_KEY_4,
    LP_KEY_5,
    LP_KEY_6,
    LP_KEY_7,

    // Line point map
    LP_MAP_2,
    LP_MAP_3,
    LP_MAP_4,
    LP_MAP_5,
    LP_MAP_6,
    LP_MAP_7,

    C1,
    C2,
    C3,
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


//-----------------------------------------------------------
inline FileId TableIdToLinePointFileId( const TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::LP_2;
        case TableId::Table3: return FileId::LP_3;
        case TableId::Table4: return FileId::LP_4;
        case TableId::Table5: return FileId::LP_5;
        case TableId::Table6: return FileId::LP_6;
        case TableId::Table7: return FileId::LP_7;

        default:
            ASSERT( 0 );
            break;
    }
    
    ASSERT( 0 );
    return FileId::None;
}

//-----------------------------------------------------------
inline FileId TableIdToLinePointKeyFileId( const TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::LP_KEY_2;
        case TableId::Table3: return FileId::LP_KEY_3;
        case TableId::Table4: return FileId::LP_KEY_4;
        case TableId::Table5: return FileId::LP_KEY_5;
        case TableId::Table6: return FileId::LP_KEY_6;
        case TableId::Table7: return FileId::LP_KEY_7;

        default:
            ASSERT( 0 );
            break;
    }
    
    ASSERT( 0 );
    return FileId::None;
}


//-----------------------------------------------------------
inline FileId TableIdToLinePointMapFileId( const TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::LP_MAP_2;
        case TableId::Table3: return FileId::LP_MAP_3;
        case TableId::Table4: return FileId::LP_MAP_4;
        case TableId::Table5: return FileId::LP_MAP_5;
        case TableId::Table6: return FileId::LP_MAP_6;
        case TableId::Table7: return FileId::LP_MAP_7;

        default:
            ASSERT( 0 );
            break;
    }
    
    ASSERT( 0 );
    return FileId::None;
}
