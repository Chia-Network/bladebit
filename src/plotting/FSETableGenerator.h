#pragma once

typedef unsigned FSE_CTable;
typedef unsigned FSE_DTable;

struct FSETableGenerator
{
    static FSE_CTable* GenCompressionTable( const double rValue, size_t* outTableSize );
    static FSE_DTable* GenDecompressionTable( double rValue, size_t* outTableSize );
};
