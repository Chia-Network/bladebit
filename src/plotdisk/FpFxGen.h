#pragma once
#include "plotting/Tables.h"
#include "threading/ThreadPool.h"
#include "plotdisk/DiskPlotInfo.h"
#include "b3/blake3.h"

template<TableId table>
struct FpYType { using Type = uint64; };

template<>
struct FpYType<TableId::Table7> { using Type = uint32; };


template<TableId table>
struct FpFxGen
{
    using TMetaIn  = typename TableMetaType<table>::MetaIn;
    using TMetaOut = typename TableMetaType<table>::MetaOut;
    using TYOut    = typename FpYType<table>::Type;

    static constexpr size_t MetaInMulti  = TableMetaIn<table>::Multiplier;
    static constexpr size_t MetaOutMulti = TableMetaOut<table>::Multiplier;

    static constexpr uint32 _k = _K;

    //-----------------------------------------------------------
    inline FpFxGen( ThreadPool& pool, const uint32 threadCount )
        : _pool       ( pool )
        , _threadCount( threadCount )
    {}

    //-----------------------------------------------------------
    inline void ComputeFxMT( const int64 entryCount, const Pair* pairs, const uint64* yIn, const TMetaIn* metaIn,
                             TYOut* yOut, TMetaOut* metaOut )
    {
        AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {
            
            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            ComputeFx( count, pairs+offset, yIn, metaIn, yOut+offset, metaOut+offset, self->_jobId );
        });
    }

    //-----------------------------------------------------------
    static inline void ComputeFx( const int64 entryCount, const Pair* pairs, const uint64* yIn, const TMetaIn* metaIn,
                                  TYOut* yOut, TMetaOut* metaOut, const uint32 id )
    {
        static_assert( MetaInMulti != 0, "Invalid metaKMultiplier" );

        // Helper consts
        const uint32 shiftBits   = MetaOutMulti == 0 ? 0 : kExtraBits;  // Table 7 (identified by 0 metadata output) we don't have k + kExtraBits sized y's.
                                                                        // so we need to shift by 32 bits, instead of 26.
        const uint32 ySize       = _k + kExtraBits;         // = 38
        const uint32 yShift      = 64 - (_k + shiftBits);   // = 26 or 32
        const size_t metaSize    = _k * MetaInMulti;
        const size_t metaSizeLR  = metaSize * 2;

        const size_t bufferSize  = CDiv( ySize + metaSizeLR, 8 );

        // Hashing
        uint64 input [5]; // y + L + R
        uint64 output[4]; // blake3 hashed output

        blake3_hasher hasher;

        static_assert( bufferSize <= sizeof( input ), "Invalid fx input buffer size." );

        #if _DEBUG
            uint64 prevY    = yIn[pairs[0].left];
            uint64 prevLeft = 0;
        #endif

        for( int64 i = 0; i < entryCount; i++ )
        {
            const auto& pair = pairs[i];
            const uint32 left  = pair.left;
            const uint32 right = pair.right;
            ASSERT( left < right );

            const uint64 y = yIn[left];

            #if _DEBUG
                ASSERT( y >= prevY );
                ASSERT( left >= prevLeft );
                prevY    = y;
                prevLeft = left;
            #endif

            // Extract metadata
            auto& mOut = metaOut[i];

            if constexpr( MetaInMulti == 1 )
            {
                const uint64 l = metaIn[left ];
                const uint64 r = metaIn[right];

                input[0] = Swap64( y << 26 | l >> 6  );
                input[1] = Swap64( l << 58 | r << 26 );

                // Metadata is just L + R of 8 bytes
                if constexpr( MetaOutMulti == 2 )
                    mOut = l << 32 | r;
            }
            else if constexpr( MetaInMulti == 2 )
            {
                const uint64 l = metaIn[left ];
                const uint64 r = metaIn[right];

                input[0] = Swap64( y << 26 | l >> 38 );
                input[1] = Swap64( l << 26 | r >> 38 );
                input[2] = Swap64( r << 26 );

                // Metadata is just L + R again of 16 bytes
                if constexpr( MetaOutMulti == 4 )
                {
                    mOut.m0 = l;
                    mOut.m1 = r;
                }
            }
            else if constexpr( MetaInMulti == 3 )
            {
                const uint64 l0 = metaIn[left ].m0;
                const uint64 l1 = metaIn[left ].m1 & 0xFFFFFFFF;
                const uint64 r0 = metaIn[right].m0;
                const uint64 r1 = metaIn[right].m1 & 0xFFFFFFFF;
            
                input[0] = Swap64( y  << 26 | l0 >> 38 );
                input[1] = Swap64( l0 << 26 | l1 >> 6  );
                input[2] = Swap64( l1 << 58 | r0 >> 6  );
                input[3] = Swap64( r0 << 58 | r1 << 26 );
            }
            else if constexpr( MetaInMulti == 4 )
            {
                // const uint64 l0 = metaInA[left ];
                // const uint64 l1 = metaInB[left ];
                // const uint64 r0 = metaInA[right];
                // const uint64 r1 = metaInB[right];
                const Meta4 l = metaIn[left];
                const Meta4 r = metaIn[right];

                input[0] = Swap64( y    << 26 | l.m0 >> 38 );
                input[1] = Swap64( l.m0 << 26 | l.m1 >> 38 );
                input[2] = Swap64( l.m1 << 26 | r.m0 >> 38 );
                input[3] = Swap64( r.m0 << 26 | r.m1 >> 38 );
                input[4] = Swap64( r.m1 << 26 );
            }

            // Hash the input
            blake3_hasher_init( &hasher );
            blake3_hasher_update( &hasher, input, bufferSize );
            blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

            const uint64 f = Swap64( *output ) >> yShift;
            yOut[i] = (TYOut)f;

            if constexpr ( MetaOutMulti == 2 && MetaInMulti == 3 )
            {
                const uint64 h0 = Swap64( output[0] );
                const uint64 h1 = Swap64( output[1] );

                mOut = h0 << ySize | h1 >> 26;
            }
            else if constexpr ( MetaOutMulti == 3 )
            {
                const uint64 h0 = Swap64( output[0] );
                const uint64 h1 = Swap64( output[1] );
                const uint64 h2 = Swap64( output[2] );

                mOut.m0 = h0 << ySize | h1 >> 26;
                mOut.m1 = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
            }
            else if constexpr ( MetaOutMulti == 4 && MetaInMulti != 2 ) // In = 2 is calculated above with L + R
            {
                const uint64 h0 = Swap64( output[0] );
                const uint64 h1 = Swap64( output[1] );
                const uint64 h2 = Swap64( output[2] );

                mOut.m0 = h0 << ySize | h1 >> 26;
                mOut.m1 = h1 << 38    | h2 >> 26;
            }
        }
    }


private:
    ThreadPool& _pool;
    uint32      _threadCount;
};