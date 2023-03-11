#pragma once
#include "util/Util.h"

// Chiapos-compatible bitreader
class CPBitReader
{
public:

    //-----------------------------------------------------------
    inline CPBitReader() 
        : _fields  ( nullptr )
        , _sizeBits( 0 )
        , _position( 0 )
    {}

    // Expects bytesBE to be 64-bit fields in BigEndian format.
    //-----------------------------------------------------------
    inline CPBitReader( const byte* bytesBE, size_t sizeBits, uint64 bitOffset = 0 )
        : _fields  ( bytesBE      )
        , _sizeBits( sizeBits  )
        , _position( bitOffset )
    {}

    //-----------------------------------------------------------
    inline void Seek( uint64 position )
    {
        ASSERT( position <= _sizeBits );
        _position = position;
    }

    //-----------------------------------------------------------
    inline uint64 Read64( const uint32 bitCount )
    {
        ASSERT( _position + bitCount <= _sizeBits );
        const uint64 value = Read64( bitCount, _fields, _position, _sizeBits );
        _position += bitCount;
        return value;
    }

    //-----------------------------------------------------------
    inline uint64 Read64At( const uint64 position, const uint32 bitCount )
    {
        ASSERT( position + bitCount <= _sizeBits );
        const uint64 value = Read64( bitCount, _fields, position, _sizeBits );
        return value;
    }

    //-----------------------------------------------------------
    inline uint128 Read128Aligned( const uint32 bitCount )
    {
        const uint128 value = Read128Aligned( bitCount, (uint64*)_fields, _position, _sizeBits );
        _position += bitCount;
        return value;
    }

    //-----------------------------------------------------------
    inline uint64 Read64Aligned( const uint32 bitCount )
    {
        ASSERT( _position + bitCount <= _sizeBits );
        const uint64 value = Read64Aligned( bitCount, _fields, _position, _sizeBits );
        _position += bitCount;
        return value;
    }

     //-----------------------------------------------------------
    inline bool Read64Safe( const uint32 bitCount, uint64& outValue )
    {
        if( _position + bitCount > _sizeBits || bitCount > 64 )
            return false;
        
        outValue = Read64( bitCount, _fields, _position, _sizeBits );
        _position += bitCount;
        return true;
    }

    //-----------------------------------------------------------
    static inline uint64 Read64( const uint32 bitCount, const byte* fields, const uint64 position, const size_t sizeBits )
    {
        return _Read64<true>( bitCount, fields, position, sizeBits );
    }

    //-----------------------------------------------------------
    // Only use if you an guarantee that bytesBE are 
    // aligned to 8-byte boundaries, and that the last field
    // round-up to a uint64,
    //-----------------------------------------------------------
    static inline uint64 Read64Aligned( const uint32 bitCount, const byte* fields, const uint64 position, const size_t sizeBits )
    {
        ASSERT( ((uintptr_t)((uint64*)fields + (position >> 6)) & 7) == 0 )
        return _Read64<false>( bitCount, fields, position, sizeBits );
    }

    // Read 128 bits or less
    //-----------------------------------------------------------
    inline uint128 Read128Aligned( const uint32 bitCount, const uint64* fields, const uint64 position, const size_t sizeBits  )
    {
        ASSERT( ((uintptr_t)(fields + (position >> 6)) & 7) == 0 )
        ASSERT( bitCount <= 128 );
        ASSERT( position + bitCount <= sizeBits && position + bitCount > position );

        const uint64 fieldIndex    = _position >> 6; // _position / 64
        const uint32 bitsAvailable = (uint32)( ( (fieldIndex + 1) * 64 ) - _position );
        const uint32 shift         = std::max( bitCount, bitsAvailable ) - bitCount;

        uint128 value = Swap64( fields[fieldIndex] ) >> shift;

        if( bitsAvailable < bitCount )
        {
            // Have to read one more field
            const uint32 bitsNeeded = bitCount - bitsAvailable;

            if( bitsNeeded > 64 )
            {
                // Need data from 2 more fields
                const uint32 lastFieldBitsNeeded = bitsNeeded - 64;
                value = ( value << bitsNeeded ) | ( Swap64( fields[fieldIndex+1] ) << lastFieldBitsNeeded );
                value |= Swap64( fields[fieldIndex+2] ) >> ( 64 - lastFieldBitsNeeded );
            }
            else
            {
                // Only need data from 1 more field
                value = ( value << bitsNeeded ) | ( Swap64( fields[fieldIndex+1] ) >> ( 64 - bitsNeeded ) );
            }
        }

        // Mask-out part of the fields we don't need
        value &= ( ( ( (uint128)0xFFFFFFFFFFFFFFFFull << 64 ) | 0xFFFFFFFFFFFFFFFFull ) >> ( 128 - bitCount ) );

        _position += bitCount;
        return value;
    }

private:
    //-----------------------------------------------------------
    template<bool CheckAlignment>
    static inline uint64 _Read64( const uint32 bitCount, const byte* fields, const uint64 position, const size_t sizeBits )
    {
        const uint64 fieldIndex    = position >> 6;                          // position / 64
        const uint32 fieldBitIdx   = (uint32)( position - fieldIndex * 64 ); // Value start bit position from the left (MSb) in the field itself
        const uint32 bitsAvailable = 64 - fieldBitIdx;

        uint32 shift = 64 - std::min( fieldBitIdx + bitCount, 64u );

        const byte* pField = fields + fieldIndex * 8;
        
        uint64 field;
        uint64 value;

        // Check for aligned pointer
        bool isPtrAligned;
        bool isLastField;

        if constexpr ( CheckAlignment )
        {
            isPtrAligned = ((uintptr_t)pField & 7) == 0; // % 8
            isLastField  = fieldIndex == ( sizeBits >> 6 ) - 1;
            
            if( isPtrAligned && !isLastField )
                field = *((uint64*)pField);
            else if( !isLastField )
                memcpy( &field, pField, sizeof( uint64 ) );
            else
            {
                // No guarantee that the last field is complete, so copy only the bytes we know we have
                const size_t totalBytes     = CDiv( sizeBits, 8 );
                const int32  remainderBytes = (int32)( totalBytes - fieldIndex * 8 );

                field = 0;
                byte* fieldBytes = (byte*)&field;
                for( int32 i = 0; i < remainderBytes; i++ )
                    fieldBytes[i] = pField[i];
            }

            field = Swap64( field );
        }
        else
            field = Swap64( *((uint64*)pField) );

        value = field >> shift;

        // Need to read 1 more field?
        if( bitsAvailable < bitCount )
        {
            if constexpr ( CheckAlignment )
            {
                pField += 8;
                    
                if( isPtrAligned && !isLastField )    
                    field = *((uint64*)pField);
                else if( !isLastField )
                    memcpy( &field, pField, sizeof( uint64 ) );
                else
                {
                    const size_t totalBytes     = CDiv( sizeBits, 8 );
                    const int32  remainderBytes = (int32)( totalBytes - fieldIndex * 8 );

                    field = 0;
                    byte* fieldBytes = (byte*)&field;
                    for( int32 i = 0; i < remainderBytes; i++ )
                        fieldBytes[i] = pField[i];
                }

                field = Swap64( field );
            }
            else
                field = Swap64( ((uint64*)pField)[1] );

            const uint32 remainder = bitCount - bitsAvailable;
            shift = 64 - remainder;
            
            value = ( value << remainder ) | ( field >> shift );
        }

        return value & ( 0xFFFFFFFFFFFFFFFFull >> ( 64 - bitCount ) );
    }

private:
    const byte* _fields  ;  // Our fields buffer. Expected to be 64-bit field sizes in BigEndian format
    size_t      _sizeBits;  // Size of the how much data we currently have in bits
    uint64      _position;  // Read poisition in bits
};

class BitReader
{
public:

    //-----------------------------------------------------------
    inline BitReader() 
        : _fields  ( nullptr )
        , _sizeBits( 0 )
        , _position( 0 )
    {}

    //-----------------------------------------------------------
    inline BitReader( const uint64* bits, size_t sizeBits, uint64 bitOffset = 0 )
        : _fields  ( bits      )
        , _sizeBits( sizeBits  )
        , _position( bitOffset )
    {
        // #TODO: Fix plot tool now that we're not initializing BE bytes

        // ASSERT( sizeBits / 64 * 64 == sizeBits );
        // ASSERT( sizeBits <= (size_t)std::numeric_limits<ssize_t>::max() );

        // const size_t fieldCount = sizeBits / 64;

        // for( uint64 i = 0; i < fieldCount; i++ )
        //     bytesBE[i] = Swap64( bytesBE[i] );
        
        // // Also swap any remainder bytes
        // const size_t bitsRemainder = sizeBits - fieldCount * 64;
        // if( bitsRemainder )
        //     bytesBE[fieldCount] = Swap64( bytesBE[fieldCount] << ( 64 - bitsRemainder ) );
    }

    // Read 64 bits or less
    //-----------------------------------------------------------
    inline uint64 ReadBits64( const uint32 bitCount )
    {
        // ASSERT( _position + bitCount <= _sizeBits ); // #TODO: Enable this

        const auto value = ReadBits64( bitCount, _fields, _position );

        _position += bitCount;
        return value;
    }

    // Read 128 bits or less
    //-----------------------------------------------------------
    inline uint128 ReadBits128( const uint32 bitCount )
    {
        ASSERT( 0 );    // #TODO: Update to support changed encoding
        ASSERT( bitCount <= 128 );
        ASSERT( _position + bitCount <= _sizeBits );

        const uint64 fieldIndex    = _position >> 6; // _position / 64
        const uint32 bitsAvailable = (uint32)( ( (fieldIndex + 1) * 64 ) - _position );
        const uint32 shift         = std::max( bitCount, bitsAvailable ) - bitCount;

        uint128 value = _fields[fieldIndex] >> shift;

        if( bitsAvailable < bitCount )
        {
            // Have to read one more field
            const uint32 bitsNeeded = bitCount - bitsAvailable;

            if( bitsNeeded > 64 )
            {
                // Need data from 2 more fields
                const uint32 lastFieldBitsNeeded = bitsNeeded - 64;
                value = ( value << bitsNeeded ) | ( _fields[fieldIndex+1] << lastFieldBitsNeeded );
                value |= _fields[fieldIndex+2] >> ( 64 - lastFieldBitsNeeded );
            }
            else
            {
                // Only need data from 1 more field
                value = ( value << bitsNeeded ) | ( _fields[fieldIndex+1] >> ( 64 - bitsNeeded ) );
            }
        }

        // Mask-out part of the fields we don't need
        value &= ( ( ( (uint128)0xFFFFFFFFFFFFFFFFull << 64 ) | 0xFFFFFFFFFFFFFFFFull ) >> ( 128 - bitCount ) );

        _position += bitCount;
        return value;
    }

    //-----------------------------------------------------------
    inline static uint64 ReadBits64( const uint32 bitCount, const uint64* fields, const uint64 position )
    {
        ASSERT( bitCount <= 64 );

        const uint64 fieldIndex    = position >> 6; // position / 64
        const uint32 fieldBits     = (uint32)( position - fieldIndex * 64 ); // Bit offset in the field itself
        const uint32 bitsAvailable = 64 - fieldBits;

        uint64 value = fields[fieldIndex] >> fieldBits;

        if( bitsAvailable < bitCount )
        {
            // Have to read one more field
            const uint64 mask = ( 1ull << bitsAvailable ) - 1 ;
            value = ( value & mask ) | ( fields[fieldIndex+1] << bitsAvailable );
        }

        // Mask-out part of the value we don't need
        return value & ( 0xFFFFFFFFFFFFFFFFull >> (64 - bitCount) );
    }

    // Read in BigEndian mode for compatibility w/ chiapos
    //-----------------------------------------------------------
    inline static uint64 ReadBits64BE( const uint32 bitCount, const uint64* fields, const uint64 position )
    {
        ASSERT( bitCount <= 64 );

        const uint64 fieldIndex    = position >> 6; // position / 64
        const uint32 fieldBits     = (uint32)( position - fieldIndex * 64 );  // Bit offset in the field itself
        const uint32 bitsAvailable = 64 - fieldBits;

        uint64 value = Swap64( fields[fieldIndex] ) >> fieldBits;

        if( bitsAvailable < bitCount )
        {
            // Have to read one more field
            const uint64 mask = ( 1ull << bitsAvailable ) - 1 ;
            value = ( value & mask ) | ( fields[fieldIndex+1] << bitsAvailable );
        }

        // Mask-out part of the value we don't need
        return value & ( 0xFFFFFFFFFFFFFFFFull >> (64 - bitCount) );
    }

    //-----------------------------------------------------------
    inline void Bump( uint32 bitCount )
    {
        ASSERT( _position + bitCount > _position );
        _position += bitCount;
    }

    //-----------------------------------------------------------
    inline void Seek( uint64 position )
    {
        ASSERT( position <= _sizeBits );
        _position = position;
    }

    //-----------------------------------------------------------
    inline void SeekRelative( int64 offset )
    {
        if( offset < 0 )
        {
            const auto subtrahend = (uint64)std::abs( offset );
            ASSERT( subtrahend <= _position );

            _position -= subtrahend;
        }
        else
        {
            Seek( _position + (uint64)offset );
        }
    }

private:
    const uint64* _fields  ;  // Our fields buffer
    size_t        _sizeBits;  // Size of the how much data we currently have in bits
    uint64        _position;  // Read poisition in bits
};


class BitWriter
{
public:

    //-----------------------------------------------------------
    inline BitWriter( uint64* fields, size_t capacity, uint64 startBitOffset = 0 )
        : _fields  ( fields   )
        , _capacity( capacity )
        , _position( startBitOffset )
    {
        ASSERT( _position <= _capacity );
    }

    //-----------------------------------------------------------
    inline BitWriter( const BitWriter& src )
        : _fields  ( src._fields   )
        , _capacity( src._capacity )
        , _position( src._position )
    {}

    //-----------------------------------------------------------
    inline BitWriter( BitWriter&& src )
        : _fields  ( src._fields   )
        , _capacity( src._capacity )
        , _position( src._position )
    {
        src._fields   = nullptr;
        src._capacity = 0;
        src._position = 0;
    }

    //-----------------------------------------------------------
    inline BitWriter()
        : _fields  ( nullptr )
        , _capacity( 0 )
        , _position( 0 )
    {}

     //-----------------------------------------------------------
    inline BitWriter& operator=( const BitWriter& other ) noexcept
    {
        this->_fields   = other._fields  ;
        this->_capacity = other._capacity;
        this->_position = other._position;
        
        return *this;
    }

    //-----------------------------------------------------------
    inline BitWriter& operator=( BitWriter&& other ) noexcept
    {
        this->_fields   = other._fields  ;
        this->_capacity = other._capacity;
        this->_position = other._position;
        
        other._fields   = nullptr;
        other._capacity = 0;
        other._position = 0;
        
        return *this;
    }

    //-----------------------------------------------------------
    inline void Write( const uint64 value, const uint32 bitCount )
    {
        ASSERT( _capacity - _position >= bitCount );
        WriteBits64( _fields, _position, value, bitCount );
        _position += bitCount;
    }

    //-----------------------------------------------------------
    inline void Write64BE( const uint64 value, const uint32 bitCount )
    {
        ASSERT( _capacity - _position >= bitCount );
        WriteBits64BE( _fields, _position, value, bitCount );
        _position += bitCount;
    }

    //-----------------------------------------------------------
    // dstOffset: Offset in bits as to where to start writing in fields
    //-----------------------------------------------------------
    inline static void WriteBits64( uint64* fields, const uint64 dstOffset, const uint64 value, const uint32 bitCount )
    {
        ASSERT( bitCount <= 64 );
        ASSERT( dstOffset + bitCount > dstOffset );

        const uint64 fieldIndex = dstOffset >> 6;
        const uint32 fieldBits  = (uint32)( dstOffset - fieldIndex * 64 );
        const uint32 bitsFree   = 64 - fieldBits;

        // Determine how many bits to write to this current field
        const uint32 bitWrite  = bitCount < bitsFree ? bitCount : bitsFree;// std::min( bitCount, bitsFree );// & 63; // Mod 64
        // const uint32 shift     = bitWrite & 63; // Mod 64

        // Clear out our new value region
        // uint64 mask = ( ( 1ull << (64 - bitWrite) ) - 1 ) << shift;
        uint64 mask = ( 0xFFFFFFFFFFFFFFFFull >> ( 64 - bitWrite ) ) << fieldBits;

        fields[fieldIndex] = ( fields[fieldIndex] & (~mask) ) | ( ( value << fieldBits ) & mask );

        // If we still have bits to write, then write in the next field
        if( bitWrite < bitCount )
        {
            const uint32 remainder = bitCount - bitWrite;
                         mask      = 0xFFFFFFFFFFFFFFFFull >> ( 64 - remainder );
            fields[fieldIndex+1]   = ( fields[fieldIndex+1] & (~mask) ) | ( ( value >> bitWrite ) & mask );
        }
    }

    //-----------------------------------------------------------
    inline uint64* Fields() const { return _fields; }

    //-----------------------------------------------------------
    inline uint64  Position() const { return _position; }

    //-----------------------------------------------------------
    inline uint64  Capacity() const { return _capacity; }

    //-----------------------------------------------------------
    inline void Bump( uint64 bitCount )
    {
        ASSERT( _position + bitCount >= _position );
        _position += bitCount;
    }

    //-----------------------------------------------------------
    // Chiapos compatible method. Disabling this here for now since this makes it hard to use in a multi-threaded manner
    // dstOffset: Offset in bits as to where to start writing in fields
    //-----------------------------------------------------------
    inline static void WriteBits64BE( uint64* fields, const uint64 dstOffset, const uint64 value, const uint32 bitCount )
    {
        ASSERT( bitCount <= 64 );
        ASSERT( dstOffset + bitCount > dstOffset );

        const uint64 fieldIndex = dstOffset >> 6;
        const uint32 bitsFree   = (uint32)( ( ( fieldIndex + 1 ) * 64 ) - dstOffset );

        // Determine how many bits to write to this current field
        const uint32 bitWrite  = std::min( bitCount, bitsFree );// & 63; // Mod 64
        const uint32 shift     = bitWrite & 63; // Mod 64

        // Clear out our new value region
        // uint64 mask = ( ( 1ull << (64 - bitWrite) ) - 1 ) << shift;
        uint64 mask = 0xFFFFFFFFFFFFFFFFull >> ( 64 - bitWrite );

        fields[fieldIndex] = ( ( fields[fieldIndex] << shift ) & (~mask) ) | ( ( value >> ( bitCount - bitWrite ) & mask ) );

        // If we still have bits to write, then write in the next field
        if( bitWrite < bitCount )
        {
            const uint32 remainder = bitCount - shift;
                         mask      = 0xFFFFFFFFFFFFFFFFull >> ( 64 - remainder );
            fields[fieldIndex+1]   = value & mask;
        }
    }

private:
    uint64* _fields  ;  // Our fields buffer
    size_t  _capacity;  // How many bits can we hold
    uint64  _position;  // Write poisition in bits
};

template<size_t BitSize>
class Bits
{
public:
    //-----------------------------------------------------------
    inline Bits()
    {
        if constexpr ( BitSize > 0 )
            _fields[0] = 0;
    }

    //-----------------------------------------------------------
    inline Bits( uint64 value, uint32 sizeBits )
    {
        static_assert( BitSize > 0, "Attempting to write to a zero-sized Bits." );
        
        _fields[0] = 0;
        Write( value, sizeBits );
    }

    //-----------------------------------------------------------
    inline Bits( const byte* bytesBE, uint32 sizeBits, uint64 bitOffset )
    {
        ASSERT( sizeBits <= BitSize );

        const uint64 startField = bitOffset >> 6; // div 64
        const uint64 endField   = ( startField * 64 + sizeBits ) >> 6; // div 64
        const uint64 fieldCount = ( endField - startField ) + 1;

        bytesBE += startField * sizeof( uint64 );

        // Make the bit offset local to the starting field
        bitOffset -= startField * 64;

        _fields[0] = 0;
        _length    = 0;

        // Write partial initial field
        // #TODO: Use the field directly when have 64-bit aligned memory
        {
            uint64 field;
            memcpy( &field, bytesBE, sizeof( uint64 ) );
            field = Swap64( field );

            const uint32 firstFieldAvail = (uint32)( 64 - bitOffset );
            const uint32 firstFieldBits  = std::min( firstFieldAvail, sizeBits );
            const uint64 mask            = ( 0xFFFFFFFFFFFFFFFFull >> ( 64 - firstFieldBits ) );

            field = ( field >> ( firstFieldAvail - firstFieldBits ) & mask );   // Need to mask-out because the field may have offset and less bits than 64
                                                                                // so we don't want the high-order bits to make it into the stored field
            Write( field, firstFieldBits );
            
            bytesBE += sizeof( uint64 );
        }

        // Write any full fields
        const int64 fullFieldCount = (uint64)std::max( 0ll, (int64)fieldCount - 2 );
        for( int64 i = 0; i < fullFieldCount; i++ )
        {
            uint64 field;
            memcpy( &field, bytesBE, sizeof( uint64 ) );
            field = Swap64( field );

            Write( field, 64 );

            bytesBE += sizeof( uint64 );
        }

        // Write any partial final field
        if( fieldCount > 1 )
        {
            const uint32 lastFieldBits = (uint32)( (sizeBits + bitOffset) - (fieldCount-1) * 64 );

            uint64 field;
            memcpy( &field, bytesBE, sizeof( uint64 ) );
            field = Swap64( field );
            field >>= ( 64 - lastFieldBits );

            _fields[fieldCount-1] = 0;
            Write( field, lastFieldBits );
        }
    }

    //-----------------------------------------------------------
    template<size_t TBitsSize>
    inline Bits( const Bits<TBitsSize>& src )
    {
        static_assert( TBitsSize <= BitSize, "Bit size mismatch." );
        _length = src._length;
        memcpy( _fields, src._fields, CDiv( src._length, 8 ) );
    }

    //-----------------------------------------------------------
    inline void Clear()
    {
        _length = 0;

        if constexpr ( BitSize )
            _fields[0] = 0;
    }

    //-----------------------------------------------------------
    inline void Write( uint64 value, uint32 bitCount )
    {
        BitWriter::WriteBits64BE( _fields, _length, value, bitCount );
        _length += bitCount;
    }

    //-----------------------------------------------------------
    template<size_t TBitsSize>
    inline void Write( const Bits<TBitsSize>& other )
    {
        const size_t  inLength = other.Length();
        const uint64* inFields = other.Fields();

        ASSERT( _length + inLength <= BitSize );

        const uint64 fieldCount = inLength >> 6;
        for( uint64 i = 0; i < fieldCount; i++ )
            Write( inFields[i], 64 );

        const uint32 remainderBits = (uint32)( inLength - fieldCount * 64 );
        if( remainderBits )
            Write( inFields[fieldCount], remainderBits );
    }

    //-----------------------------------------------------------
    inline uint64 ReadUInt64( uint32 bitCount )
    {
        ASSERT( bitCount <= BitSize );
        const uint64 value = CPBitReader::Read64( bitCount, _fields, 0 );

        return value;
    }
    
    //-----------------------------------------------------------
    template<size_t TR>
    inline Bits<BitSize>& operator=( const Bits<TR>& r ) const
    {
        static_assert( TR <= BitSize, "Bit size mismatch." );
        
        _length = r._length;
        memcpy( _fields, r._fields, CDiv( r._length, 8 ) );

        return *this;
    }

    //-----------------------------------------------------------
    template<size_t TR>
    inline Bits<BitSize> operator+( const Bits<TR>& r ) const
    {
        Bits<BitSize> result( *this );
        result.Write( r );

        return result;
    }

    //-----------------------------------------------------------
    template<size_t TR>
    inline Bits<BitSize>& operator+=( const Bits<TR>& r )
    {
        this->Write( r );
        return *this;;
    }

    //-----------------------------------------------------------
    inline void ToBytes( byte* bytes )
    {
        const uint64 fieldCount = _length >> 6;
        for( uint64 i = 0; i < fieldCount; i++ )
        {
            const uint64 field = Swap64( _fields[i] );
            memcpy( bytes, &field, sizeof( field ) );
            bytes += sizeof( field );
        }
        
        const uint32 remainderBits = (uint32)( _length - fieldCount * 64 );
        if( remainderBits )
        {
            const uint64 field = Swap64( _fields[fieldCount] << ( 64 - remainderBits ) );
            const size_t size  = CDiv( remainderBits, 8 );

            memcpy( bytes, &field, size );
        }
    }

    // Length in bits
    //-----------------------------------------------------------
    inline size_t Length() const { return _length; }

    inline size_t LengthBytes() const { return CDiv( _length, 8 ); }

    const uint64* Fields() const { return _fields; }

private:
    uint64 _fields[CDiv( CDiv( BitSize, 8 ), 8)];
    uint64 _length = 0; // In bits
};
// class BitView
// {
// public:
//     //-----------------------------------------------------------
//     BitView( uint64* buffer, size_t size, size_t entrySizeBits )
//         : _fields   ( buffer )
//         , _size     ( size   )
//         , _EntrySize( entrySizeBits )
//     {
//         ASSERT( buffer );
//         ASSERT( size   );
//         ASSERT( entrySizeBits );

//         // #TODO: Allow entries greater than 64 bits...
//         static_assert( entrySizeBits > 0 && entrySizeBits < 65, "_EntrySize must be > 0 < 65" );
//     }

//     // Read an entry of _EntrySize bits at index.
//     // Unsafe, this will not check bounds on release mode.
//     //-----------------------------------------------------------
//     uint64 Get( uint64 index )
//     {

//     }

//     // Write an entry of _EntrySize bits at index.
//     // Unsafe, this will not check bounds on release mode.
//     //-----------------------------------------------------------
//     inline void Set( uint64 index, uint64 value )
//     {

//     }

//     uint64 operator[] ( uint64 index );

// private:
//     uint64 GetWordIndex( uint64 entryIndex );

// private:
//     uint64* _fields;
//     size_t  _EntrySize; // Size of a single entry, in bits
//     size_t  _size ;     // Maximum size as uint64 entries. 
//                         // That is, how many uint64 entries can we hold.
// };


/// <summary>
/// Static-typed version of BitView
/// _EntrySize must be in bits.
/// </summary>
template<size_t _EntrySize>
class BitViewT
{
public:
    BitViewT( uint64* buffer, size_t64 size );

    // Read an entry of _EntrySize bits at index.
    // Unsafe, this will not check bounds on release mode.
    uint64 Get( uint64 index );

    // Write an entry of _EntrySize bits at index.
    // Unsafe, this will not check bounds on release mode.
    void   Set( uint64 index, uint64 value );

    uint64 operator[] ( uint64 index );

private:
    uint64 GetWordIndex( uint64 entryIndex );

private:
    uint64*     _fields;
    size_t64    _size ;     // Maximum size as uint64 entries. 
                            // That is, how many uint64 entries can we hold.
};

// Bits occupied by the entry size
#define _EntryBits      ( ( 1ull << _EntrySize ) -1 )

// Bits not occupied by the entry size
#define _InvEntryBits   ( ~_EntryBits )
#define _BitInv( x ) (~(x))

//-----------------------------------------------------------
template<size_t _EntrySize>
inline BitViewT<_EntrySize>::BitViewT( uint64* buffer, size_t64 size )
    : _fields( buffer )
    , _size  ( size   )
{
    ASSERT( buffer );
    ASSERT( size   );

    static_assert( _EntrySize > 0 && _EntrySize < 65, "_EntrySize must be > 0 < 65" );
}


//-----------------------------------------------------------
template<size_t _EntrySize>
inline uint64 BitViewT<_EntrySize>::Get( uint64 index )
{
    ASSERT( index * _EntrySize + _EntrySize <= _size * 64 );

    const uint64 wordIdx = GetWordIndex( index );

    // 64 because our 'word' size is 64-bits/8 bytes
    const uint64 bitStart = ( index * _EntrySize ) - ( wordIdx * 64 );
    const uint64 bitCount = 64 - bitStart;    // Bits available in the WORD
    
    uint64 value = _fields[wordIdx] >> bitStart;

    // Need to read next field?
    if( bitCount < _EntrySize )
        value |= _fields[wordIdx+1] << bitCount;
    
    return value & _EntryBits;
}

//-----------------------------------------------------------
template<size_t _EntrySize>
inline void BitViewT<_EntrySize>::Set( uint64 index, uint64 value )
{
    ASSERT( index * _EntrySize + _EntrySize <= _size * 64 );

    const uint64 wordIdx = GetWordIndex( index );
    
    // 64 because our 'word' size is 64-bits/8 bytes
    const uint64 bitStart = ( index * _EntrySize ) - ( wordIdx * 64 );
    const uint64 bitCount = 64 - bitStart;    // Bits available in the WORD

    // Mask with only the bits of our interest 
//     value &= _EntryBits;

    // Mask out where the value will placed in the current WORD
    uint64 idxValue  = _fields[wordIdx] & _BitInv( _EntryBits << bitStart );
    _fields[wordIdx] = idxValue | ( (value & _EntryBits) << bitStart );

    if( bitCount < _EntrySize )
    {
        // Fit it into 2 consecutive WORDs
        idxValue           = _fields[wordIdx+1] & ( _InvEntryBits >> bitCount );
        _fields[wordIdx+1] = idxValue | ( ( value & _EntryBits ) >> bitCount );
    }
}

//-----------------------------------------------------------
template<size_t _EntrySize>
inline uint64 BitViewT<_EntrySize>::operator[]( uint64 index )
{
    return Get( index );
}


//-----------------------------------------------------------
template<size_t _EntrySize>
inline uint64 BitViewT<_EntrySize>::GetWordIndex( uint64 entryIndex )
{
    // 64 because our 'word' size is 64-bits/8 bytes
    return (entryIndex * _EntrySize) / 64;
}

#undef _EntryBits
#undef _InvEntryBits
#undef _BitInv
