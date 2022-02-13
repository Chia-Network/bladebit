#pragma once

// This is a utility class for reading entries encoded in bits, whic
// are aligned to uint64 boundaries and written as big-endian.
// This helps us read values by firest transforming the data back to
// little endian and making it easy to read variable-sized bit values.
// The format of the bits is such that the entries are shifted all the way to
// the left of a field. For example, an initial 32-bit value will be found in the MSbits of the first field.
class BitReader
{
public:

    //-----------------------------------------------------------
    inline BitReader() 
        : _fields  ( nullptr )
        , _sizeBits( 0 )
        , _position( 0 )
    {}

    // bytesBE must be rounded-up to 64-bit boundaries
    // This expects the bytes to be encoded as 64-bit big-endian fields.
    // The last bytes will be shifted to the right then swaped as 64-bits as well.
    //-----------------------------------------------------------
    inline BitReader( uint64* bytesBE, size_t sizeBits )
        : _fields  ( bytesBE  )
        , _sizeBits( sizeBits )
        , _position( 0 )
    {
        ASSERT( sizeBits / 64 * 64 == sizeBits );
        ASSERT( sizeBits <= (size_t)std::numeric_limits<ssize_t>::max() );

        const size_t fieldCount = sizeBits / 64;

        for( uint64 i = 0; i < fieldCount; i++ )
            bytesBE[i] = Swap64( bytesBE[i] );
        
        // Also swap any remainder bytes
        const size_t bitsRemainder = sizeBits - fieldCount * 64;
        if( bitsRemainder )
            bytesBE[fieldCount] = Swap64( bytesBE[fieldCount] << ( 64 - bitsRemainder ) );
    }

    // Read 64 bits or less
    //-----------------------------------------------------------
    inline uint64 ReadBits64( const uint32 bitCount )
    {
        // #TODO: Use shared static version
        ASSERT( bitCount <= 64 );
        ASSERT( _position + bitCount <= _sizeBits );

        const uint64 fieldIndex    = _position >> 6; // _position / 64
        const uint32 bitsAvailable = ( ( fieldIndex + 1 ) * 64 ) - _position;
        const uint32 shift         = std::max( bitCount, bitsAvailable ) - bitCount;

        uint64 value = _fields[fieldIndex] >> shift;

        if( bitsAvailable < bitCount )
        {
            // Have to read one more field
            const uint32 bitsNeeded = bitCount - bitsAvailable;
            value = ( value << bitsNeeded ) | ( _fields[fieldIndex+1] >> ( 64 - bitsNeeded ) );
        }

        // Mask-out part of the fields we don't need
        value &= ( 0xFFFFFFFFFFFFFFFFull >> (64 - bitCount ) );

        _position += bitCount;
        return value;
    }

    // Read 128 bits or less
    //-----------------------------------------------------------
    inline uint128 ReadBits128( const uint32 bitCount )
    {
        ASSERT( bitCount <= 128 );
        ASSERT( _position + bitCount <= _sizeBits );

        const uint64 fieldIndex    = _position >> 6; // _position / 64
        const uint32 bitsAvailable = ( ( fieldIndex + 1 ) * 64 ) - _position;
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

        const uint64 fieldIndex    = position >> 6; // _position / 64
        const uint32 bitsAvailable = ( ( fieldIndex + 1 ) * 64 ) - position;
        const uint32 shift         = std::max( bitCount, bitsAvailable ) - bitCount;

        uint64 value = fields[fieldIndex] >> shift;

        if( bitsAvailable < bitCount )
        {
            // Have to read one more field
            const uint32 bitsNeeded = bitCount - bitsAvailable;
            value = ( value << bitsNeeded ) | ( fields[fieldIndex+1] >> ( 64 - bitsNeeded ) );
        }

        // Mask-out part of the fields we don't need
        value &= ( 0xFFFFFFFFFFFFFFFFull >> (64 - bitCount ) );

        return value;
    }

    //-----------------------------------------------------------
    void Seek( uint64 position )
    {
        ASSERT( position <= _sizeBits );
        _position = position;
    }

private:
    uint64* _fields  ;  // Our fields buffer
    size_t  _sizeBits;  // Size of the how much data we currently have in bits
    uint64  _position;  // Read poisition in bits
};


template<size_t BitSize>
class Bits
{
public:
    //-----------------------------------------------------------
    inline Bits()
    {
        if constexpr ( BitSize )
            _fields[0] = 0;
    }

    //-----------------------------------------------------------
    inline Bits( uint64 value, uint32 sizeBits )
    {
        static_assert( BitSize, "Attempting to write to a zero-sized Bits." );
        
        _fields[0] = 0;
        Write( value, sizeBits );
    }

    //-----------------------------------------------------------
    inline Bits( const byte* bytesBE, uint32 sizeBits, uint32 bitOffset )
    {
        ASSERT( sizeBits <= BitSize );

        const uint64 startField = bitOffset >> 6; // div 64
        const uint64 endField   = ( startField * 64 + sizeBits ) >> 6; // div 64
        const uint64 fieldCount = ( endField - startField ) + 1;

        bytesBE += startField * sizeof( uint64 );

        // Make the bit offset local to the starting field
        bitOffset -= startField * 64;


        _length = 0;

        // Write partial initial field
        // #TODO: Use the field directly when have 64-bit aligned memory

        {
            uint64 field;
            memcpy( &field, bytesBE, sizeof( uint64 ) );
            field = Swap64( field );

            const uint32 firstFieldAvail = 64 - bitOffset;
            const uint32 firstFieldBits  = std::min( firstFieldAvail, sizeBits );

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
        ASSERT( bitCount <= 64 );
        ASSERT( _length + bitCount <= BitSize );

        const uint64 fieldIndex = _length >> 6;
        const uint32 bitsFree   = ( ( fieldIndex + 1 ) * 64 ) - _length;

        // Determine how many bits to write to this current field
        const uint32 bitWrite  = std::min( bitCount, bitsFree ) & 63; // Mod 64
        const uint32 shift     = bitWrite & 63; // Mod 64

        // Clear out our ne value region
        uint64 mask = ( ( 1ull << (64 - bitWrite) ) - 1 ) << shift;

        _fields[fieldIndex] = ( ( _fields[fieldIndex] << shift ) & mask ) | ( value >> ( bitCount - bitWrite ) );

        // If we still have bits to write, then write in the next field
        if( bitWrite < bitCount )
        {
            const uint32 remainder = bitCount - shift;
                         mask      = 0xFFFFFFFFFFFFFFFFull >> ( 64 - remainder );
            _fields[fieldIndex+1]  = value & mask;
        }

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
        const uint64 value = BitReader::ReadBits64( bitCount, _fields, 0 );
        
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
