#pragma once


///
/// Unsafe use: It does not do any bounds checking.
///
class BitField
{
public:

    //-----------------------------------------------------------
    inline BitField()
        :_fields( nullptr )
    {}

    //-----------------------------------------------------------
    inline BitField( uint64* buffer )
        : _fields( buffer )
    {
    }

    //-----------------------------------------------------------
    inline bool Get( uint64 index ) const
    {
        const uint64 fieldIdx = index >> 6;  // Divide by 64. Safe to do with power of 2. (shift right == log2(64))
        const uint64 field    = _fields[fieldIdx];

        const uint32 rShift   = (uint32)(index - (fieldIdx << 6));  // Multiply by fieldIdx (shift left == log2(64))
        return (bool)((field >> rShift) & 1u);
    }

    //-----------------------------------------------------------
    inline void Set( uint64 index )
    {
        uint64* fields = _fields;

        const uint64 fieldIdx = index >> 6;
        const uint64 field    = fields[fieldIdx];

        const uint32 lShift   = (uint32)(index - (fieldIdx << 6));

        fields[fieldIdx] = field | (1ull << lShift);
    }

    //-----------------------------------------------------------
    inline void SetBit( uint64 index, const uint64 bit )
    {
        uint64* fields = _fields;

        const uint64 fieldIdx = index >> 6;
        const uint64 field    = fields[fieldIdx];

        const uint32 lShift   = (uint32)(index - (fieldIdx << 6));

        fields[fieldIdx] = field | (bit << lShift);
    }

    //-----------------------------------------------------------
    inline void Clear( uint64 index )
    {
        uint64* fields = _fields;

        const uint64 fieldIdx = index >> 6;
        const uint64 field    = fields[fieldIdx];

        const uint32 lShift   = (uint32)(index - (fieldIdx << 6));

        fields[fieldIdx] = field & (~(1ull << lShift));
    }

    //-----------------------------------------------------------
    inline bool operator[] ( uint64 index ) const { return Get( index ); }
    inline bool operator[] ( int64  index ) const { return Get( (uint64)index ); }
    inline bool operator[] ( size_t index ) const { return Get( (uint64)index ); }

private:
    uint64* _fields;
};
