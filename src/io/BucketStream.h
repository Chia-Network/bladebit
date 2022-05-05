#include "IStream.h"

class BucketStream : public IStream
{
public:
    enum Mode
    {
        Sequential = 0,
        Interleaved
    };

    BucketStream( IStream& baseStream, const size_t sliceSize, const uint32 numBuckets );

    virtual ~BucketStream();

    ssize_t Read( void* buffer, size_t size ) override;

    ssize_t Write( const void* buffer, size_t size ) override;

    void WriteBucketSlices( const void* slices, const uint32* sliceSizes );

    void ReadBucket( const size_t size, void* readBuffer );

    bool Seek( int64 offset, SeekOrigin origin ) override;

    bool Flush() override;

    size_t BlockSize() const override;

    ssize_t Size() override;

    bool Truncate( const ssize_t length ) override;

    int GetError() override;

    inline Mode GetWriteMode() const { return _writeMode; }
    inline Mode GetReadMode() const { return (Mode)(((uint32)_writeMode + 1) & 1); }

private:
    inline void SwitchMode()
    {
        _writeMode = (Mode)(((uint32)_writeMode + 1) & 1); // (mode + 1) % 2
    }

    struct Slice
    {
        uint32 position;
        uint32 size;
    };
private:
    IStream&      _baseStream;          // Backing stream where we actually write data
    Span<size_t*> _sequentialSlices;    // Info about each bucket slice
    Span<size_t*> _interleavedSlices;   // Info about each bucket slice
    size_t        _sliceCapacity;       // Maximum size of a single bucket slice
    size_t        _bucketCapacity;      // Maximum capacity of a single bucket
    uint32        _numBuckets;

    // Keep track of current bucket or slice. These are
    // opposite to each other depending on the current mode being used.
    // In sequential mode you write in slices across all buckets, and read in buckets.
    // In interleaved mode you write in buckets and read in slices across all buckets.
    union {
        uint16 _writeSlice = 0;
        uint16 _writeBucket;
    };

    union {
        uint16 _readSlice = 0;
        uint16 _readBucket;
    };

    Mode _writeMode = Sequential; // Current write mode. The read mode is the opposite
};

