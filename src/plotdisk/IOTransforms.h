#pragma once

class IIOTransform
{
public:
    struct TransformData
    {
        void*   buffer;
        uint32  numBuckets;
        uint32* bucketSizes;

        // void*       output;
        // size_t      inputSize;
        // void*       userData;
    };

public:
    inline virtual ~IIOTransform() {}
    // virtual void ReadTransform( TransformData& data ) = 0;
    // virtual void WriteTransform( TransformData& data ) = 0;

    inline virtual void Read( TransformData& data ) {}
    inline virtual void Write( TransformData& data ) {}
};
