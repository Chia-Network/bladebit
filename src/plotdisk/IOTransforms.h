#pragma once

class IIOTransform
{
public:
    struct TransformData
    {
        const void* intput;
        void*       output;
        size_t      inputSize;
        uint32      bucketSizes;    // For when we are using buckets
        void*       userData;
    };

public:
    inline virtual ~IIOTransform() {}
    virtual void ReadTransform( TransformData& data ) = 0;
    virtual void WriteTransform( TransformData& data ) = 0;
};
