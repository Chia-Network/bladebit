#include "plotdisk/IOTransforms.h"
#include "util/BitView.h"

// class FxTransform : public IIOTransform
// {
// public:
//     //-----------------------------------------------------------
//     inline void ReadTransform( TransformData& data ) override
//     {
//         // uint32*
//         uint32* y;
//         uint64* meta;
//     }

//     //-----------------------------------------------------------
//     inline void WriteTransform( TransformData& data ) override
//     {
        
//     }

// private:

//     //-----------------------------------------------------------
//     template<typename TMeta, typename TAddress>
//     inline void PackFx( void* outBuffer, size_t outBufferSize, 
//                         const uint32* y, const TMeta* meta, const TAddress* key,
//                         const int64 entryCount, const uint32 k, const uint32 keyBits, const uint32 metaBits )
//     {
//         BitWriter writer( (uint64*)outBuffer, outBufferSize * 8 );

//         for( int64 i = 0; i < entryCount; i++ )
//         {
//             writer.Write( y   [i], k        );
//             writer.Write( key [i], keyBits  );
//             writer.Write( meta[i], metaBits );
//         }
//     }
// };
