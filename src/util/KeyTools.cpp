#include "KeyTools.h"
#include "util/Util.h"
#include "util/Log.h"
#include "BLS.h"

//-----------------------------------------------------------
bool KeyTools::HexPKeyToG1Element( const char* hexKey, bls::G1Element& pkey )
{
    ASSERT( hexKey );
    
    size_t length = strlen( hexKey );

    if( length < bls::G1Element::SIZE*2 )
        return false;

    if( hexKey[0] == '0' && hexKey[1] == 'x' )
    {
        hexKey += 2;
        length -= 2;
    }

    if( length != bls::G1Element::SIZE*2 )
        return false;

    byte g1Buffer[bls::G1Element::SIZE];
    HexStrToBytes( hexKey, length, g1Buffer, sizeof( g1Buffer ) );

    bls::Bytes g1Bytes( g1Buffer, sizeof( g1Buffer ) );

    pkey = bls::G1Element::FromBytes( g1Bytes );
    
    return pkey.IsValid();
}

//-----------------------------------------------------------
bls::PrivateKey KeyTools::MasterSkToLocalSK( const bls::PrivateKey& sk )
{
    // #SEE: chia-blockchain: derive-keys.py
    // EIP 2334 bls key derivation
    // https://eips.ethereum.org/EIPS/eip-2334
    // 12381 = bls spec number
    // 8444  = Chia blockchain number and port number
    // 0, 1, 2, 3, 4, 5, 6 farmer, pool, wallet, local, backup key, singleton, pooling authentication key numbers

    const uint32 blsSpecNum         = 12381;
    const uint32 chiaBlockchainPort = 8444; 
    const uint32 localIdx           = 3;

    bls::PrivateKey ssk = bls::AugSchemeMPL().DeriveChildSk( sk, blsSpecNum );
    ssk = bls::AugSchemeMPL().DeriveChildSk( ssk, chiaBlockchainPort );
    ssk = bls::AugSchemeMPL().DeriveChildSk( ssk, localIdx );
    ssk = bls::AugSchemeMPL().DeriveChildSk( ssk, 0        );

    return std::move( ssk );
}

//-----------------------------------------------------------
void KeyTools::PrintPK( const bls::G1Element& key )
{
    std::vector<uint8_t> bytes = key.Serialize();
    Log::Line( "%s", BytesToHexStdString( (byte*)bytes.data(), bytes.size() ).c_str() );
}

//-----------------------------------------------------------
void KeyTools::PrintSK( const bls::PrivateKey& key )
{
    std::vector<uint8_t> bytes = key.Serialize();
    Log::Line( "%s", BytesToHexStdString( (byte*)bytes.data(), bytes.size() ).c_str() );
}


///
/// PuzzleHash
///
//-----------------------------------------------------------
bool PuzzleHash::FromAddress( PuzzleHash& hash, const char address[CHIA_ADDRESS_MAX_LENGTH+1] )
{
    ASSERT( address );
    if( !address )
        return false;

    const size_t addrLen = strlen( address );
    if( addrLen != CHIA_ADDRESS_LENGTH && addrLen != CHIA_TESTNET_ADDRESS_LENGTH )
        return false;

    char hrp [CHIA_ADDRESS_MAX_LENGTH-5] = {};
    byte data[CHIA_ADDRESS_MAX_LENGTH-8];

    size_t dataLen = 0;
    if( bech32_decode( hrp, data, &dataLen, address ) != BECH32_ENCODING_BECH32M )
        return false;

    if( memcmp( "xch",  hrp, sizeof( "xch" ) )  != 0 &&
        memcmp( "txch", hrp, sizeof( "txch" ) ) != 0 )
        return false;

    byte decoded[40];
    size_t decodedSize = 0;
    if( !bech32_convert_bits( decoded, &decodedSize, 8, data, dataLen, 5, 0 ) )
        return false;

    if( decodedSize != CHIA_PUZZLE_HASH_SIZE )
        return false;
    
    memcpy( hash.data, decoded, CHIA_PUZZLE_HASH_SIZE );
    return true;
}

//-----------------------------------------------------------
bool PuzzleHash::FromHex( const char hex[CHIA_PUZZLE_HASH_SIZE*2+1], PuzzleHash& outHash )
{
    const size_t hexLen = strlen( hex );
    if( hexLen != CHIA_PUZZLE_HASH_SIZE*2 )
        return false;

    return HexStrToBytesSafe( hex, hexLen, outHash.data, CHIA_PUZZLE_HASH_SIZE );
}

