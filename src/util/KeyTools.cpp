#include "KeyTools.h"
#include "util/Util.h"
#include "util/Log.h"



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
bls::PrivateKey KeyTools::MasterSkToLocalSK( bls::PrivateKey& sk )
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

    return ssk;
}

//-----------------------------------------------------------
void KeyTools::PrintPK( const bls::G1Element&  key )
{
    std::vector<uint8_t> bytes = key.Serialize();
    Log::Line( "%s", HexToString( (byte*)bytes.data(), bytes.size() ).c_str() );
}

//-----------------------------------------------------------
void KeyTools::PrintSK( const bls::PrivateKey& key )
{
    std::vector<uint8_t> bytes = key.Serialize();
    Log::Line( "%s", HexToString( (byte*)bytes.data(), bytes.size() ).c_str() );
}


///
/// PuzzleHash
///
//-----------------------------------------------------------
bool PuzzleHash::FromAddress( PuzzleHash& hash, const char address[CHIA_ADDRESS_LENGTH+1] )
{
    ASSERT( address );
    if( !address )
        return false;

    size_t programSize = CHIA_PUZZLE_HASH_SIZE;
    int witver = 0;

    return segwit_addr_decode( &witver, hash.data, &programSize, "xch", address ) == 1;
}

