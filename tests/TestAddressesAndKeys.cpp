#include "TestUtil.h"
#include "util/KeyTools.h"
#include "TestXchAddressData.h"
#include "util/Util.h"

//-----------------------------------------------------------
TEST_CASE( "mainnet-contract-address-decode", "[unit-core][contract-address]" )
{
    const uint64 numAddresses = (uint64)( sizeof( TestMainnetAddress) / sizeof( AddrHashTestPair ) );
    PuzzleHash generatedHash;

    for( uint64 i = 0; i < numAddresses; i++ )
    {
        const AddrHashTestPair& pair = TestMainnetAddress[i];

        ENSURE( PuzzleHash::FromAddress( generatedHash, pair.address ) );

        PuzzleHash expectedHash;
        ENSURE( PuzzleHash::FromHex( pair.hash, expectedHash ) );

        ENSURE( memcmp( generatedHash.data, expectedHash.data, CHIA_PUZZLE_HASH_SIZE ) == 0 );
    }
}

//-----------------------------------------------------------
TEST_CASE( "testnet-contract-address-decode", "[unit-core][contract-address]" )
{
    const uint64 numAddresses = (uint64)( sizeof( TestTestnetAddress) / sizeof( AddrHashTestPair ) );
    PuzzleHash generatedHash;

    for( uint64 i = 0; i < numAddresses; i++ )
    {
        const AddrHashTestPair& pair = TestTestnetAddress[i];

        ENSURE( PuzzleHash::FromAddress( generatedHash, pair.address ) );

        PuzzleHash expectedHash;
        ENSURE( PuzzleHash::FromHex( pair.hash, expectedHash ) );

        ENSURE( memcmp( generatedHash.data, expectedHash.data, CHIA_PUZZLE_HASH_SIZE ) == 0 );
    }
}