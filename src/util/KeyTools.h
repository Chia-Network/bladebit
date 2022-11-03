#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"

#pragma warning( push )

extern "C" {
    #include "bech32/segwit_addr.h"
}

#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning( disable : 6287  )
#pragma warning( disable : 4267  )
#pragma warning( disable : 26495 )
#include "bls.hpp"
#include "elements.hpp"
#include "schemes.hpp"
#include "util.hpp"
#pragma GCC diagnostic pop
#pragma warning( pop )


#define CHIA_PUZZLE_HASH_SIZE   32
#define CHIA_ADDRESS_MAX_LENGTH 63 // 4 (hrp) + 1 (divisor) + 52 (data) + 6 (checksum)
                                   // hrp is either xch or txch

 #define CHIA_ADDRESS_LENGTH         62
 #define CHIA_TESTNET_ADDRESS_LENGTH 63

struct XchAddress : NBytes<CHIA_ADDRESS_LENGTH+1>
{};

struct PuzzleHash : NBytes<CHIA_PUZZLE_HASH_SIZE>
{
    static bool FromAddress( PuzzleHash& hash, const char address[CHIA_ADDRESS_MAX_LENGTH+1] );

    void ToAddress( char address[CHIA_ADDRESS_MAX_LENGTH+1] );
    std::string ToAddressString();

    void ToHex( char hex[CHIA_PUZZLE_HASH_SIZE+1] ) const;
    std::string ToHex() const;

    static bool FromHex( const char hex[CHIA_PUZZLE_HASH_SIZE*2+1], PuzzleHash& outHash );

};

class KeyTools
{
public:
    static bool HexPKeyToG1Element( const char* hexKey, bls::G1Element& pkey );

    static bls::PrivateKey MasterSkToLocalSK( bls::PrivateKey& sk );

    static void PrintPK( const bls::G1Element&  key );
    static void PrintSK( const bls::PrivateKey& key );
};

