#include "PlotTools.h"
#include "util/Util.h"


#define PLOT_FILE_PREFIX_LEN (sizeof("plot-k32-2021-08-05-18-55-")-1)

//-----------------------------------------------------------
void PlotTools::GenPlotFileName( const byte plotId[BB_PLOT_ID_LEN], char outPlotFileName[BB_PLOT_FILE_LEN] )
{
    ASSERT( plotId );
    ASSERT( outPlotFileName );

    time_t     now = time( nullptr );
    struct tm* t   = localtime( &now ); ASSERT( t );
    
    const size_t r = strftime( outPlotFileName, BB_PLOT_FILE_LEN, "plot-k32-%Y-%m-%d-%H-%M-", t );
    if( r != PLOT_FILE_PREFIX_LEN )
        Fatal( "Failed to generate plot file." );

    PlotIdToString( plotId, outPlotFileName + r );
    memcpy( outPlotFileName + r + BB_PLOT_ID_HEX_LEN, ".plot.tmp", sizeof( ".plot.tmp" ) );
}

//-----------------------------------------------------------
void PlotTools::PlotIdToString( const byte plotId[BB_PLOT_ID_LEN], char plotIdString[BB_PLOT_ID_HEX_LEN+1] )
{
    ASSERT( plotId );
    ASSERT( plotIdString );

    size_t numEncoded = 0;
    BytesToHexStr( plotId, BB_PLOT_ID_LEN, plotIdString, BB_PLOT_ID_HEX_LEN, numEncoded );
    ASSERT( numEncoded == BB_PLOT_ID_LEN );

    plotIdString[BB_PLOT_ID_HEX_LEN] = '\0';
}

//-----------------------------------------------------------
 bool PlotTools::PlotStringToId( const char plotIdString[BB_PLOT_ID_HEX_LEN+1], byte plotId[BB_PLOT_ID_LEN] )
 {
    const size_t len = strlen( plotIdString );
    if( len < 64 && len != 66 )
        return false;
    
    if( len == 66 )
    {
        if( plotIdString[0] == '0' && plotIdString[1] == 'x' )
            plotIdString += 2;
        else
            return false;
    }

    HexStrToBytes( plotIdString, len, plotId, BB_PLOT_ID_LEN );
    return true;
 }

//-----------------------------------------------------------
bls::G1Element PlotTools::GeneratePlotPublicKey( const bls::G1Element& localPk, bls::G1Element& farmerPk, const bool includeTaproot )
{
    bls::G1Element plotPublicKey;

    if( includeTaproot )
    {
        std::vector<uint8_t> taprootMsg = (localPk + farmerPk).Serialize();
        taprootMsg = BytesConcat( taprootMsg, localPk.Serialize(), farmerPk.Serialize() );
        
        byte tapRootHash[32];
        bls::Util::Hash256( tapRootHash, taprootMsg.data(), taprootMsg.size() );

        bls::PrivateKey taprootSk = bls::AugSchemeMPL().KeyGen( bls::Bytes( tapRootHash, sizeof( tapRootHash ) ) );
        
        plotPublicKey = localPk + farmerPk + taprootSk.GetG1Element();
    }
    else
    {
        plotPublicKey = localPk + farmerPk;
    }

    return plotPublicKey;
}

//-----------------------------------------------------------
void PlotTools::GeneratePlotIdAndMemo( 
    byte            plotId  [BB_PLOT_ID_LEN], 
    byte            plotMemo[BB_PLOT_MEMO_MAX_SIZE], 
    uint16&         outMemoSize,
    bls::G1Element& farmerPK,
    bls::G1Element* poolPK,
    PuzzleHash*     contractPuzzleHash
    )
{
    // Generate random master secret key
    byte seed[32];
    SysHost::Random( seed, sizeof( seed ) );

    bls::PrivateKey sk      = bls::AugSchemeMPL().KeyGen( bls::Bytes( seed, sizeof( seed ) ) );
    bls::G1Element  localPk = std::move( KeyTools::MasterSkToLocalSK( sk ) ).GetG1Element();

    // #See: chia-blockchain create_plots.py
    //       The plot public key is the combination of the harvester and farmer keys
    //       New plots will also include a taproot of the keys, for extensibility
    const bool includeTaproot = contractPuzzleHash != nullptr;
    
    bls::G1Element plotPublicKey = std::move( GeneratePlotPublicKey( localPk, farmerPK, includeTaproot ) );
    
    std::vector<uint8_t> farmerPkBytes = farmerPK.Serialize();
    std::vector<uint8_t> localSkBytes  = sk.Serialize();

    // The plot id is based on the harvester, farmer, and pool keys
    if( !includeTaproot )
    {
        std::vector<uint8_t> bytes = poolPK->Serialize();
        
        // Gen plot id
        auto plotPkBytes = plotPublicKey.Serialize();
        bytes.insert( bytes.end(), plotPkBytes.begin(), plotPkBytes.end() );

        bls::Util::Hash256( plotId, bytes.data(), bytes.size() );

        // Gen memo
        auto memoBytes = BytesConcat( poolPK->Serialize(), farmerPkBytes, localSkBytes );

        const size_t poolMemoSize = 48 + 48 + 32;
        ASSERT( memoBytes.size() == poolMemoSize );

        memcpy( plotMemo, memoBytes.data(), poolMemoSize );
        outMemoSize = (uint16)poolMemoSize;
    }
    else
    {
        // Create a pool plot with a contract puzzle hash
        ASSERT( contractPuzzleHash );

        const auto& ph = *contractPuzzleHash;
        std::vector<uint8_t> phBytes( (uint8_t*)ph.data, (uint8_t*)ph.data + CHIA_PUZZLE_HASH_SIZE );
        
        // Gen plot id
        std::vector<uint8_t> plotIdBytes = phBytes;
        auto plotPkBytes = plotPublicKey.Serialize();

        plotIdBytes.insert( plotIdBytes.end(), plotPkBytes.begin(), plotPkBytes.end() );
        bls::Util::Hash256( plotId, plotIdBytes.data(), plotIdBytes.size() );

        // Gen memo
        auto memoBytes = BytesConcat( phBytes, farmerPkBytes, localSkBytes );

        const size_t phMemoSize = 32 + 48 + 32;
        ASSERT( memoBytes.size() == phMemoSize );

        memcpy( plotMemo, memoBytes.data(), phMemoSize );
        outMemoSize = (uint16)phMemoSize;
    }
}
