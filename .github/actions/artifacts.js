const { Octokit } = require( '@octokit/action' );
const log = console.log;

const API_BASE = '/repos/Chia-Network/bladebit';

function failIfError( response, error )
{
    if( response.status !== 200 )
        throw new Error( error );
}

async function getArtifactUrl( argv )
{
    const API = `${API_BASE}/actions/artifacts`;
    
    const artifactName = argv[0];
    if( !artifactName )
        throw new Error( 'No artifact name given.' );

    const octokit = new Octokit();

    let response = await octokit.request( `GET ${API}` );
    failIfError( resposne, 'Invalid artifacts response.' );
    
    // log( `Looking for artifact for version ${opts.version}` );
    const artifacts = response.data.artifacts;
    // log( JSON.stringify( artifacts, 4 ) );

    const artifact = artifacts.find( a => a.name === artifactName );
    // log( JSON.stringify( artifact, 4 ) );

    if( !artifact )
        throw new Error( `Failed to find an artifact name '${artifactName}'.` );

    if( artifact.expired )
        throw new Error( `Artifact '${artifactName}' has expired.` );
    
    response = await octokit.request( `GET ${API}/{artifact_id}/{archive_format}`, {
        artifact_id   : artifact.id,
        archive_format: 'zip'
    });
    // log( JSON.stringify( response, 4 ) );

    // Docs say 302, but the returned code is actually 200
    failIfError( resposne, 'Failed to retrieve artifact download url.'  );
    
    const downloadUrl = response.url;
    return downloadUrl;
}

async function uploadReleaseAsset( argv )
{
    const API = `${API_BASE}/releases`;

    const version   = argv[0];
    const assetPath = argv[1];
    
    const octokit = new Octokit();

    let response = await octokit.request( `GET ${API}` );
    failIfError( response, 'Failed to retrieve releases' );

    log( JSON.stringify( response.data, null, 4 ) );
}

async function main()
{
    const command = process.args[2];
    const argv    = process.args.slice( 3 );
    
    switch( command )
    {
        case 'get-artifact-url':
            log( ( await getArtifactUrl( argv ) ) );
        break;
    
        case 'upload-release-asset':
            log( ( await uploadReleaseAsset( argv ) ) );
        break;
    }
}


main()
.then( () => process.exit( 0 ) )
.catch( err => { console.error( err ); process.exit( 1 ); } );


