import { Octokit } from '@octokit/action';
import * as FS   from 'fs';
import * as Path from 'path';

const log = console.log;

const OWNER    = 'Chia-Network';
const REPO     = 'bladebit';
const API_BASE = `/repos/${OWNER}/${REPO}`;

function failIf( condition, error )
{
    if( condition )
        throw new Error( error );
}

function failIfResponseIsNot( response, expectedCode, error )
{
    if( response.status !== expectedCode )
        throw new Error( error );
}

function failIfErrorResponse( response, error )
{
    if( response.status !== 200 )
        throw new Error( error );
}


async function getArtifactUrl( argv )
{
    // Get args
    // const version      = argv[0].toLowerCase();
    // const os           = argv[1].toLowerCase();
    // const arch         = argv[2].toLowerCase();
    // const ext          = os === 'windows' ? 'zip' : 'tar.gz';

    // const artifactName = `bladebit-v${version}-${os}-${arch}.${ext}`;
    const artifactName = argv[0];
    
    if( !artifactName )
        throw new Error( 'No artifact name given.' );
    
    const API     = `${API_BASE}/actions/artifacts`;
    const octokit = new Octokit();

    let response = await octokit.request( `GET ${API}` );
    failIfErrorResponse( response, 'Invalid artifacts response.' );
    
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
    failIfErrorResponse( response, 'Failed to retrieve artifact download url.'  );
    
    const downloadUrl = response.url;
    return downloadUrl;
}

async function uploadReleaseAsset( argv )
{
    const API = `${API_BASE}/releases`;

    const version   = argv[0];
    const assetName = argv[1];
    const assetPath = argv[2];
    const tag       = 'v' + version;
    
    const octokit = new Octokit();

    let response = await octokit.request( `GET ${API}` );
    failIfErrorResponse( response, 'Failed to retrieve releases' );
    // log( JSON.stringify( response.data, null, 4 ) );

    // Find the specified release
    const release = response.data.find( a => a.tag_name === tag );
    failIf( !release, `Failed to obtain release for version ${version}` );
    // log( JSON.stringify( release, null, 4 ) );

    const asset = FS.readFileSync( assetPath );

    // Upload the artifact to the release as an asset
    response = await octokit.rest.repos.uploadReleaseAsset({
        owner     : OWNER,
        repo      : REPO,
        release_id: release.id,
        name      : assetName,
        data      : asset,
    });
    failIfResponseIsNot( response, 201, `Failed to upload asset ${assetName}` );
}

async function main()
{
    const command = process.argv[2];
    const argv    = process.argv.slice( 3 );
    
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

await main();
