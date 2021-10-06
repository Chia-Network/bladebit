const { Octokit } = require( '@octokit/action' );
const { FS      } = require( 'fs' );
const log = console.log;

const API = '/repos/Chia-Network/bladebit/actions/artifacts';

// function getArtifactName( os, arch, version )
// {
//     const ext = os === 'windows' ? 'zip' : 'tar.gz';
//     return `bladebit-v${version}-${os}-${arch}.${ext}`;
// }

// async function uploadArtifact( artifacts, os, arch, version )
// {

// }

async function main()
{
    // const argv = process.args.slice( 2 );
    // const opts = JSON.parse( argv[0] )
    const opts = { version: '1.2.0' };

    const octokit = new Octokit();
    
    log( `Retrieving artifacts` );
    let response = await octokit.request( `GET ${API}` );
    
    if( response.status !== 200 )
        throw new Error( 'Invalid artifacts response.' );
    
    log( `Looking for artifact for version ${opts.version}` );
    
    const artifacts = response.data.artifacts;
    // log( JSON.stringify( artifacts, 4 ) );

    const artifactName = 'bladebit-v1.2.0-ubuntu-x86-64.tar.gz'// getArtifactName( opts.os, opts.version, opts.arch );
    const artifact     = artifacts.find( a => a.name === artifactName );

    log( JSON.stringify( artifact, 4 ) );

    if( !artifact )
        throw new Error( `Failed to find an artifact name '${artifactName}'.` );

    if( artifact.expired )
        throw new Error( `Artifact '${artifactName}' has expired.` );
    
    log( `Downloading ${artifactName}` );
    response = await octokit.request( `GET ${API}/{artifact_id}/{archive_format}`, {
        artifact_id   : artifact.id,
        archive_format: 'zip'
    });

    log( JSON.stringify( response, 4 ) );

    if( response !== 200 && response !== 302 )
        throw new Error( `Failed to download artifact.` );
    
    // const artifactUrl = artifact.archive_download_url;
    // FS.
}

main()
.then( () => process.exit( 0 ) )
.catch( err => { console.error( err ); process.exit( 1 ); } );


