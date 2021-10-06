const { Octokit } = require( '@octokit/action' );

const octokit = new Octokit();

const artifacts = await octokit.request( 'GET /repos/{owner}/{repo}/actions/artifacts', {
    owner: 'Chia-Network',
    repo : 'bladebit'
});

console.log( JSON.stringify( artifacts, 4 ) );