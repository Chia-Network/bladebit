# Navigate to the script's directory
$scriptPath = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location -Path $scriptPath

# Arguments
$ver_component = $args[0]  # The user-specified component from the full version

# Read the version from the file
$version_str = (Get-Content 'VERSION' | Select-Object -First 1 | Out-String).Trim()
$bb_version_suffix = (Get-Content 'VERSION' | Select-Object -Last 1 | Out-String).Trim()
$version_header = 'src\Version.h'

if ($version_str -eq $bb_version_suffix) {
    $bb_version_suffix = ""
}

# Prepend a '-' to the suffix, if necessary
if (-Not [string]::IsNullOrEmpty($bb_version_suffix) -and $bb_version_suffix[0] -ne '-') {
    $bb_version_suffix = "-$bb_version_suffix"
}

# Parse the major, minor, and revision numbers
$bb_ver_maj, $bb_ver_min, $bb_ver_rev = $version_str -split '\.' | ForEach-Object { $_.Trim() }

# Get the Git commit hash
$bb_git_commit = $env:GITHUB_SHA
if ([string]::IsNullOrEmpty($bb_git_commit)) {
    $bb_git_commit = & git rev-parse HEAD
}

if ([string]::IsNullOrEmpty($bb_git_commit)) {
    $bb_git_commit = "unknown"
}

# Check if the user wants a specific component
if (-Not [string]::IsNullOrEmpty($ver_component)) {
    switch ($ver_component) {
        "major" {
            Write-Host -NoNewline $bb_ver_maj
        }
        "minor" {
            Write-Host -NoNewline $bb_ver_min
        }
        "revision" {
            Write-Host -NoNewline $bb_ver_rev
        }
        "suffix" {
            Write-Host -NoNewline $bb_version_suffix
        }
        "commit" {
            Write-Host -NoNewline $bb_git_commit
        }
        default {
            Write-Error "Invalid version component '$ver_component'"
            exit 1
        }
    }
    exit 0
}

