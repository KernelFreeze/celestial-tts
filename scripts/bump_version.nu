#!/usr/bin/env nu

# Bump the version in pyproject.toml
# Usage: ./bump_version.nu <new-version>

const pyproject_path = path self "../pyproject.toml"

def main [version: string] {
    # Validate version format (basic semver check)
    if not ($version =~ '^\d+\.\d+\.\d+') {
        print $"Error: Invalid version format '($version)'. Expected format: X.Y.Z"
        exit 1
    }

    # Read the current file
    mut content = open $pyproject_path

    # Replace the version line
    let updated = $content | update project { $in | update version $version }

    # Write back to file
    $updated | save -f $pyproject_path

    print $"✓ Bumped version to ($version) in pyproject.toml"
}
