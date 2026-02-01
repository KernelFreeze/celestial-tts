#!/usr/bin/env nu

# Create a release: stage changes, commit, tag with version from pyproject.toml, and push
# Usage: ./release.nu

const pyproject_path = path self "../pyproject.toml"

def main [] {
    # Extract version from pyproject.toml
    let content = open $pyproject_path
    let version = $content | get project | get version

    if ($version | is-empty) {
        print "Error: Could not extract version from pyproject.toml"
        exit 1
    }

    print $"📦 Preparing release for version ($version)"

    # Stage all changes
    print "→ Staging changes..."
    git add .

    # Commit with turbocommit
    print "→ Creating commit with turbocommit..."
    turbocommit -c ~/.config/turbocommit/config.yml

    # Create git tag
    print $"→ Creating tag ($version)..."
    git tag -m "Release $version" $version

    # Push repository and tags
    print "→ Pushing to remote..."
    git push
    git push --tags

    print $"✓ Release ($version) complete!"
}
