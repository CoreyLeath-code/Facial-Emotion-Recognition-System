# GHCR release recovery

If a semantic-version tag exists and its GitHub Release is already published but the container package was not pushed, use the `workflow_dispatch` input on `.github/workflows/release.yml` with that existing tag. The recovery path validates the tag against package metadata and the changelog, reruns release validation, and republishes the container without creating a new release version.
