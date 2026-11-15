#!/bin/bash

# Regenerate vendor wheels from the uv lockfile
uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate --output-file requirements.txt
rm -rf flatpak/vendor
pip download --no-deps --dest flatpak/vendor -r requirements.txt
pip download --no-deps --dest flatpak/vendor -r <(echo "uv-build")

# Build the Flatpak (no network access during build)
flatpak-builder --force-clean --user --install-deps-from=flathub --repo=repo --install builddir flatpak/com.alukens.qr-cli.yaml