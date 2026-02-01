#!/bin/bash

# Regenerate vendor wheels from requirements
poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev
rm -rf flatpak/vendor
pip download --no-deps --dest flatpak/vendor -r requirements.txt
pip download --no-deps --dest flatpak/vendor -r <(echo "poetry-core")

# Build the Flatpak (no network access during build)
flatpak-builder --force-clean --user --install-deps-from=flathub --repo=repo --install builddir flatpak/com.alukens.qr-cli.yaml