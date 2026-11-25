QR Code Reader CLI

Specify an image containing a QR code, and print the contents out to the console.
Or redirect to a file for qr codes containing file data.

Generate a QR code from text and show it in the terminal:

```sh
uv run qr-cli --generate "https://example.com"
uv run qr-cli --generate "https://example.com" -o qr.png
```

The `-o` flag saves the generated QR code image to a PNG file instead of displaying it.
Displaying requires a terminal; stdout redirected to a file or pipe fails with a hint to use `-o`.
Extremely long text exceeds QR capacity and is rejected with a usage error.

## Development

Dependencies are managed with [uv](https://docs.astral.sh/uv/); versions are pinned in `uv.lock`.

```sh
uv sync          # install into .venv
uv run qr-cli    # run the CLI
uv run pytest    # tests
```

Releases are cut from Actions -> Release -> Run workflow with the new version number.
Each release gets an amd64 snap and a `py3-none-any` wheel attached.

```sh
uv build --wheel   # local wheel, dist/
```
