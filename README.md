QR Code Reader CLI

Specify an image containing a QR code, and print the contents out to the console.
Or redirect to a file for qr codes containing file data.

## Development

Dependencies are managed with [uv](https://docs.astral.sh/uv/); versions are pinned in `uv.lock`.

```sh
uv sync          # install into .venv
uv run qr-cli    # run the CLI
uv run pytest    # tests
```

Releases are cut from Actions -> Release -> Run workflow with the new version number.
