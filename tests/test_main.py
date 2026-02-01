import urllib.error
from pathlib import Path

import pytest

from qr_reader_cli.main import handle_image_path_parse, read_qr_code

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"


def test_read_qr_code_from_url_empty():
    # Test with a valid URL
    url = "https://fastly.picsum.photos/id/455/200/200.jpg?hmac=YZhCbBjCYF0ha5dR9ElToDVwWcw05w0e4pAv5S9nZYg"
    many_ok = False
    output_path = None

    # Call the function
    img = handle_image_path_parse(url)
    result = read_qr_code(img=img, many_ok=many_ok, output_path=output_path)

    # Check the result
    # no QR code should be found
    assert len(result) == 0

def test_read_qr_code_invalid_url():
    # Test with an invalid URL
    url = "https://example.com/non_existent_image.jpg"
    many_ok = False
    output_path = None

    # Call the function and expect an exception
    with pytest.raises(urllib.error.HTTPError):
        img = handle_image_path_parse(url)
        read_qr_code(img=img, many_ok=many_ok, output_path=output_path)


def test_read_from_file_success():
    # Test with a valid file path
    file_path = TEST_DATA_DIR / "IMG_0827.jpg"
    img = handle_image_path_parse(str(file_path))
    many_ok = False
    output_path = None

    # Call the function
    result = read_qr_code(img=img, many_ok=many_ok, output_path=output_path)

    # Check the result
    assert len(result) == 1
    assert result[0] == "http://alukens.com"
