import urllib.error
from pathlib import Path

import pytest

from qr_reader_cli.main import (
    generate_qr_code,
    handle_image,
    handle_image_path_parse,
    qcd,
    read_qr_code,
    save_qr_code,
)

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

def test_generate_qr_code_round_trip():
    text = "https://example.com"
    img = generate_qr_code(text)
    assert img.ndim == 2
    assert img.shape[0] == img.shape[1]
    result = handle_image(img, qcd)
    assert list(result) == [text]

def test_generate_qr_code_round_trip_long_text():
    text = "https://example.com/" + "a" * 200
    img = generate_qr_code(text)
    result = handle_image(img, qcd)
    assert list(result) == [text]


def test_save_qr_code_writes_scannable_png(tmp_path):
    text = "https://example.com"
    img = generate_qr_code(text)
    out = tmp_path / "qr.png"
    save_qr_code(img, out)
    assert out.exists()
    import cv2

    saved = cv2.imread(str(out))
    assert saved is not None
    assert list(handle_image(saved, qcd)) == [text]


def test_save_qr_code_bad_parent_dir_raises(tmp_path):
    img = generate_qr_code("https://example.com")
    with pytest.raises(OSError):
        save_qr_code(img, tmp_path / "missing" / "qr.png")
