from pathlib import Path

from qr_reader_cli.main import handle_image_path_parse, read_qr_code

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"


def test_read_qr_code_from_url_empty():
    # Test with a valid URL
    url = "https://example.com/qr_code.png"
    many_ok = False
    output_path = None

    # Call the function
    url = handle_image_path_parse(url)
    result = read_qr_code(image_path=url, many_ok=many_ok, output_path=output_path)

    # Check the result
    # no QR code should be found
    assert len(result) == 0


def test_read_from_file_success():
    # Test with a valid file path
    file_path = TEST_DATA_DIR / "IMG_0827.jpg"
    many_ok = False
    output_path = None

    # Call the function
    result = read_qr_code(
        image_path=file_path, many_ok=many_ok, output_path=output_path
    )

    # Check the result
    assert len(result) == 1
    assert result[0] == "http://alukens.com"
