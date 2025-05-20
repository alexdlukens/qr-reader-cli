#!/usr/bin/env python3
import argparse
import json
import logging
import urllib
from pathlib import Path
from urllib.parse import ParseResult, urlparse
import os
import numpy as np

# set log level for OpenCV to ERROR before importing cv2
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Read QR codes from an image file.")
parser.add_argument("image_path", type=str, help="Path to the image file containing the QR code.")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
parser.add_argument("-o", "--output", type=Path, help="Path to save the decoded QR code output.")
parser.add_argument("--many-ok", action="store_true", help="Allow multiple QR codes in the image.")

qcd = cv2.QRCodeDetector()

def read_image_from_url(url: str) -> np.ndarray | None:
    logger.debug("Reading image from URL: %s", url)
    try:
        response = urllib.request.urlopen(url)
        image_data = np.array(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.exception("Error reading image from URL: %s", e)
        return None

def read_qr_code(image_path: ParseResult | Path, many_ok: bool, output_path: Path) -> list[str]:
    logger.debug("Reading QR code from: %s", image_path)
    if isinstance(image_path, ParseResult):
        image_path = f"{image_path.scheme}://{image_path.netloc}{image_path.path}"
        # read image from URL
        img = read_image_from_url(image_path)
    else:
        try:
            img = cv2.imread(image_path)
        except Exception as e:
            logger.exception("Error reading image from file: %s", e)
            img = None

    if img is None:
        logger.error("Failed to read image from path: %s", image_path)
        return []
    retval, decoded_info, _, _ = qcd.detectAndDecodeMulti(img)
    if not retval:
        logger.warning("No QR code found in the image.")
        return []
    if not many_ok and len(decoded_info) > 1:
        logger.error("Multiple QR codes found in the image, but --many-ok not specified.")
        return []
    
    if decoded_info:
        logger.debug("Decoded QR code data: %s", decoded_info)
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(decoded_info, f)
            logger.debug("QR code data saved to %s", output_path)
        else:
            logger.debug("No output path specified, printing decoded QR code data.")
            for result in decoded_info:
                print(result)
                
    return decoded_info


def handle_image_path_parse(image_path: str) -> ParseResult | Path:
    parsed_path = urlparse(image_path)
    if parsed_path.scheme in ["http", "https"]:
        logger.debug("URL path provided: %s", parsed_path)
        return parsed_path
    else:
        logger.debug("File path provided: %s", image_path)
        if not Path(image_path).exists():
            logger.error("File does not exist: %s", image_path)
            raise FileNotFoundError(f"File does not exist: {image_path}")
        return Path(image_path)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Arguments: %s", args)
        
    image_path = handle_image_path_parse(args.image_path)
    read_qr_code(image_path=image_path, many_ok=args.many_ok, output_path=args.output)
