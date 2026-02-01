#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence
from urllib.parse import ParseResult, urlparse

import numpy as np
import questionary
from PIL import Image
from rich.live import Live
from rich_pixels import Pixels

# set log level for OpenCV to ERROR before importing cv2
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser(description="Read QR codes from an image file.")
parser.add_argument(
    "-i",
    "--image-path",
    type=str,
    help="Path to the image file containing the QR code.",
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output."
)
parser.add_argument(
    "-o", "--output", type=Path, help="Path to save the decoded QR code output."
)
parser.add_argument(
    "--many-ok", action="store_true", help="Allow multiple QR codes in the image."
)
parser.add_argument(
    "--use-wechat", action="store_true", help="Use WeChat QR code detection."
)

is_snap = os.environ.get("SNAP_NAME", "") != ""
qcd = cv2.QRCodeDetector()
wechat_qcd = cv2.wechat_qrcode.WeChatQRCode()


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


def read_qr_code(
    image_path: ParseResult | Path,
    many_ok: bool,
    output_path: Path,
    use_wechat: bool = False,
) -> Sequence[str]:
    logger.debug("Reading QR code from: %s", image_path)
    if isinstance(image_path, ParseResult):
        web_image_path = f"{image_path.scheme}://{image_path.netloc}{image_path.path}"
        # read image from URL
        img = read_image_from_url(web_image_path)
    else:
        try:
            img = cv2.imread(str(image_path))
        except Exception as e:
            logger.exception("Error reading image from file: %s", e)
            img = None

    if img is None:
        logger.error("Failed to read image from path: %s", image_path)
        return []
    decoded_info = handle_image(img, qcd if not use_wechat else wechat_qcd)
    if decoded_info is None:
        return []
    if not many_ok and len(decoded_info) > 1:
        logger.error(
            "Multiple QR codes found in the image, but --many-ok not specified."
        )
        return []

    if decoded_info:
        logger.debug("Decoded QR code data: %s", decoded_info)
        if output_path:
            with open(output_path, "w") as f:
                json.dump(decoded_info, f)
            logger.debug("QR code data saved to %s", output_path)
        else:
            logger.debug("No output path specified, printing decoded QR code data.")
            for result in decoded_info:
                print(result)

    return decoded_info


def display_image_on_console(live: Live, image: np.ndarray):
    # get size of console
    console_size = live.console.size

    # Convert OpenCV image (BGR) to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    live.update(
        Pixels.from_image(
            pil_image, resize=(console_size.width, console_size.height * 2)
        )
    )
    live.refresh()


def set_capture_max_resolution(cap: cv2.VideoCapture):
    # set maximum resolution for webcam capture
    # it will automatically adjust to the maximum supported resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)


def handle_image(
    frame, qcd: cv2.QRCodeDetector | cv2.wechat_qrcode.WeChatQRCode
) -> Sequence[str] | None:
    logger.debug("Handling image for QR code detection.")

    def process_detection_result(
        decoded_data: Sequence[str] | None, success: bool = True
    ) -> Sequence[str] | None:
        if success and decoded_data:
            logger.debug("QR code detected in image.")
            return decoded_data
        else:
            logger.debug("No QR code found in the image.")
            return None

    if isinstance(qcd, cv2.wechat_qrcode.WeChatQRCode):
        decoded_data, _ = qcd.detectAndDecode(frame, None)
        return process_detection_result(decoded_data)
    else:
        success, decoded_data, _, _ = qcd.detectAndDecodeMulti(frame)
        return process_detection_result(decoded_data, success)


@contextmanager
def get_image_from_webcam(webcam_path: str, use_wechat: bool = False):
    # open webcam using OpenCV
    logger.debug("Opening webcam...")
    frame = None
    cap = cv2.VideoCapture(webcam_path)
    set_capture_max_resolution(cap)

    try:
        if not cap.isOpened():
            logger.error("Could not open webcam.")
            return None
        logger.debug("Webcam opened successfully.")

        with Live(refresh_per_second=10, screen=True) as live:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture image from webcam.")
                    break
                decoded_data = handle_image(
                    frame, qcd=wechat_qcd if use_wechat else qcd
                )
                display_image_on_console(live=live, image=frame)
                if decoded_data:
                    logger.debug("QR code detected in webcam image.")
                    break
    finally:
        cap.release()

    # store frame in temporary file with path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        if frame is None:
            logger.error("No frame captured from webcam.")
            yield None
        else:
            cv2.imwrite(tmpfile.name, frame)
            temp_path = Path(tmpfile.name)

            yield temp_path


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


def get_webcam_name_from_sysfs(dev_path: str) -> str | None:
    try:
        base = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(dev_path)}")
        with open(os.path.join(base, "name"), "r") as f:
            return f.read().strip()
    except Exception:
        return None


def get_selected_webcam() -> str:
    webcams = [str(path) for path in Path("/dev").glob("video*")]
    if not webcams:
        logger.error(
            "No webcams found. Please specify an image path or connect a webcam"
        )
        exit(1)
    # prompt user to select webcam if multiple are available
    selected_webcam = webcams[0]
    if len(webcams) > 1:
        logger.info("Multiple webcams found: %s", webcams)
        webcam_choices = []
        for i, webcam in enumerate(webcams):
            webcam_name = get_webcam_name_from_sysfs(webcam)
            webcam_choices.append(f"{i}: {webcam} ({webcam_name})")

        selected = None
        # use questionary to prompt user to select webcam
        logger.debug("Prompting user to select webcam...")
        selected = questionary.select(
            "Multiple webcams detected. Select one:", choices=webcam_choices
        ).ask()

        if selected is None:
            logger.error("No webcam selected. Exiting.")
            exit(1)

        selected_index = webcam_choices.index(selected)
        selected_webcam = webcams[selected_index]

    return selected_webcam


def main():
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Arguments: %s", args)
    if args.image_path:
        image_path = handle_image_path_parse(args.image_path)
        read_qr_code(
            image_path=image_path,
            many_ok=args.many_ok,
            output_path=args.output,
            use_wechat=args.use_wechat,
        )
    else:
        logger.info("No image path provided, using webcam to capture QR code.")
        if is_snap:
            # use snapctl to see if camera interface is connected
            result = subprocess.run(
                "snapctl is-connected camera",
                shell=True,
                check=False,
                capture_output=True,
            )
            if result.returncode != 0:
                logger.error(
                    "Camera interface is not connected. Please connect it using 'snap connect <snap>:camera'"
                )
                exit(1)
            logger.debug("Camera interface is connected.")
        else:
            logger.debug(
                "Not running in a snap, assuming camera interface is available."
            )

        # open webcam
        selected_webcam = get_selected_webcam()
        try:
            while True:
                with get_image_from_webcam(
                    webcam_path=selected_webcam, use_wechat=args.use_wechat
                ) as webcam_image:
                    if webcam_image is None:
                        logger.error("No image captured from webcam.")
                        exit(1)
                    image_path = webcam_image
                    decoded_data = read_qr_code(
                        image_path=image_path,
                        many_ok=args.many_ok,
                        output_path=args.output,
                        use_wechat=args.use_wechat,
                    )
                    if all(not data for data in decoded_data):
                        logger.error("No QR code data decoded from the image.")
                        continue
                    break
        except KeyboardInterrupt:
            logger.info("QR code reading interrupted by user.")
            exit(0)


if __name__ == "__main__":
    main()
