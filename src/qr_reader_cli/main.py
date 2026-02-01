#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import numpy as np
import questionary
from PIL import Image
from rich.live import Live
from rich.spinner import Spinner
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
parser.add_argument(
    "--downscale-factor", type=float, default=1.0, help="Image downscale factor."
)
parser.add_argument("--skip-frames", type=int, default=5, help="Number of frames to skip.")
parser.add_argument("--no-display", action="store_true", help="Do not display webcam feed in console.")

is_snap = os.environ.get("SNAP_NAME", "") != ""
qcd = cv2.QRCodeDetector()
wechat_qcd = cv2.wechat_qrcode.WeChatQRCode()


def read_image_from_url(url: str) -> np.ndarray:
    logger.debug("Reading image from URL: %s", url)
    response = urllib.request.urlopen(url)
    image_data = np.array(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image from URL: {url}")
    return img


def read_qr_code(
    img: np.ndarray,
    many_ok: bool,
    output_path: Path,
    use_wechat: bool = False,
) -> Sequence[str]:
    logger.debug("Reading QR code from image")

    decoded_info = handle_image(img, qcd if not use_wechat else wechat_qcd)
    if not decoded_info:
        logger.debug("No QR code data decoded.")
        return []
    if not many_ok and len(decoded_info) > 1:
        logger.error(
            "Multiple QR codes found in the image, but --many-ok not specified."
        )
        return []

    logger.debug("Decoded QR code data: %s", decoded_info)
    if output_path:
        with open(output_path, "w") as f:
            json.dump(decoded_info, f)
        logger.info("QR code data saved to %s", output_path)
    else:
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
def get_image_from_webcam(
    webcam_path: str, skip_frames: int, use_wechat: bool = False, downscale_factor: float = 1.0, display: bool = True
):
    # open webcam using OpenCV
    logger.debug("Opening webcam...")
    frame = None
    cap = cv2.VideoCapture(webcam_path)
    frame_count = 0
    set_capture_max_resolution(cap)

    try:
        if not cap.isOpened():
            logger.error("Could not open webcam.")
            return None
        logger.debug("Webcam opened successfully.")
        
        with Live(refresh_per_second=10, screen=display) as live:
            while True:
                # Drain buffered frames to get the latest frame (skip 4 old frames)
                for _ in range(skip_frames):
                    cap.grab()
                # Now read the latest frame
                ret, frame = cap.retrieve()
                if not ret:
                    logger.error("Failed to capture image from webcam.")
                    break
                decoded_data = handle_image(
                    frame, qcd=wechat_qcd if use_wechat else qcd
                )
                if display:
                    display_image_on_console(live=live, image=frame)
                else:
                    spinner = Spinner("dots", text=f"[bold green]Processing image from webcam... ({frame_count})[/bold green]")
                    live.update(spinner)
                if decoded_data:
                    logger.debug("QR code detected in webcam image.")
                    break
                frame_count +=1
    finally:
        cap.release()

    # store frame in temporary file with path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        if frame is None:
            logger.error("No frame captured from webcam.")
            raise Exception("No frame captured from webcam.")

        cv2.imwrite(tmpfile.name, frame)
        temp_path = Path(tmpfile.name)
        yield temp_path


def handle_image_path_parse(image_path: str) -> np.ndarray:
    """Load an image from a file path or URL."""
    parsed_path = urlparse(image_path)
    if parsed_path.scheme in ["http", "https"]:
        logger.debug("URL path provided: %s", parsed_path)
        return read_image_from_url(image_path)
    else:
        logger.debug("File path provided: %s", image_path)
        if not Path(image_path).exists():
            logger.error("File does not exist: %s", image_path)
            raise FileNotFoundError(f"File does not exist: {image_path}")
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image from file: {image_path}")
        return img


def get_webcam_name_from_sysfs(dev_path: str) -> str | None:
    try:
        base = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(dev_path)}")
        with open(os.path.join(base, "name"), "r") as f:
            return f.read().strip()
    except Exception:
        return None


def get_selected_webcam() -> str:
    webcams = sorted([str(path) for path in Path("/dev").glob("video*")])
    if not webcams:
        logger.error(
            "No webcams found. Please specify an image path or connect a webcam"
        )
        sys.exit(1)
    # prompt user to select webcam if multiple are available
    if len(webcams) == 1:
        logger.debug("Single webcam found: %s", webcams[0])
        return webcams[0]

    logger.debug("Multiple webcams found: %s", webcams)
    webcam_choices = []
    for i, webcam in enumerate(webcams):
        webcam_name = get_webcam_name_from_sysfs(webcam)
        webcam_choices.append(
            questionary.Choice(title=f"{i}: {webcam} ({webcam_name})", value=webcam)
        )

    selected = None
    # use questionary to prompt user to select webcam
    selected = questionary.select(
        "Multiple webcams detected. Select one:", choices=webcam_choices
    ).ask()

    if selected is None:
        logger.error("No webcam selected. Exiting.")
        sys.exit(1)

    return selected


def main():
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Arguments: %s", args)
    if args.image_path:
        read_qr_code(
            img=handle_image_path_parse(args.image_path),
            many_ok=args.many_ok,
            output_path=args.output,
            use_wechat=args.use_wechat,
        )
        return

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
            sys.exit(1)
        logger.debug("Camera interface is connected.")
    else:
        logger.debug("Not running in a snap, assuming camera interface is available.")

    # open webcam
    selected_webcam = get_selected_webcam()
    try:
        while True:
            with get_image_from_webcam(
                webcam_path=selected_webcam,
                skip_frames=args.skip_frames,
                use_wechat=args.use_wechat,
                downscale_factor=args.downscale_factor,
                display=not args.no_display,
            ) as webcam_image:
                if webcam_image is None:
                    logger.error("No image captured from webcam.")
                    sys.exit(1)
                # Load the captured image from temp file
                img = cv2.imread(str(webcam_image))
                if img is None:
                    logger.error("Failed to load webcam image.")
                    continue
                decoded_data = read_qr_code(
                    img=img,
                    many_ok=args.many_ok,
                    output_path=args.output,
                    use_wechat=args.use_wechat,
                )
                if all(not data for data in decoded_data):
                    logger.error("No QR code data decoded from the image.")
                    continue
                break
    except KeyboardInterrupt:
        logger.info("QR code reading interrupted by user.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
