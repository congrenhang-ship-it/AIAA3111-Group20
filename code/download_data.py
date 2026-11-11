"""Download and verify the original ULB file without third-party packages."""

import argparse
from http.client import HTTPException
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from preprocessing import DEFAULT_DATA_PATH, ULB_DOWNLOAD_URL, ULB_SHA256, ULB_SIZE_BYTES, data_sha256


def verify_file(path):
    size = path.stat().st_size
    if size != ULB_SIZE_BYTES:
        raise ValueError(f"{path} has {size} bytes; expected {ULB_SIZE_BYTES}.")
    digest = data_sha256(path)
    if digest != ULB_SHA256:
        raise ValueError(f"{path} has SHA-256 {digest}; expected {ULB_SHA256}.")


def download_data(destination):
    """Reuse a verified file or install a verified download at the destination."""
    destination = Path(destination)
    if destination.exists():
        verify_file(destination)
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(ULB_DOWNLOAD_URL, headers={"User-Agent": "AIAA3111-Group20/1.0"})
    print(f"Downloading the pinned Zenodo CSV to {destination.parent} ...", file=sys.stderr)
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False,
        ) as output:
            temporary = Path(output.name)
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(f"Expected a full download (HTTP 200), got HTTP {response.status}.")
                progress_step = 16 * 1024 * 1024
                next_update = progress_step
                while chunk := response.read1(1024 * 1024):
                    received = output.tell() + len(chunk)
                    if received > ULB_SIZE_BYTES:
                        raise ValueError(f"Download exceeds the expected {ULB_SIZE_BYTES} bytes.")
                    output.write(chunk)
                    if received >= next_update:
                        print(f"Received {received / 1024**2:.0f} of {ULB_SIZE_BYTES / 1024**2:.0f} MiB", file=sys.stderr)
                        next_update += progress_step
        verify_file(temporary)
        if destination.exists():
            verify_file(destination)
            return False
        temporary.replace(destination)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH, help="Destination CSV path.")
    args = parser.parse_args()
    try:
        downloaded = download_data(args.output)
    except (OSError, HTTPException, ValueError) as exc:
        parser.error(f"{exc} See data/README.md for the manual download and verification steps.")
    action = "Downloaded and verified" if downloaded else "Verified existing file"
    print(f"{action}: {args.output}\nSHA-256: {ULB_SHA256}")


if __name__ == "__main__":
    main()
