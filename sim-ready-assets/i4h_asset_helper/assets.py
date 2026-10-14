# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import importlib
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import as_completed
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from tqdm import tqdm

__all__ = [
    "get_i4h_asset_hash",
    "get_i4h_asset_path",
    "get_i4h_local_asset_path",
    "retrieve_asset",
    "verify_asset",
    "BaseI4HAssets",
]

_I4H_ASSET_ROOT = {
    "dev": "https://isaac-dev.ov.nvidia.com/omni/web3/omniverse://isaac-dev.ov.nvidia.com/Library/IsaacHealthcare",
    "staging": "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare",
    "production": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare",
}


_NUCLEUS_CLIENT_REQUIRED = (
    "Nucleus downloads need omni.client: install `sim-ready-assets[nucleus]` or start the Isaac Sim app "
    "before the asset helper."
)

# S3 bucket mapping
_S3_BUCKETS = {
    "staging": "omniverse-content-staging",
    "production": "omniverse-content-production",
}

# S3 region mapping
_S3_REGIONS = {
    "staging": "us-west-2",
    "production": "us-west-2",
}


def _is_s3_environment() -> bool:
    """Check if current environment uses S3 for asset storage."""
    env = _get_asset_env()
    return env in ["staging", "production"]


def _parse_s3_url(url: str) -> Tuple[str, str]:
    """
    Parse S3 URL to extract bucket and key.

    Args:
        url: The S3 URL (https://bucket-name.s3-region.amazonaws.com/path/to/key)

    Returns:
        Tuple of (bucket_name, key)
    """
    parsed = urlparse(url)

    # Extract bucket name from hostname
    hostname = parsed.netloc
    if hostname.endswith(".amazonaws.com"):
        bucket = hostname.split(".s3-")[0]
    else:
        # Use mapping if hostname doesn't match expected pattern
        env = _get_asset_env()
        bucket = _S3_BUCKETS.get(env)

    # Extract key from path (remove leading slash)
    key = parsed.path
    if key.startswith("/"):
        key = key[1:]

    return bucket, key


def _get_s3_client():
    """Get boto3 S3 client with anonymous configuration for public buckets."""
    env = _get_asset_env()

    try:
        import boto3
        import botocore.config

        # Create an anonymous/unsigned config for public access
        config = botocore.config.Config(
            signature_version=botocore.UNSIGNED,
            retries={
                "max_attempts": 5,
                "mode": "adaptive",  # Use adaptive mode for exponential backoff with jitter
            },
        )

        return boto3.client("s3", region_name=_S3_REGIONS.get(env), config=config)
    except ImportError:
        raise ImportError("boto3 is required for S3 access. Install with 'pip install boto3'")


def _is_import_ready(package_name: str):
    """
    Check if we need to start the simulation app to import the package. If it does, it will return True.
    """
    # try if the package is already imported
    if package_name in sys.modules:
        return True

    try:
        importlib.import_module(package_name)
    except ImportError:
        return False
    return True


def _get_asset_env() -> str:
    """Get the current configuration of the asset root."""
    return os.getenv("I4H_ASSET_ENV", "dev")


def _get_download_dir() -> str:
    """Get the download directory for the current configuration."""
    default_dir = os.path.join(os.path.expanduser("~"), ".cache", "i4h-assets")
    return os.getenv("I4H_ASSET_DOWNLOAD_DIR", default_dir)


def _unify_path(path: str) -> str:
    """
    Unify the path in different configurations.

    When the file is on a nucleus server, the _list_file function returns that path starting with "omniverse://".
    So we need to unify the path to the same format, if the path is not returned by the _list_file function.
    """
    if _get_asset_env() == "dev":
        return path.replace("https://isaac-dev.ov.nvidia.com/omni/web3/", "")
    return path


def _get_default_version() -> str:
    """Get the default version of the i4h asset."""
    with open(os.path.join(os.path.dirname(__file__), "assets_sha256.json"), "r") as f:
        return list(json.load(f).keys())[-1]


def get_i4h_asset_version() -> str:
    """Get the version of the i4h asset."""
    return os.getenv("I4H_ASSET_VERSION", _get_default_version())


def get_i4h_asset_hash(version: str | None = None) -> str | None:
    """Get the sha256 hash for the given version."""
    version = version if version is not None else get_i4h_asset_version()
    # Get it from the environment variable if it exists
    if os.environ.get("I4H_ASSET_SHA256_HASH"):
        return os.environ.get("I4H_ASSET_SHA256_HASH")
    # Otherwise, get it from the file
    with open(os.path.join(os.path.dirname(__file__), "assets_sha256.json"), "r") as f:
        return json.load(f).get(version, None)


def get_i4h_asset_path(version: str | None = None, hash: str | None = None) -> str:
    """
    Get the path to the i4h asset for the given version.

    Args:
        version: The version of the asset to get.
        hash: The sha256 hash of the asset.

    Returns:
        The path to the i4h asset.
    """
    asset_root = _I4H_ASSET_ROOT.get(_get_asset_env())
    version = version if version is not None else get_i4h_asset_version()

    # if the environment is not S3, the hash will be None
    if not _is_s3_environment():
        remote_path = f"{asset_root}/{version}"
        return remote_path

    hash = hash if hash is not None else get_i4h_asset_hash(version=version)
    remote_path = f"{asset_root}/{version}/{hash}"

    return remote_path


def get_i4h_local_asset_path(
    version: str | None = None, download_dir: str | None = None, hash: str | None = None
) -> str:
    """
    Get the path to the i4h asset for the given version.

    Args:
        version: The version of the asset to get.
        download_dir: The directory to download the asset to.
        hash: The sha256 hash of the asset.

    Returns:
        The path to the local asset.
    """
    version = version if version is not None else get_i4h_asset_version()
    if download_dir is None:
        download_dir = _get_download_dir()

    if not _is_s3_environment():
        return os.path.join(download_dir, version)

    hash = hash if hash is not None else get_i4h_asset_hash(version=version)

    if hash is None:
        # If no hash is available for this version, use version as directory name
        return os.path.join(download_dir, version)
    else:
        return os.path.join(download_dir, hash)


_VERIFY_REPORT_LIMIT = 20


def _verify_against_catalog(
    version: str,
    hash: str | None,
    local_dir: str,
    sub_path: str | None = None,
    verbose: bool = False,
) -> bool:
    """
    Verify a download against the remote catalog listing.

    No catalog version publishes per-file checksums, so the strongest available check is that every object
    the catalog lists is present locally at its full size. That is also the check that matters in practice,
    because an interrupted or timed-out download leaves files missing or half-written.
    """
    remote_path = get_i4h_asset_path(version, hash)
    if sub_path is not None:
        remote_path = remote_path + "/" + sub_path

    print(f"Verifying {local_dir} against catalog listing {remote_path} ...")
    entries = _list_asset_entries(remote_path)
    if not entries:
        print("Verification FAILED — the catalog listing contains no files!")
        return False

    missing: List[str] = []
    wrong_size: List[Tuple[str, int, int]] = []

    for url, remote_size in entries:
        relative_path = _get_asset_relpath(url, version, hash)
        local_path = os.path.join(local_dir, relative_path)
        if not os.path.isfile(local_path):
            missing.append(relative_path)
            continue
        local_size = os.path.getsize(local_path)
        if remote_size is not None and local_size != remote_size:
            wrong_size.append((relative_path, remote_size, local_size))
        elif verbose:
            print(f"  ok: {relative_path}")

    print(f"  Catalog files: {len(entries)}")
    print(f"  Missing:       {len(missing)}")
    print(f"  Wrong size:    {len(wrong_size)}")

    if not missing and not wrong_size:
        print("Verification PASSED")
        return True

    for relative_path in missing[:_VERIFY_REPORT_LIMIT]:
        print(f"  missing: {relative_path}")
    if len(missing) > _VERIFY_REPORT_LIMIT:
        print(f"  ... and {len(missing) - _VERIFY_REPORT_LIMIT} more missing files")

    for relative_path, remote_size, local_size in wrong_size[:_VERIFY_REPORT_LIMIT]:
        print(f"  wrong size: {relative_path} (catalog {remote_size} B, local {local_size} B)")
    if len(wrong_size) > _VERIFY_REPORT_LIMIT:
        print(f"  ... and {len(wrong_size) - _VERIFY_REPORT_LIMIT} more size mismatches")

    print("Verification FAILED — the local copy does not match the catalog!")
    return False


def verify_asset(
    version: str | None = None,
    download_dir: str | None = None,
    hash: str | None = None,
    sub_path: str | None = None,
    verbose: bool = False,
) -> bool:
    """
    Verify a downloaded asset folder against the catalog listing for its version.

    Every object the catalog holds under the version's remote prefix must exist locally at its full size.
    The value in assets_sha256.json only names that prefix and the local directory. From 0.5.0 onwards it
    is the catalog's short commit id. Up to 0.3.0 it is the SHA-256 of the asset ZIP package the release
    was built from. That ZIP is not what the S3 catalog serves, so no digest of the download can match it.

    Args:
        version: The version of the asset.
        download_dir: The directory where the asset was downloaded.
        hash: The catalog identifier that names the remote prefix and local directory. If None, looked
            up from assets_sha256.json.
        sub_path: Restrict verification to the sub path that was downloaded, matching the ``--sub-path``
            used to retrieve it.
        verbose: If True, print per-file progress.

    Returns:
        True if the download matches the catalog.

    Raises:
        ValueError: If no catalog entry is available or the folder doesn't exist.
    """
    version = version if version is not None else get_i4h_asset_version()
    if hash is None and get_i4h_asset_hash(version=version) is None:
        raise ValueError(f"No catalog entry found for version {version}")

    local_dir = get_i4h_local_asset_path(version, download_dir, hash)
    if not os.path.isdir(local_dir):
        raise ValueError(f"Asset folder does not exist: {local_dir}")

    return _verify_against_catalog(version, hash, local_dir, sub_path=sub_path, verbose=verbose)


def _get_asset_relpath(url_entry: str, version: str = get_i4h_asset_version(), hash: str | None = None) -> str:
    """
    Get relative path of the item specified by the url_entry should be located in the local asset directory.

    Args:
        url_entry: The entry of the item
        version: The version of the asset
        hash: The sha256 hash of the asset

    Returns:
        The relative path of the item.
    """
    asset_root = get_i4h_asset_path(version, hash)
    asset_root = _unify_path(asset_root)

    if not url_entry.startswith(asset_root):
        raise ValueError(f"URL entry {url_entry} expects to begin with {asset_root}")

    return os.path.relpath(url_entry, asset_root)


def _is_url_folder(url_entry: str) -> bool:
    """Check if the url_entry is a folder."""

    if not _is_import_ready("omni.client"):
        if not _is_s3_environment():
            raise ValueError(_NUCLEUS_CLIENT_REQUIRED)
        # Fallback for S3 environments: boto3
        try:
            bucket, key = _parse_s3_url(url_entry)
            s3_client = _get_s3_client()

            # For S3, a folder is indicated by a key that ends with a '/'
            # If key doesn't end with '/', check if objects exist under "<key>/"
            if not key.endswith("/"):
                # The bare key is not a usable prefix: a file's own key matches it, and so does a sibling such as
                # "NuRec" for "Nu". Only objects below "<key>/" make it a folder.
                max_retries = 5
                retry_count = 0
                backoff_time = 1  # Start with 1 second

                while True:
                    try:
                        response = s3_client.list_objects_v2(
                            Bucket=bucket, Prefix=key + "/", Delimiter="/", MaxKeys=1
                        )
                        # If CommonPrefixes exist, it's a folder
                        return "CommonPrefixes" in response or response.get("KeyCount", 0) > 0
                    except ClientError as e:
                        error_code = e.response.get("Error", {}).get("Code", "")
                        if error_code == "SlowDown" and retry_count < max_retries:
                            # If we hit a rate limit, wait with exponential backoff
                            retry_count += 1
                            sleep_time = backoff_time * (1 + 0.5 * (2.0**retry_count))
                            print(f"Rate limit hit. Retrying... (attempt {retry_count}/{max_retries})")
                            time.sleep(sleep_time)
                        else:
                            # If error is not rate limiting or we're out of retries, raise the exception
                            raise ValueError(f"The remote path {url_entry} returned an error: {str(e)}")
            return True
        except Exception as e:
            raise ValueError(f"The remote path {url_entry} is not valid: {str(e)}")

    import omni.client

    url_entry_unified = _unify_path(url_entry)
    result, entries = omni.client.stat(url_entry_unified)
    if result != omni.client.Result.OK:
        raise ValueError(f"The remote path {url_entry_unified} is not valid")
    return entries.size == 0


def _list_asset_entries(url_entry: str) -> List[Tuple[str, int | None]]:
    """
    List all the items in the url_entry as (url, size) pairs.

    When it is a folder, it will return all the items in the folder. When it is a file, it will return a list
    with the file itself. The size is None when the backend does not report one, which is the case for
    Nucleus / omni.client listings.
    """
    # Resolve S3 files directly so an exact object is not mistaken for a directory,
    # and verification gets its size. Preserve the omni.client path for other backends.
    use_s3 = not _is_import_ready("omni.client") and _is_s3_environment()
    if not use_s3 and not _is_url_folder(url_entry):
        return [(_unify_path(url_entry), None)]

    if use_s3 or not _is_import_ready("isaacsim.storage.native.nucleus"):
        if not _is_s3_environment():
            if _is_import_ready("omni.client"):
                return _list_omni_client_entries(_unify_path(url_entry))
            raise ValueError(_NUCLEUS_CLIENT_REQUIRED)
        # Fallback for S3 environments: boto3
        try:
            bucket, key = _parse_s3_url(url_entry)
            s3_client = _get_s3_client()

            if key and not key.endswith("/"):
                try:
                    metadata = s3_client.head_object(Bucket=bucket, Key=key)
                    return [(url_entry, metadata["ContentLength"])]
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                        raise

            # Ensure key ends with '/'
            if key and not key.endswith("/"):
                key = key + "/"

            # List all objects with this prefix
            entries: List[Tuple[str, int | None]] = []
            paginator = s3_client.get_paginator("list_objects_v2")

            # Get files with retry logic for rate limiting
            max_retries = 5
            retry_count = 0
            backoff_time = 1  # Start with 1 second

            while True:
                try:
                    # Get files
                    for page in paginator.paginate(Bucket=bucket, Prefix=key):
                        if "Contents" in page:
                            for obj in page["Contents"]:
                                obj_key = obj["Key"]
                                if obj_key == key or obj_key.endswith("/"):
                                    continue
                                obj_url = (
                                    f"https://{bucket}.s3-{_S3_REGIONS.get(_get_asset_env())}.amazonaws.com/{obj_key}"
                                )
                                entries.append((obj_url, obj.get("Size")))
                    break  # Success - exit the retry loop
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code == "SlowDown" and retry_count < max_retries:
                        # If we hit a rate limit, wait with exponential backoff
                        retry_count += 1
                        sleep_time = backoff_time * (1 + 0.5 * (2.0**retry_count))
                        print(f"Rate limit hit. Retrying... (attempt {retry_count}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        # If error is not rate limiting or we're out of retries, raise the exception
                        raise ValueError(f"Failed to list S3 objects at {url_entry}: {str(e)}")

            return entries
        except Exception as e:
            raise ValueError(f"Failed to list S3 objects at {url_entry}: {str(e)}")

    from isaacsim.storage.native.nucleus import _list_files

    # _list_files is an async function
    _, nucleus_entries = asyncio.run(_list_files(url_entry))
    return [(entry, None) for entry in nucleus_entries]


def _list_omni_client_entries(folder_url: str) -> List[Tuple[str, int | None]]:
    """List every file under a Nucleus folder with the standalone omni.client, outside Isaac Sim."""
    import omni.client

    result, items = omni.client.list(folder_url)
    if result != omni.client.Result.OK:
        raise ValueError(f"Failed to list {folder_url}: {result}")

    entries: List[Tuple[str, int | None]] = []
    for item in items:
        item_url = f"{folder_url.rstrip('/')}/{item.relative_path}"
        is_file = item.flags & omni.client.ItemFlags.READABLE_FILE
        if item.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN and not is_file:
            entries.extend(_list_omni_client_entries(item_url))
        else:
            entries.append((item_url, item.size))
    return entries


def _list_asset_url(url_entry: str) -> List[str]:
    """
    List all the items in the url_entry. When it is a folder, it will return all the items in the folder.
    When it is a file, it will return a list with the file itself.
    """
    return [url for url, _ in _list_asset_entries(url_entry)]


def _filter_downloaded_assets(
    url_entries: List[str], local_dir: str, version: str | None = None, hash: str | None = None, verbose: bool = False
) -> List[str]:
    """
    Filter the url_entries to only include the ones that are not downloaded.

    Args:
        url_entries: The url entries to filter.
        local_dir: The local directory to check for downloaded assets.
        version: The version of the asset.
        hash: The sha256 hash of the asset.

    Returns:
        The filtered url entries.
    """
    results = []
    # we will check if the asset is already downloaded
    for entry_url in url_entries:
        local_path = os.path.join(local_dir, _get_asset_relpath(entry_url, version, hash))
        if os.path.isfile(local_path):
            if verbose:
                print(f"Asset already downloaded to: {local_path}. Skipping download.")
        else:
            results.append(entry_url)
    return results


class _S3Downloader:
    """
    One s3transfer TransferManager shared by the download pool, so that the transfers still in flight when
    the budget runs out can be aborted rather than abandoned.

    ``s3_client.download_file`` hides its transfer inside a private TransferManager that nothing can cancel.
    Abandoned there, it deadlocks interpreter exit: ``concurrent.futures`` stops s3transfer's worker threads
    at exit and then joins the pool thread that is still waiting on them.
    """

    def __init__(self, concurrency: int):
        from boto3.s3.transfer import TransferConfig, create_transfer_manager

        # download_file gives every file its own manager with boto3's default of 10 request threads; keep that
        # per-file headroom now that the manager is shared.
        config = TransferConfig(max_concurrency=10 * concurrency)
        self._manager = create_transfer_manager(_get_s3_client(), config)
        self._futures = []
        self._cancelled = False
        self._lock = threading.Lock()

    def download(self, bucket: str, key: str, local_path: str):
        # Submitting under the lock means cancel() cannot miss a transfer that starts while it runs.
        with self._lock:
            if self._cancelled:
                raise RuntimeError("Download cancelled")
            future = self._manager.download(bucket, key, local_path)
            self._futures.append(future)
        future.result()

    def cancel(self):
        """Abort the in-flight transfers. s3transfer deletes their partial temp files."""
        with self._lock:
            self._cancelled = True
            for future in self._futures:
                future.cancel()

    def close(self):
        """Wait for the transfers to wind down and stop the manager's threads."""
        # Not shutdown(cancel=True): s3transfer passes its arguments to _shutdown in the wrong order there,
        # which raises TypeError as soon as a transfer is actually in flight.
        self._manager.shutdown()


def _download_individual_asset(
    url_entry: str,
    download_dir: str,
    version: str | None = None,
    hash: str | None = None,
    s3_downloader: _S3Downloader | None = None,
):
    version = version if version is not None else get_i4h_asset_version()

    if url_entry.endswith("/"):
        return None

    local_path = os.path.join(download_dir, _get_asset_relpath(url_entry, version, hash))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not _is_import_ready("omni.client"):
        if not _is_s3_environment():
            raise ValueError(_NUCLEUS_CLIENT_REQUIRED)
        # Fallback for S3 environments: boto3
        try:
            bucket, key = _parse_s3_url(url_entry)

            # Download file directly to local path
            if os.path.exists(local_path):
                os.remove(local_path)

            # Both write to a temp file and only rename it into place once the download is complete.
            if s3_downloader is not None:
                s3_downloader.download(bucket, key, local_path)
            else:
                _get_s3_client().download_file(bucket, key, local_path)
            return local_path
        except Exception as e:
            message = f"Failed to download asset from S3: {url_entry}: {str(e)}"
            env = _get_asset_env()
            error_code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else None
            is_not_found = error_code in ("404", "NoSuchKey")
            if is_not_found and env != "production":
                # Only production mirrors every published catalog; the other servers hold partial content.
                message += (
                    f". The '{env}' asset server may not host this catalog version; "
                    "unset I4H_ASSET_ENV (or set it to 'production') to use the public production catalog."
                )
            raise ValueError(message)

    import omni.client

    result, _, file_content = omni.client.read_file(url_entry)
    if result != omni.client.Result.OK:
        raise ValueError(f"Failed to download asset: {url_entry}")

    # Write to a temp file and rename it into place, so an interrupted write never leaves a truncated file
    # under the final name, where it would be skipped as already downloaded.
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(local_path), prefix=f".{os.path.basename(local_path)}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_content)
        os.replace(temp_path, local_path)
    except BaseException:
        os.remove(temp_path)
        raise

    return local_path


DEFAULT_DOWNLOAD_CONCURRENCY = 2
DEFAULT_DOWNLOAD_TIMEOUT = 3600.0


def _download_assets(
    url_entries: List[str],
    download_dir: str,
    version: str | None = None,
    hash: str | None = None,
    concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
):
    """
    Download assets from url entry sources to local directory.

    Args:
        url_entries: The url entries to download.
        download_dir: The directory to download the asset to. Default is the cache directory.
        version: The version of the asset.
        hash: The sha256 hash of the asset.
        concurrency: The number of concurrent downloads.
        timeout: Budget in seconds for the whole download. Once it runs out the queued downloads are
            cancelled and TimeoutError is raised; files already fetched are left in place.

    Returns:
        The path to the local asset.
    """
    version = version if version is not None else get_i4h_asset_version()
    total = len(url_entries)
    completed = 0

    # Only the boto3 fallback can abort an in-flight file; omni.client.read_file cannot be interrupted.
    use_s3 = not _is_import_ready("omni.client") and _is_s3_environment()
    s3_downloader = _S3Downloader(concurrency) if use_s3 else None
    finished = False

    # Deliberately not a `with` block: its __exit__ shuts the pool down with wait=True, which on timeout
    # would keep draining the queue and block until every remaining file had downloaded anyway.
    executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures_to_url = {
            executor.submit(
                _download_individual_asset, url_entry, download_dir, version, hash, s3_downloader=s3_downloader
            ): url_entry
            for url_entry in url_entries
        }

        # Use tqdm to show progress
        with tqdm(total=total, desc=f"Downloading assets to {download_dir}", unit="files") as pbar:
            for future in as_completed(futures_to_url, timeout=timeout):
                future.result()
                completed += 1
                pbar.update(1)
        finished = True
    # Only an alias of the builtin from Python 3.11 onwards, so catch it under its own name.
    except FuturesTimeoutError as e:
        raise TimeoutError(
            f"Downloaded {completed} of {total} files before the {timeout:g}s budget ran out. Finished files "
            "are kept, so re-running continues from here; pass a larger --timeout to fetch the rest in one go."
        ) from e
    finally:
        # cancel_futures drops the downloads still queued so as_completed's timeout ends the transfer
        # instead of only ending the wait for it.
        executor.shutdown(wait=False, cancel_futures=True)
        if s3_downloader is not None:
            # The files already in flight have to be aborted too. Left running, their pool threads would
            # block interpreter exit forever once concurrent.futures stops s3transfer's workers.
            if not finished:
                s3_downloader.cancel()
            s3_downloader.close()


def retrieve_asset(
    version: str | None = None,
    download_dir: str | None = None,
    sub_path: str | None = None,
    hash: str | None = None,
    force_download: bool = False,
    verbose: bool = False,
    concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
) -> str:
    """
    Download the asset from the remote path to the download directory.

    Args:
        version: The version of the asset to download.
        download_dir: The directory to download the asset to.
        sub_path: The sub path of the asset to download.
        hash: The sha256 hash of the asset.
        force_download: If True, the asset will be downloaded even if it already exists.
        verbose: If True, it will print more information.
        concurrency: The number of concurrent downloads.
        timeout: Budget in seconds for the whole download. A full catalog fetch can take well over an
            hour on a slow link, so raise this rather than letting it abort part-way.
    Returns:
        The path to the local asset.
    """
    version = version if version is not None else get_i4h_asset_version()
    local_dir = get_i4h_local_asset_path(version, download_dir, hash)
    remote_path = get_i4h_asset_path(version, hash)

    if sub_path is not None:
        remote_path = remote_path + "/" + sub_path

    paths = _list_asset_url(remote_path)
    if not paths:
        # Reporting success with nothing downloaded hides a mistyped or unsupported sub-path from scripts.
        raise FileNotFoundError(f"No asset files found at {remote_path}")

    if force_download:
        url_entries = paths
    else:
        url_entries = _filter_downloaded_assets(paths, local_dir, version, hash, verbose)

    if len(url_entries) > 0:
        _download_assets(url_entries, local_dir, version, hash, concurrency=concurrency, timeout=timeout)

    return local_dir


class BaseI4HAssets:
    """
    Base class for i4h assets with public attributes defining the relative paths to the assets.

    When accessing any public attribute of this class, the corresponding asset will be automatically downloaded
    if it doesn't exist locally. For USD files (with extensions .usd, .usda, or .usdc), the entire directory
    containing the file will be downloaded to ensure all dependencies are available. If skip_download_usd is True,
    USD files will use the remote path directly without downloading.

    The downloaded assets will be stored in the specified download directory (defaults to ~/.cache/i4h-assets).
    """

    def __init__(
        self, download_dir: str | None = None, skip_download_usd: bool = False, recursive_file_check: bool | None = None
    ):
        """
        Initialize the assets

        Args:
            download_dir: The directory to download the assets to
            skip_download_usd: If True, it will always use the USD file from the remote asset path. Default is False.
            recursive_file_check: If True, it will check recursively into the folder and see if every file is ready.
                When the environment is not S3, it will be set to False automatically, otherwise it will be set to True.
        """
        self._remote_asset_path = get_i4h_asset_path()
        self._download_dir = _get_download_dir() if download_dir is None else download_dir
        self._skip_download_usd = skip_download_usd
        if recursive_file_check is None:
            self._recursive_file_check = _is_s3_environment()
        else:
            self._recursive_file_check = recursive_file_check

    def __getattribute__(self, name):
        """Override to print a message when any attribute is accessed."""
        if name.startswith("_"):
            # skip the private attributes
            return super().__getattribute__(name)

        value = super().__getattribute__(name)
        if Path(value).suffix in [".usd", ".usda", ".usdc"]:
            if self._skip_download_usd:
                # return the remote asset path and skip the download
                return os.path.join(self._remote_asset_path, value)
            else:
                # A single USD file can depend on other files in the same directory.
                # Need to download the directory instead.
                _value = os.path.dirname(value)
        else:
            _value = value

        if not self._recursive_file_check:
            local_path = get_i4h_local_asset_path(download_dir=self._download_dir)
            local_sub_path = os.path.join(local_path, value)
            if os.path.isdir(local_sub_path) or os.path.isfile(local_sub_path):
                return local_sub_path

        # trigger download of the asset
        local_path = retrieve_asset(download_dir=self._download_dir, sub_path=_value)
        return os.path.join(local_path, value)
