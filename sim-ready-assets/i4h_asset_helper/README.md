# I4H Assets Catalog Helper

## Requirements

- Python 3.10

### Installation

The helper itself needs Python 3.10+ and does not require Isaac Sim for the
public S3 catalogs (`staging` / `production`). From this directory:

```bash
uv sync
```

That installs `boto3`, `tqdm`, `requests`, and pytest. On Linux aarch64 (for
example DGX Spark) this is the supported path: Isaac Sim has no ARM64 pip
wheels.

The default catalog is on the internal dev Nucleus server (`I4H_ASSET_ENV=dev`),
which needs `omni.client`. Install the `nucleus` extra to get the standalone
`omniverseclient` package (Python 3.10–3.12; NVIDIA proprietary license):

```bash
uv sync --extra nucleus
```

The first Nucleus request prints a device-login URL and code; open the URL and
enter the code to sign in to `isaac-dev.ov.nvidia.com`.

Isaac Sim is an alternative way to provide `omni.client`, and is otherwise
only needed for `--force_omni_client` without the extra. It is not a package
extra: its pip wheel stub cannot be locked from aarch64. On Linux x86_64 with
Python 3.10, install it into the environment separately:

```bash
uv sync --python 3.10
uv pip install --python 3.10 --extra-index-url https://pypi.nvidia.com "isaacsim[all,extscache]"
```

Alternatively, use conda as described in the [IsaacSim Pip Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html#installation-using-pip):

```bash
conda create -n i4h-assets python=3.10
conda activate i4h-assets
git clone https://github.com/isaac-for-healthcare/i4h-asset-catalog.git
cd i4h-asset-catalog
pip install -e .
pip install --extra-index-url https://pypi.nvidia.com "isaacsim[all,extscache]"
```

### Usage

To download the asset to a local directory `~/.cache/i4h-assets/<VERSION>` (dev Nucleus) or
`~/.cache/i4h-assets/<SHA256_HASH>` (S3):

### v0.9.0 asset release

The default catalog is `v0.9.0` on the dev Nucleus server:
`omniverse://isaac-dev.ov.nvidia.com/Library/IsaacHealthcare/v0.9.0/{Props,Robots,Test}/...`,
with no build directory. It downloads to `~/.cache/i4h-assets/v0.9.0`. This
complete catalog includes the corrected OR editor icon and repaired organ
materials (NVBugs 6646936 and 6657585).

`assets_sha256.json` records the build ID `c02e5b5e3d71`, the first 12
characters of the catalog's SHA-256 digest. It is not a content digest, so
`--verify` checks the download against the Nucleus listing, including partial
downloads. If the catalog is later published to S3 under that build directory,
`I4H_ASSET_ENV=production` selects `v0.9.0/c02e5b5e3d71`.

Older catalogs are on S3: use `I4H_ASSET_ENV=production` with `--version 0.7.0`
or `I4H_ASSET_VERSION=0.7.0`.

```bash
i4h-asset-retrieve --sub-path Props/shared_OR_without_Mark --verify
```

The staged `main.usd` is already repaired; no local repair command is needed
for `v0.9.0`. The opt-in utility below remains useful for older catalogs.

### Python Usage

```python
from i4h_asset_helper import BaseI4HAssets

class MyAssets(BaseI4HAssets):
    """Assets manager for the your workflow."""
    dVRK_ECM = "Robots/dVRK/ECM/ecm.usd"


my_assets = MyAssets()

# When you use the asset, it will check if the asset is downloaded.
# If not, it will download the asset to the default download directory.
print(my_assets.dVRK_ECM)
```

#### CLI Usage

```bash
i4h-asset-retrieve [-h] [--version ] [--force] [--download-dir DOWNLOAD_DIR] [--sub-path SUB_PATH] [--hash HASH] [--force_omni_client]
                   [--skip-download] [--verify] [--concurrency CONCURRENCY] [--timeout TIMEOUT]
```

##### Options

- `-h, --help`: Show help message and exit
- `--version`: Asset version to retrieve (default: the latest version)
- `--force`: Force download even if assets already exist (default: False)
- `--download-dir DOWNLOAD_DIR`: Directory to download assets to (default: ~/.cache/i4h-assets)
- `--sub-path SUB_PATH`: Either a subfolder path or a subfile path under the asset catalog. Only support a single path, like `Robots` or `Robots/Franka` (default: None)
- `--hash HASH`: Hash of the asset to retrieve (default: None)
- `--force_omni_client`: Force use of omni.client. (default: False)
- `--skip-download`: Skip downloading and only verify existing assets (default: False)
- `--verify`: Verify the download once it finishes (default: False). See [Verification](#verification).
- `--concurrency CONCURRENCY`: Number of concurrent downloads (default: 2)
- `--timeout TIMEOUT`: Seconds allowed for the whole download (default: 3600). A full catalog fetch is over 6000 files and can take a couple of hours on a slow link, so raise this rather than letting it abort part-way.

##### Example

```bash
# Download a specific subfolder of assets
i4h-asset-retrieve --sub-path Robots

# Download assets with a specific hash
i4h-asset-retrieve --hash abc123def456

# Force re-download of assets to a custom directory
i4h-asset-retrieve --force --download-dir ~/my-assets

# Fetch the whole catalog with a three-hour budget, then check it arrived intact
i4h-asset-retrieve --timeout 10800 --concurrency 8 --verify

# Check an existing download without re-fetching it
i4h-asset-retrieve --skip-download --verify --sub-path Robots
```

##### Verification

What `--verify` can check depends on what the catalog version publishes in `assets_sha256.json`:

- Versions up to `0.3.0` publish a SHA-256 of the whole published tree, so the download is hashed and compared against it.
- Versions from `0.5.0` onwards publish the catalog's short commit id instead (for example `724f82e`), which names the S3 prefix the assets are served from and the local directory they land in. That value describes where the assets live, not what they contain, so there is no digest to compare against. These versions are instead verified against the catalog listing: every object the catalog holds must be present locally at its full size.

Listing verification needs network access and is the check that catches a partial download, which is the usual failure when a fetch is interrupted or exceeds `--timeout`. Pass `--sub-path` alongside `--verify` to scope it to the sub path you downloaded; without it, verification expects the entire catalog.

### Operating-room node-graph icon repair

Catalog `0.7.0/724f82e` contains an unresolved
`core_definitions.file_texture.png` reference in
`Props/shared_OR_without_Mark/main.usd`. It is the
`ui:nodegraph:node:icon` default on the bed-sheet material's `file_texture`
node graph, not a material texture input. A complete download can therefore
pass `--verify` while USD dependency validation still reports this missing icon.

An opt-in repair clears only that known icon default in a new sibling layer.
Install `usd-core` (validated with `25.11`) or use Isaac Sim's USD runtime:

```bash
uv pip install usd-core==25.11
scene="$HOME/.cache/i4h-assets/724f82e/Props/shared_OR_without_Mark"
python -m i4h_asset_helper.repair_or_scene "$scene/main.usd" "$scene/main.repaired.usd"
```

Open `main.repaired.usd` to use the corrected scene. Keep it beside `main.usd`
so relative asset paths continue to resolve. The source is preserved, existing
output files are refused, and an unexpected icon value aborts the repair.
No rendering inputs, connections, or material library references are changed.

With plain USD 25.11, `UsdUtils.ComputeAllDependencies` reports seven unresolved
paths before this repair and six afterward. The remaining names are
`OmniEmissive.mdl`, `OmniGlass.mdl`, `OmniPBR.mdl`, `OmniSurface.mdl`,
`nvidia/aux_definitions.mdl`, and `nvidia/core_definitions.mdl`; the reported
Isaac Sim environment supplies these through its MDL search path. Do not ignore
arbitrary missing `.mdl` files when validating other scenes.

The `v0.9.0` release applies this same repair to the authored scene before
publication. The asset binaries are hosted outside this repository; this utility
does not upload assets or silently alter downloads. The node uses its fallback
UI icon. Validate the staged release in Isaac Sim before production publication.

Run the USD regression tests with `usd-core` installed:

```bash
python -m pytest i4h_asset_helper/tests/test_repair_or_scene.py
```

### Environment Variables

#### I4H_ASSET_DOWNLOAD_DIR

- You can set the `I4H_ASSET_DOWNLOAD_DIR` environment variable to the directory to download assets to.
- The default directory is `~/.cache/i4h-assets`.
- A subfolder named after the version (dev Nucleus) or the hash of the asset (S3) will be created in this directory.

#### I4H_ASSET_ENV

- There are three different asset server environments: `dev`, `staging`, and `production`. `staging` and `production` are publicly accessible and `dev` is only accessible by the internal team. You can set the `I4H_ASSET_ENV` environment variable to `dev`, `staging`, or `production` to use the corresponding asset server.
- The current default environment is `dev`, which serves the `v0.9.0` catalog. Set `I4H_ASSET_ENV=production` for the public S3 catalogs, including all versions before `v0.9.0`.
- The `dev` environment needs `omni.client`, from the `nucleus` extra or Isaac Sim. On the first run, open the printed device-login URL in a web browser and enter the code to authenticate.

#### I4H_ASSET_SHA256_HASH

- Identifies which build of the asset catalog to retrieve. Despite the name, this is a SHA-256 digest only for versions up to `0.3.0`; later versions use a short build identifier (a catalog commit ID or a content-digest prefix), which selects the remote prefix and the local directory name. See [Verification](#verification).
- You can set the `I4H_ASSET_SHA256_HASH` environment variable to the identifier of the asset to retrieve.
- When you use the CLI, you can use the `--hash` argument to specify the identifier.
  - Priority order: CLI argument > environment variable > `assets_sha256.json` file in the `i4h_asset_helper` package.
