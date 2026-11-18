> **AI authorship:** Created by OpenAI Codex. **Model:** the exact runtime model identifier was not exposed to the assistant.

# Use a design from Python, the CLI, HTTP or MCP

A design identifies its generator and contains the settings for one experiment. The four interfaces accept the same settings and preserve the seed, resolved design, and output files. Run the commands below in a Python 3.12+ environment. To install from a clone, replace `fovea-synthdata` with `.` inside the install argument.

## Python and CLI

```sh
python -m pip install 'fovea-synthdata[particle_plate,cell_plate]'
synthdata list
synthdata describe particle_plate
synthdata preset list cell_plate
synthdata preset export particle_plate/small -o my-plate.yaml
synthdata preview my-plate.yaml -o preview.png
synthdata generate my-plate.yaml -o my-data --writer tiff
synthdata check my-data
```

`describe` returns the available presets, writers, operations and JSON Schema for the generator's settings. A base install supports discovery, validation, and preset export. Install the matching generator extra for rendering. The optional `report` extra adds `synthdata report DATASET` for particle plates.

```python
from synthdata import sdk

print(sdk.list_generators())
catalog = sdk.describe_generator("particle_plate")
print(catalog["presets"], catalog["writers"])
design = sdk.load_preset("particle_plate/small")
design["conditions"]["large"]["diameter"]["mean"] = 24
settings = sdk.validate_design(design)
sdk.preview(settings).save("my-preview.png")
manifest = sdk.generate(settings, "python-data", writer="tiff")
print(manifest.design_sha256, manifest.extra["n_images"])
print(sdk.check("python-data").passed)
```

Each output directory belongs to one dataset. Generation writes the manifest last and refuses a nonempty output directory by default. The manifest records file hashes, seed, writer and generator version. Use it to compare results across interfaces with identical designs and software versions.

## HTTP

```sh
python -m pip install 'fovea-synthdata[particle_plate,cell_plate,api]'
SYNTHDATA_OUTPUT_DIR=./runs synthdata-api --host 127.0.0.1 --port 8000
```

Open the interactive API reference at `http://127.0.0.1:8000/docs`. From a second terminal, run this example with `curl` and `jq`. It edits the same particle diameter, downloads a preview, submits a TIFF run, waits for its manifest, and prints the recoverability verdict.

```sh
set -e
api=http://127.0.0.1:8000
curl -fsS "$api/generators/particle_plate/presets/small" |
  jq '.conditions.large.diameter.mean = 24 | {design: .}' > request.json
curl -fsS "$api/validate" --json @request.json | jq '.valid'
curl -fsS "$api/preview" --json @request.json -o http-preview.png
id=$(jq '. + {writer:"tiff"}' request.json | curl -fsS "$api/runs" --json @- | jq -er .id)
while true; do
  state=$(curl -fsS "$api/runs/$id")
  case "$(echo "$state" | jq -r .status)" in
    succeeded) break ;;
    failed) echo "$state" | jq -r .error >&2; exit 1 ;;
    *) sleep 1 ;;
  esac
done
curl -fsS "$api/runs/$id/artifacts/manifest.json" -o manifest.json
curl -fsS -X POST "$api/runs/$id/check" | jq .passed
```

The status response lists relative artifact paths. Append a listed path to `/runs/{id}/artifacts/` to download that file. A completed check adds its reports to the artifact list. Generation returns HTTP 202 with a run ID. Invalid settings return 422, unfinished datasets return 409, and a busy queue returns 429.

## MCP over stdio

Install `fovea-synthdata[particle_plate,cell_plate,mcp]` into a dedicated environment. Configure the MCP client to launch its installed executable. For example, substitute the absolute paths for your machine in this client configuration.

```json
{
  "mcpServers": {
    "fovea-synthdata": {
      "command": "/absolute/path/to/.venv/bin/synthdata-mcp",
      "env": {"SYNTHDATA_OUTPUT_DIR": "/absolute/path/to/runs"}
    }
  }
}
```

Try this request in the connected client:

> Load the small particle plate preset. Change the mean diameter for the large condition to 24 pixels and show me the preview. Generate TIFFs, check the dataset, and return the manifest.

| Task | MCP tool |
| --- | --- |
| Discover generators and settings | `list_generators`, `describe_generator` |
| Load and check editable settings | `get_preset`, `validate_design` |
| Preview without generating a dataset | `preview_design` |
| Start and follow a dataset | `generate_dataset`, `get_run` |
| Check or retrieve the result | `check_dataset`, `get_artifact` |

Use the ID returned by `generate_dataset` for the remaining calls. `get_run` pages artifact names, and `get_artifact` returns a local file URI and base64 bytes. Pass its `next_offset` back to retrieve the rest of a large file.

## Running the services

HTTP and MCP use the same on-disk run layout. Set `SYNTHDATA_OUTPUT_DIR` to share a run directory or choose separate roots. The default is `~/.local/share/fovea-synthdata/runs`. Completed datasets persist across restarts. Each server renders one job at a time. Local users administer storage and can remove completed run directories when finished.

HTTP listens on loopback by default. The service assumes a trusted caller with access to every run in its output root. For a shared deployment, put authentication and access controls in front of it and provision storage for the selected designs.
