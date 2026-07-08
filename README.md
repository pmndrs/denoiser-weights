# denoiser-weights

Pre-converted **ONNX** models for [`pmndrs/denoiser`](https://github.com/pmndrs/denoiser) —
[Intel Open Image Denoise (OIDN)](https://github.com/RenderKit/oidn) U-Nets,
built directly from the official OIDN `.tza` weights.

**46 models**: every OIDN RT + lightmap variant × fp32 (`.onnx`) + fp16
(`.fp16.onnx`). A denoiser instance fetches exactly ONE file for its
configuration (0.6–15 MB) — never the set.

Plus **20 aux split-graph artifacts** (`models-v2+`) — see
[Aux split-graph artifacts](#aux-split-graph-artifacts) — for the 9-channel
cleanAux denoisers, which the default runtime fetches automatically.

## Serving / usage

Files are committed as plain git blobs (no LFS) specifically so
**jsDelivr's GitHub endpoint** can serve them version-pinned, CORS-enabled:

```
https://cdn.jsdelivr.net/gh/pmndrs/denoiser-weights@models-v1/models/rt_ldr_small.fp16.onnx
```

This is the `denoiser` package's default `weightsUrl`. For production apps we
recommend self-hosting: copy `models/` into your static assets and pass
`Denoiser.create({ weightsUrl: '/models' })`.

## Versioning

`models-vN` tags are **immutable** — jsDelivr caches tagged URLs permanently.
If weights are ever regenerated (upstream retrain, converter change), a new
`models-v(N+1)` tag is cut and the `denoiser` release that consumes it bumps
its default. Never re-point an existing tag.

| tag | contents |
|---|---|
| `models-v1` | OIDN weights as of upstream `RenderKit/oidn-weights` master 2026-06 (byte-identical since ~2.x), converted with `tools/onnx-convert`, opset 17, dynamic batch/H/W dims, final activation relu6 |
| `models-v2` | `models-v1` **unchanged** + 20 aux split-graph artifacts (`*.tail.onnx` / `*.enc0.bin`) for the 10 cleanAux models. No model file changed — purely additive. |

## Aux split-graph artifacts

onnxruntime-web's WebGPU execution provider miscomputes the **first Conv that
reduces the raw >3-channel input** — so the 9-channel cleanAux denoisers
(`rt_{hdr,ldr}_calb_cnrm{,_small,_large}`) speckle on WebGPU while every other
model is fine (repro + isolation: `pmndrs/denoiser` →
`tools/ort-webgpu-aux-repro`, `tools/ort-webgpu-aux-split`). The fix runs that one
conv in a WGSL kernel and the rest on ORT-WebGPU. Each cleanAux model ships two
sidecar files (same precision suffix as the model):

- `<name>[.fp16].tail.onnx` — the network from the second conv on; two inputs
  `[<first-conv-relu6-output>, input]` (the input skip still needs the raw image).
- `<name>[.fp16].enc0.bin` — that first conv's weights, **f32** OIHW
  `[COUT, 9, 3, 3]` then bias `[COUT]` (`COUT` = 32 base/small, 64 large).

`denoiser`'s `splitAux` (default on) fetches these next to the model and picks the
right precision automatically; if absent it falls back to the plain model. When
the upstream ORT bug is fixed these become unnecessary (drop them / set
`splitAux:false`) — the models themselves are already correct (54 dB vs native on
the CPU/WASM provider). Regenerate with `pmndrs/denoiser` →
`tools/aux-split-artifacts/generate.py`.

## Regenerating

The converter lives in the main repo —
[`pmndrs/denoiser` → `tools/onnx-convert`](https://github.com/pmndrs/denoiser/tree/main/tools/onnx-convert)
(Python: numpy + onnx, no PyTorch). It parses `.tza`, builds the graph 1:1 with
the reference implementation, and verifies NumPy-vs-onnxruntime parity (~1e-6):

```sh
python convert.py path/to/tzas/*.tza -o models/          # fp32
python convert.py path/to/tzas/*.tza -o models/ --fp16   # fp16
```

## License

The network weights are © Intel Corporation, **Apache-2.0** (see [LICENSE](LICENSE)) —
this repository redistributes them in converted form. The conversion tooling is
MIT (main repo).
