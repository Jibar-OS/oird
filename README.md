# oird — OIR native inference daemon

Native C++ daemon that owns model residency and inference for the Open Intelligence Runtime. Runs as a system service under its own SELinux domain (`u:r:oird:s0`), registers `oir_worker` with `servicemanager`, and serves requests from `OIRService` over AIDL.

## What it does

- Loads LLM / VLM / ONNX / whisper models on demand.
- Shares loaded models across every app that asks for the same capability — one copy in memory, N callers.
- Pools inference contexts per model (`ContextPool` for llama-backed, `WhisperPool` for whisper) with priority-aware wait queues.
- Accounts KV-cache memory in the resident budget so eviction decisions are accurate.
- Dispatches across backends (llama.cpp, whisper.cpp, ONNX Runtime, libmtmd) based on capability.

## Tree location

Installs as `/system_ext/bin/oird` via `prebuilt_etc`. Lives at `system/oird/` in the AOSP tree.

## Building

oird is built as part of a JibarOS tree:

```bash
cd ~/aaosp
source build/envsetup.sh
lunch aosp_cf_x86_64_phone-trunk_staging-userdebug
m -j8 oird
```

## Dependencies

- AIDL interfaces from [`oir-framework-addons`](https://github.com/jibar-os/oir-framework-addons)
- [`platform_external_llamacpp`](https://github.com/jibar-os/platform_external_llamacpp)
- [`platform_external_whispercpp`](https://github.com/jibar-os/platform_external_whispercpp)
- [`platform_external_onnxruntime`](https://github.com/jibar-os/platform_external_onnxruntime)

## See also

[`github.com/jibar-os/docs`](https://github.com/jibar-os/docs) for architecture + capability model.

## Migration status

🚧 Code migration in progress.
