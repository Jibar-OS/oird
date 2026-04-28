# oird — Open Intelligence Runtime Daemon

Native C++ inference daemon for AOSP. Owns model residency, dispatch, and concurrency for the [Open Intelligence Runtime](https://github.com/Jibar-OS/JibarOS) — LLM, VLM, ASR, TTS, vision, OCR, and audio-classification capabilities served to apps via a single AIDL surface.

Runs in its own process under SELinux domain `u:r:oird:s0`. Native crashes can't take down `system_server` or other platform services.

## Why a daemon?

Most on-device ML on Android today is per-app: every app links its own copy of llama.cpp / whisper.cpp / ONNX Runtime, loads its own copy of every model, and competes with every other app for memory. A 7B LLM duplicated across three apps is 12 GB of RAM nobody actually needs.

oird flips that: **one daemon, one copy of each model, shared across all callers.** App #2 asking for the same `text.complete` model gets the already-loaded handle. App #3 transcribing audio runs concurrently through a per-model context pool. When budget pressure hits, eviction is one LRU decision against one shared budget — not N independent ones.

Crash isolation is the other half: an OOM in inference, a malformed GGUF, a corrupt ONNX — none of it touches `system_server`. The daemon respawns under init; apps re-attach.

## What it does

12 capabilities across 4 backends, all served behind a single AIDL worker (`oir_worker`):

| Capability         | Backend         | Default model                  |
|--------------------|-----------------|--------------------------------|
| `text.complete`    | llama.cpp       | Qwen 2.5 0.5B Q4_K_M           |
| `text.embed`       | llama.cpp       | all-MiniLM-L6 Q8_0 (384-dim)   |
| `text.translate`   | llama.cpp       | (reuses text.complete)         |
| `text.classify`    | ONNX Runtime    | OEM-baked (HF tokenizer sidecar) |
| `text.rerank`      | ONNX Runtime    | OEM-baked (cross-encoder)      |
| `audio.transcribe` | whisper.cpp     | whisper-tiny.en Q5             |
| `audio.synthesize` | ONNX Runtime    | Piper VITS (G2P sidecar)       |
| `audio.vad`        | ONNX Runtime    | Silero VAD                     |
| `vision.detect`    | ONNX Runtime    | RT-DETR-r50vd                  |
| `vision.embed`     | ONNX Runtime    | SigLIP-base-patch16-224 (768-dim) |
| `vision.ocr`       | ONNX Runtime    | OEM-baked (det+rec+vocab triplet) |
| `vision.describe`  | libmtmd + llama | LLaVA-family (OEM-baked)       |

Models are mmap'd from `/product/etc/oir/` (platform defaults) or `/vendor/etc/oir/` (OEM overrides). Each capability is OEM-tunable via `oir_config.xml` knobs (15+ supported — see [`KNOBS.md`](https://github.com/Jibar-OS/JibarOS/blob/main/docs/KNOBS.md) in JibarOS).

## Architecture

```
                       AIDL (oir_worker)
                              │
  ┌───────────────────────────┴────────────────────────────┐
  │                  OirdService (AIDL stub)                │
  │              thin router; all bodies in backends         │
  └──┬──────────────┬──────────────┬───────────────────┬───┘
     │              │              │                   │
     ▼              ▼              ▼                   ▼
  LlamaBackend  WhisperBackend  OrtBackend         VlmBackend
  text.*        audio.transcribe audio.{vad,synth} vision.describe
                                vision.{det,emb,
                                  ocr}, text.{cls,
                                  rerank}
     │              │              │                   │
     └──────┬───────┴──────────────┴───────────────────┘
            │
            ▼
   ┌────────────────────────────────────────┐
   │ Runtime — cross-cutting state           │
   │   mLock, mModels handle table           │
   │   Budget (resident bytes + LRU)         │
   │   Scheduler (priority-aware workers)    │
   │   LoadRegistry (concurrent-load dedup)  │
   │   Resource registry (eviction policy)   │
   └────────────────────────────────────────┘
```

Each backend owns its capability bodies, knobs, and per-handle pool state. Cross-cutting concerns live in `Runtime` and are accessed by reference. `OirdService` itself is a thin AIDL router: every method is a one-line forward to `mLlama` / `mWhisper` / `mOrt` / `mVlm`.

### Concurrency

- **Inbound binder calls** dispatch on a small pool (4-8 threads, scaled with cores). Most calls return fast: enqueue to Scheduler and return.
- **Inference work** happens on a `Scheduler` with N workers (`clamp(hardware_concurrency, 4, 16)`). Priorities are POSIX-niceness style: `audio.*` preempts `text.*` on shared queues.
- **Within a single model**, multiple concurrent submissions interleave through a per-handle `ContextPool` (llama / VLM) or `WhisperPool` (whisper). ORT sessions are thread-safe by design.
- **Cancellation** is plumbed end-to-end: `ICancellationSignal` from the SDK reaches the worker thread which checks an atomic flag between tokens / windows / batches.

### Memory accounting

`Budget` tracks resident bytes including KV cache estimates per context, per pool. When a `load*()` would breach budget, the eviction coordinator walks the `Resource` registry in `(priority asc, lastAccessMs asc)` order and pauses-then-evicts until enough is reclaimed. `warm()` marks a handle unevictable for `warm_ttl_seconds` (default 60s); apps that know they'll re-use a model immediately call this.

## Tree layout

Builds as `/system_ext/bin/oird` via `prebuilt_etc`. Lives at `system/oird/` in the AOSP tree.

```
system/oird/
├── oird.cpp                     # main() + binder lifecycle (~70 lines)
├── service/
│   ├── oir_service.h            # AIDL stub class declaration
│   └── oir_service.cpp          # ctor/dtor, setConfig, AIDL forwarders
├── backend/                     # per-backend method bodies + knobs
│   ├── llama_backend.{h,cpp}    # text.{complete,embed,translate}
│   ├── whisper_backend.{h,cpp}  # audio.transcribe
│   ├── ort_backend.{h,cpp}      # vision.{detect,embed,ocr}, audio.{vad,synthesize}, text.{classify,rerank}
│   └── vlm_backend.{h,cpp}      # vision.describe (mtmd over llama)
├── runtime/                     # cross-cutting state
│   ├── runtime.{h,cpp}          # mLock, mModels, scheduler hooks
│   ├── budget.h                 # resident bytes + LRU eviction
│   ├── load_registry.{h,cpp}    # concurrent-load dedup
│   ├── model_resource.{h,cpp}   # per-handle Resource adapter
│   └── resource.h               # Resource interface (v0.8 ObserveSession inherits this)
├── pool/
│   ├── context_pool.{h,cpp}     # llama_context pool (text + VLM)
│   └── whisper_pool.{h,cpp}     # whisper_context pool
├── sched/
│   └── scheduler.{h,cpp}        # priority-aware worker pool
├── tokenizer/                   # HF + phoneme sidecar parsers
├── validation/
│   └── ort_contract.{h,cpp}     # ORT shape validation at load time
├── common/
│   ├── error_codes.h            # W_* error constants → OIRError mapping
│   └── json_util.{h,cpp}
└── image_decode.{h,cpp}         # JPEG/PNG decode for vision.* (hardened against malformed input)
```

## Building

oird builds as part of a JibarOS / AOSP tree:

```bash
cd ~/aaosp
source build/envsetup.sh
lunch aosp_cf_x86_64_phone-trunk_staging-userdebug
m -j8 oird
```

Install on a running Cuttlefish:

```bash
adb root && adb remount
adb push out/target/product/vsoc_x86_64/system_ext/bin/oird /system_ext/bin/oird
adb shell stop oird && adb shell start oird
```

## Using

oird is a binder daemon — apps go through `OIRService` (system_server), not directly. For dev / debug, the platform ships a `cmd oir` shell:

```bash
adb shell cmd oir help
adb shell cmd oir submit "What is the capital of France?"
adb shell cmd oir transcribe /sdcard/voice-sample.wav
adb shell cmd oir detect /sdcard/photo.jpg
adb shell cmd oir embed "the quick brown fox"
adb shell cmd oir vembed /sdcard/cat.jpg
adb shell cmd oir describe /sdcard/cat.jpg "What's in this image?"
adb shell cmd oir warm text.complete
adb shell cmd oir dumpsys capabilities
adb shell cmd oir memory
```

For app integration, see [`oir-sdk`](https://github.com/Jibar-OS/oir-sdk) — Kotlin coroutine API on top of the same AIDL surface.

## Dependencies

| Repo | Purpose |
|------|---------|
| [`oir-framework-addons`](https://github.com/Jibar-OS/oir-framework-addons) | AIDL interfaces (`IOirWorker`, callbacks, parcelables) + `OIRService` (system_server side) |
| [`platform_external_llamacpp`](https://github.com/Jibar-OS/platform_external_llamacpp) | llama.cpp + libmtmd (text.* + vision.describe) |
| [`platform_external_whispercpp`](https://github.com/Jibar-OS/platform_external_whispercpp) | whisper.cpp (audio.transcribe) |
| [`platform_external_onnxruntime`](https://github.com/Jibar-OS/platform_external_onnxruntime) | ONNX Runtime (vision.* + audio.{vad,synthesize} + text.{classify,rerank}) |

## License

Apache 2.0.

## See also

- [`Jibar-OS/JibarOS`](https://github.com/Jibar-OS/JibarOS) — architecture, design docs, capability model, [ROADMAP](https://github.com/Jibar-OS/JibarOS/blob/main/docs/ROADMAP.md)
- [`oir-sdk`](https://github.com/Jibar-OS/oir-sdk) — Kotlin app API
- [`oir-vendor-models`](https://github.com/Jibar-OS/oir-vendor-models) — bundled default models (Git LFS)
- [`oir-demo`](https://github.com/Jibar-OS/oir-demo) — Mission Control sample showcasing concurrency + cancellation
