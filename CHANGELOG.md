# Changelog

User-visible and API-visible changes to oird, grouped by release. Per-commit detail is in `git log`; this file is the "what shipped" view at release granularity.

Format loosely follows [Keep a Changelog](https://keepachangelog.com). Pre-v0.7 history is in commit messages and the [JibarOS roadmap](https://github.com/Jibar-OS/JibarOS/blob/main/docs/ROADMAP.md).

---

## [Unreleased] — v0.7

The "tighten what we have" milestone. No new capabilities; daemon decomposition, pool semantics hardening, observability fields, build hygiene.

### Daemon decomposition

OirdService is now a thin AIDL router. Each capability body, knob, and per-handle pool lives on its respective backend class. v0.8 ObserveSession will land on the `Resource` interface that came out of step F.

- `Runtime` struct extracts cross-cutting state (`mLock`, `mModels`, `Budget`, `Scheduler`, `LoadRegistry`, handle counters, in-flight tracking, warm TTL). Backends hold `Runtime&`. (`8c19072`)
- `InFlightGuard` machinery moves to Runtime — backends produce/release guards through their `Runtime&` reference. (`75eab94`)
- `Resource` abstraction + eviction coordinator — polymorphic interface (`residentBytes` / `lastAccessMs` / `priority` / `pause` / `evict`); each backend's `load()` registers a `ModelResource` with backend-specific tear-down; Runtime walks the registry in `(priority asc, lastAccessMs asc)` order. The 5 inline 35-line LRU eviction loops collapse to a single `mRt.evictForBytesLocked(needed)` call. (`a576466`)
- `LlamaBackend` — owns text.{complete,embed,translate}: 6 AIDL methods + 12 knobs + `mPools` map. (skeleton `66d04b5`, full `1eea94b`)
- `WhisperBackend` — owns audio.transcribe: 2 AIDL methods + 4 knobs + `mPools` map. (skeleton `c75bd43`, full `81adca3`)
- `OrtBackend` — owns vision.{detect,embed,ocr}, audio.{vad,synthesize}, text.{classify,rerank}: 10 AIDL methods + 19 knobs + `mOrtEnv` + `mOcrRec`. (skeleton `15662c2`, full `5c594ee`)
- `VlmBackend` — owns vision.describe (libmtmd over llama): 2 AIDL methods + 6 knobs. Holds `LlamaBackend&` because mtmd contexts live in the llama pool. (skeleton `15662c2`, full `edc4224`)

**Behavior change** in step 4b: `text.classify` / `text.rerank` and `vision.{detect,embed,ocr}` priorities are now hardcoded to `PRIO_NORMAL` instead of tracking `mLlama.textEmbedPriority` (defaults unchanged; OEM-tunable knobs can be added if needed).

### Pool + scheduler semantics

- `InFlightGuard` RAII — 11 submit paths converted from manual `++` / `releaseInflight()` to `shared_ptr<InFlightGuard>` captured into Scheduler::Task lambdas. v0.6.8 ordering preserved: explicit `guard->release()` before terminal callback. (`4b7c4f5`)
- Empty-pool rejection at construction — `ContextPool` + `WhisperPool` ctors `LOG(FATAL)` on empty input. (`7fe78f9`)
- FIFO tiebreaker in `ContextPool::Waiter` ordering — comparator now `(priority, enqueueMs, id)`. (`7fe78f9`)

### Observability

- `getMemoryStats()` extended — `MemoryStats` parcelable now carries `backendLabels` / `poolSizes` / `busyCounts` / `waitingCounts` per loaded model. SDK consumers get pool telemetry through the typed AIDL surface (no `dumpRuntimeStats` + TSV-parse round-trip). `OIRShellCommand.cmdDumpsysMemory` collapsed accordingly. AIDL parcelable extension is append-only / backward-compatible. (`74ad6bf`; framework half is `oir-framework-addons@6e4d73a`)

### Build hygiene + cleanup

- Binder dispatch thread pool size — `4` magic number replaced with `clamp(hardware_concurrency / 2, 4, 8)` plus a startup log line. Auto-scales between cvd (4 cores → 4 threads) and beefier phones (16+ cores → 8 threads). (`4efce66`)
- Dead code removed from OirdService:
  - `runInference` declaration only (no definition existed)
  - `cleanupRequest` defined but no callers (backends use `mRt.cleanupRequest`)
  - `registerModelResourceLocked` kitchen-sink lambda (each backend uses `register{Llama,Whisper,Ort,Vlm}ModelResourceLocked` with backend-specific tear-down)
  - `readWav16` static helper (moved into `whisper_backend.cpp` anon namespace, the only consumer)
- 4 inline helpers relocated out of `oir_service.h`:
  - `fileSizeBytes` → `runtime/runtime.h`
  - `estimateKvBytesPerContext` → `backend/llama_backend.h`
  - `llama_batch_clear_local` / `_add_local` → `backend/llama_backend.cpp` anon namespace
  - `readDetectClassLabels` → `backend/ort_backend.cpp` anon namespace
- 8 heavy includes (`<llama.h>`, `<whisper.h>`, `<mtmd.h>`, `<onnxruntime_cxx_api.h>`, `<fstream>`, `<fcntl.h>`, `<sys/stat.h>`, `<unistd.h>`) leave the AIDL header. `oir_service.h`: 603 → 290 lines. Net: −174 lines, −300 in oir_service.h alone. (`c16968e`)
- Stale comments swept from oir_service.{h,cpp} and per-backend headers — `v0.7-post step XYZ` markers left to commit history; class-level summaries updated to reflect post-decomposition reality.

### Documentation

- README rewrite — capability table, post-decomposition architecture diagram, tree layout, build/install/use examples, dependencies. 39 → 187 lines. (`aee14a8`)

### Validation

Smoke-tested on cvd after each step: `text.complete`, `text.embed`, `audio.transcribe`, `vision.detect`, `vision.embed` (+ `cmd oir memory` for the MemoryStats extension; concurrent load test confirmed `busy=4` under 6 submits vs pool=4). Full clean rebuild after step 5b: 1m38; incremental rebuild after cleanup: 23s.

### Earlier v0.7 work (mechanical file split + class extraction)

`oird.cpp` went from 5,220 lines to 56 (main + binder lifecycle only). 12 commits between `f6c79b6` and `a0a74b9` carved out `pool/`, `sched/`, `runtime/{budget,load_registry}`, `tokenizer/`, `validation/`, `common/`, the `OirdService` class declaration, and per-backend method bodies. Every public symbol preserved; smoke-tested after each commit.

---

## v0.6.9 and earlier

Pre-v0.7 history lives in `git log` and the [JibarOS roadmap](https://github.com/Jibar-OS/JibarOS/blob/main/docs/ROADMAP.md). Highlights: concurrent-load deadlock fix + load-dedup registry (v0.6.9), terminal-callback lease defer (v0.6.8), cross-backend scheduler infrastructure (v0.6.3), per-model context pooling + KV accounting (v0.6 phase A), libmtmd bump for modern VLMs (v0.5), per-UID rate limits + OEM tuning knobs (v0.5).
