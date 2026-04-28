// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// service/oir_service.h — OirdService AIDL implementation + supporting
// state. v0.7 step 6 of the daemon decomposition: the entire anonymous
// namespace from oird.cpp moved into namespace oird here. Method bodies
// remain inline in this header for now; step 7+ splits them out into
// backend/{llama,whisper,vlm,ort}.cpp files.
//
// Contents (in declaration order):
//   - error code constants (W_*) — pulled from common/error_codes.h
//   - llama_batch_clear_local / llama_batch_add_local (inline helpers)
//   - LoadedModel struct (per-handle resident state, all backends)
//   - estimateKvBytesPerContext / fileSizeBytes / readDetectClassLabels
//     (load-path helpers, llama + ort)
//   - InFlightGuard (RAII for LoadedModel::inFlightCount)
//   - OirdService class (AIDL implementation; ~30 capability handlers)
#pragma once

#include <android-base/logging.h>
#include <android/binder_manager.h>
#include <android/binder_process.h>

#include <llama.h>
#include <whisper.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <onnxruntime_cxx_api.h>
#include <fstream>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unordered_map>
#include <vector>
#include <functional>
#include <queue>

#include <aidl/com/android/server/oir/BnOirWorker.h>
#include <aidl/com/android/server/oir/IOirWorkerCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerVectorCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerAudioCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerBboxCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerRealtimeBooleanCallback.h>
#include <aidl/com/android/server/oir/MemoryStats.h>

#include "image_decode.h"
#include "backend/llama_backend.h"
#include "backend/ort_backend.h"
#include "backend/vlm_backend.h"
#include "backend/whisper_backend.h"
#include "runtime/model_resource.h"
#include "common/error_codes.h"
#include "common/json_util.h"
#include "pool/context_pool.h"
#include "pool/whisper_pool.h"
#include "runtime/budget.h"
#include "runtime/load_registry.h"
#include "runtime/runtime.h"
#include "sched/scheduler.h"
#include "tokenizer/hf_tokenizer.h"
#include "tokenizer/phoneme_loader.h"
#include "validation/ort_contract.h"

namespace oird {

using aidl::com::android::server::oir::BnOirWorker;
using aidl::com::android::server::oir::IOirWorkerCallback;
using aidl::com::android::server::oir::IOirWorkerVectorCallback;
using aidl::com::android::server::oir::IOirWorkerAudioCallback;
using aidl::com::android::server::oir::IOirWorkerBboxCallback;
using aidl::com::android::server::oir::IOirWorkerRealtimeBooleanCallback;
using aidl::com::android::server::oir::MemoryStats;

// Inline batch helpers lifted from AAOSP's llm_jni.cpp.
inline void llama_batch_clear_local(struct llama_batch& batch) {
    batch.n_tokens = 0;
}

inline void llama_batch_add_local(
        struct llama_batch& batch,
        llama_token id,
        llama_pos pos,
        const std::vector<llama_seq_id>& seq_ids,
        bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = (int32_t)seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

// v0.7: ContextPool / WhisperPool moved to pool/ — see #include above.


// v0.7: extracted submodules now in their own translation units —
//   currentTimeMs(), ContextPool, ContextLease     → pool/context_pool.h
//   WhisperPool, WhisperLease                       → pool/whisper_pool.h
//   Scheduler                                       → sched/scheduler.h
//   error codes (W_*)                               → common/error_codes.h
//   JSON helpers (fileExists, parseJsonString, ...) → common/json_util.h
//   HfTokenizer                                     → tokenizer/hf_tokenizer.h
//   PhonemeMap, graphemesToPhonemeIds               → tokenizer/phoneme_loader.h
//   validateOrtContract                             → validation/ort_contract.h

// v0.6 Phase A: estimate KV cache bytes per llama_context so the
// MemoryManager can account pool overhead in the resident budget.
//
// Formula: n_ctx × n_layer × 2 (K+V) × n_kv_head × head_dim × bytes_per_elem
//
// Elements are fp16 by default in llama.cpp (2 bytes). This is an
// estimate — real KV allocation may differ by a few % due to alignment
// and per-layer tensor padding. Close enough for budget planning.
inline int64_t estimateKvBytesPerContext(llama_model* m, int32_t n_ctx) {
    if (!m || n_ctx <= 0) return 0;
    int64_t n_layer = llama_model_n_layer(m);
    int64_t n_embd  = llama_model_n_embd(m);
    int64_t n_head  = llama_model_n_head(m);
    int64_t n_head_kv = llama_model_n_head_kv(m);
    if (n_head_kv <= 0) n_head_kv = n_head;
    if (n_head <= 0 || n_layer <= 0 || n_embd <= 0) return 0;
    int64_t head_dim = n_embd / n_head;
    // K + V, fp16.
    return (int64_t)n_ctx * n_layer * 2 * n_head_kv * head_dim * 2;
}

inline int64_t fileSizeBytes(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return st.st_size;
    return 0;
}

// v0.5 V8: read a <model>.classes.json sidecar next to an ONNX detection model.
// Derives the sidecar path by swapping a trailing ".onnx" for ".classes.json"
// (falls back to appending if the path doesn't end in .onnx). Returns an empty
// vector when the sidecar is absent, unreadable, or malformed — callers treat
// that as "use the embedded COCO-80 fallback."
//
// Narrow schema: { "classes": ["a", "b", ...], ...optional-fields... }.
// Handwritten parser — other fields (input_size, normalize, family, NMS
// thresholds) are scoped to v0.6+; capability-tuning knobs that overlap
// (score_threshold / iou_threshold) live in oir_config.xml via V7, not here.
inline std::vector<std::string> readDetectClassLabels(const std::string& modelPath) {
    std::string sidecarPath;
    const std::string dotOnnx = ".onnx";
    if (modelPath.size() > dotOnnx.size() &&
        modelPath.compare(modelPath.size() - dotOnnx.size(), dotOnnx.size(), dotOnnx) == 0) {
        sidecarPath = modelPath.substr(0, modelPath.size() - dotOnnx.size()) + ".classes.json";
    } else {
        sidecarPath = modelPath + ".classes.json";
    }

    std::ifstream f(sidecarPath);
    if (!f.is_open()) return {};
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    const size_t keyPos = content.find("\"classes\"");
    if (keyPos == std::string::npos) {
        LOG(WARNING) << "oird: sidecar " << sidecarPath << " missing \"classes\" key";
        return {};
    }
    const size_t arrStart = content.find('[', keyPos);
    if (arrStart == std::string::npos) return {};
    const size_t arrEnd = content.find(']', arrStart);
    if (arrEnd == std::string::npos) return {};

    std::vector<std::string> out;
    size_t p = arrStart + 1;
    while (p < arrEnd) {
        while (p < arrEnd && (content[p] == ' ' || content[p] == '\t' ||
                              content[p] == '\n' || content[p] == '\r' ||
                              content[p] == ',')) ++p;
        if (p >= arrEnd) break;
        if (content[p] != '"') {
            LOG(WARNING) << "oird: sidecar " << sidecarPath
                         << " unexpected char at offset " << p;
            return {};
        }
        ++p; // opening quote
        std::string s;
        while (p < arrEnd && content[p] != '"') {
            if (content[p] == '\\' && p + 1 < arrEnd) {
                const char next = content[p + 1];
                if (next == '"') s.push_back('"');
                else if (next == '\\') s.push_back('\\');
                else if (next == 'n') s.push_back('\n');
                else if (next == 't') s.push_back('\t');
                else if (next == '/') s.push_back('/');
                else s.push_back(next);
                p += 2;
            } else {
                s.push_back(content[p++]);
            }
        }
        if (p >= arrEnd) break;
        ++p; // closing quote
        out.push_back(std::move(s));
    }

    LOG(INFO) << "oird: loaded " << out.size() << " class labels from sidecar "
              << sidecarPath;
    return out;
}


// v0.6.2: same-model concurrency for whisper — see pool/whisper_pool.h
// for design notes; implementation in pool/whisper_pool.cpp.


// v0.6.3: cross-backend scheduler — see sched/scheduler.h for design
// notes; implementation in sched/scheduler.cpp.


// v0.7-post step 2a: InFlightGuard moved to runtime/runtime.h alongside
// Runtime::releaseInflight + Runtime::acquireInflightLocked so backend
// classes can construct and release guards through their Runtime&.

class OirdService : public BnOirWorker {
public:
    OirdService();

    ~OirdService() override;

    ::ndk::ScopedAStatus load(const std::string& modelPath, int64_t* _aidl_return) override;

    // v0.4 H4-A: load a model in embedding mode.
    // Separate code path from load() because llama_context_params differ materially
    // (embeddings=true, pooling_type=MEAN, smaller n_ctx is fine for sentence-level).
    ::ndk::ScopedAStatus loadEmbed(const std::string& modelPath, int64_t* _aidl_return) override;

    // v0.4 H4-A: run text through an embedding model; callback receives one pooled vector.
    // Synchronous in current v0.4 (no async thread) since embedding is ~1ms for MiniLM.
    ::ndk::ScopedAStatus submitEmbed(int64_t modelHandle,
                                     const std::string& text,
                                     const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                     int64_t* _aidl_return) override;

    // v0.4 H1: 16-bit mono 16 kHz WAV → std::vector<float>.
    // Minimal parser: handles common "fmt " + "data" and skips LIST/INFO chunks.
    static bool readWav16(const std::string& path, std::vector<float>& out);

    ::ndk::ScopedAStatus loadWhisper(const std::string& modelPath, int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus submitTranscribe(int64_t modelHandle,
                                          const std::string& audioPath,
                                          const std::shared_ptr<IOirWorkerCallback>& cb,
                                          int64_t* _aidl_return) override;

    // ========================================================================
    // ONNX Runtime integration — state as of v0.6.
    //
    // loadOnnx() is fully implemented — it creates an ORT session from an
    // on-disk .onnx model and tracks it in mRt.mModels under the same budget/LRU
    // machinery as llama and whisper. Validation fires at load time for
    // detect, synthesize, embed, and vad shapes (see validateOrtContract +
    // per-submit loadOnnx sites). Consumers can load any ONNX model through
    // this path; mismatched shapes surface MODEL_INCOMPATIBLE before the
    // first inference.
    //
    // All four submit* ONNX paths are shipping real:
    //   - submitDetect — image decode + letterbox + NMS over YOLOv8 or
    //     RT-DETR outputs; shape-validated at loadOnnx.
    //   - submitSynthesize — real Piper TTS with a permissive G2P sidecar
    //     (phonemes.json bundled per-voice). Scales + sample rate are
    //     OEM-tunable via setCapabilityFloat; shape-validated at loadOnnx
    //     (3 inputs: input/input_lengths/scales; 1 output).
    //   - submitVisionEmbed — SigLIP/CLIP-family pooled embeddings;
    //     shape-validated at loadOnnx.
    //   - submitVad — Silero VAD streaming on/off transitions;
    //     shape-validated at loadOnnx.
    //
    // Handle lifecycle + AIDL surface have been stable since v0.4.
    // ========================================================================

    ::ndk::ScopedAStatus loadOnnx(const std::string& modelPath,
                                  bool isDetection,
                                  int64_t* _aidl_return) override;

    // v0.4 H2 → v0.6 typed-error stub: Piper ORT session loads successfully
    // but the G2P (text→phoneme) frontend isn't integrated yet. Every
    // submit returns CAPABILITY_UNAVAILABLE_NO_MODEL immediately so apps
    // get a handleable error (OirCapabilityUnavailableException in SDK
    // once v0.7 ships it) instead of silent bad audio. v0.6 Phase C / G2P
    // slice lands the real impl; this whole block goes away then.
    //
    // Worker-side code W_CAPABILITY_UNSUPPORTED=12; OIRService maps to
    // Java OIRError.CAPABILITY_UNAVAILABLE_NO_MODEL=9 (same class of
    // "configured but not runnable on this device" errors).
    // v0.6 Phase C: audio.synthesize — real Piper inference path.
    //
    // Contract (Piper VITS ONNX):
    //   input tensor  "input"         int64   [1, N_phonemes]
    //   input tensor  "input_lengths" int64   [1]
    //   input tensor  "scales"        float32 [3]   (noise, length, noise_w)
    //   output tensor "output"        float32 [1, 1, n_samples]  @ 22050 Hz mono
    //
    // The model is shape-validated at loadOnnx time (see validateOrtContract).
    // G2P (grapheme→phoneme) is provided by an OEM-baked sidecar next to the
    // ONNX: `<model-path>.phonemes.json`. Format is a UTF-8 JSON object with:
    //   { "version": 1,
    //     "phoneme_ids": { "<phoneme-str>": <int>, ... },   // eSpeak IPA → model id
    //     "grapheme_map": { "<grapheme-or-word>": [<ph-id>, ...], ... } }
    // The runtime reads this once at first submitSynthesize call and caches
    // per-handle. This keeps AOSP source platform-neutral — locale-specific
    // phonetics live in the model package the OEM ships, same as whisper
    // models, vision.detect class sidecars, and VLM mmproj files.
    //
    // If the sidecar is missing, runtime returns CAPABILITY_UNAVAILABLE_NO_MODEL
    // rather than SIGSEGV or silent bad audio. This is the v0.6 "declared but
    // not runnable on this device" pattern.
    ::ndk::ScopedAStatus submitSynthesize(int64_t modelHandle,
                                          const std::string& text,
                                          const std::shared_ptr<IOirWorkerAudioCallback>& cb,
                                          int64_t* _aidl_return) override;

    // v0.6 Phase B: text.classify. Runs an ONNX classifier (encoder + logits head)
    // over the input text and returns per-label softmax scores as a Vector.
    //
    // Contract: single input tensor "input_ids" int64 [1, N_tokens],
    // optional "attention_mask" int64 [1, N_tokens]. Output "logits"
    // float32 [1, N_labels]. Tokenization requires a `<model>.tokenizer.json`
    // HuggingFace-format sidecar; runtime returns NO_MODEL if absent.
    //
    // This keeps AOSP source tokenizer-free: OEMs bake the model + its
    // tokenizer.json as a pair, same convention as vision.detect classes.
    ::ndk::ScopedAStatus submitClassify(int64_t modelHandle,
                                        const std::string& text,
                                        const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                        int64_t* _aidl_return) override;

    // v0.6 Phase B: text.rerank. Cross-encoder scoring — runs (query,
    // candidate[i]) through a reranker and emits one score per candidate.
    //
    // Contract (MS-MARCO-MiniLM style cross-encoder ONNX):
    //   input  "input_ids"      int64 [1, N_tokens]    — [CLS] q [SEP] cand [SEP]
    //   input  "attention_mask" int64 [1, N_tokens]
    //   input  "token_type_ids" int64 [1, N_tokens]    — 0 for q, 1 for cand
    //   output "logits"         float32 [1, 1]         — scalar relevance
    // Same tokenizer sidecar as submitClassify.
    ::ndk::ScopedAStatus submitRerank(int64_t modelHandle,
                                       const std::string& query,
                                       const std::vector<std::string>& candidates,
                                       const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                       int64_t* _aidl_return) override;

    // v0.6 Phase B: text.translate. Thin wrapper around submit() that forces
    // translation-leaning sampler settings (low temp, no top-k). The prompt
    // passed here has already been rewritten by OIRService into an
    // instruction-style translation request, so the llama model generates
    // the translation as plain text. Zero new runtime — reuses text.complete
    // ContextPool.
    ::ndk::ScopedAStatus submitTranslate(int64_t modelHandle,
                                          const std::string& prompt,
                                          int32_t maxTokens,
                                          const std::shared_ptr<IOirWorkerCallback>& cb,
                                          int64_t* _aidl_return) override;

    // v0.6 Phase B: vision.ocr. Detection + recognition chain.
    //
    // Contract:
    //   det  ONNX (the loaded model):  image → N regions as BoundingBoxes
    //   rec  ONNX (sidecar `<model>.rec.onnx`):  cropped region → token IDs
    //   vocab sidecar (`<model>.rec.vocab.txt`):  id → UTF-8 character
    //
    // If either sidecar is missing runtime returns CAPABILITY_UNAVAILABLE_NO_MODEL.
    // The det-only case (bboxes without text) is not useful as OCR so it
    // does not fall through to a partial result.
    ::ndk::ScopedAStatus submitOcr(int64_t modelHandle,
                                    const std::string& imagePath,
                                    const std::shared_ptr<IOirWorkerBboxCallback>& cb,
                                    int64_t* _aidl_return) override;

    // v0.4 H3: vision.detect real impl.
    // Decode → letterbox 640×640 → YOLOv8 inference → parse (1, 84, 8400)
    // → NMS → map coords back to source image → bboxes callback.
    // Default class labels = COCO-80 inline. Sidecar <model>.classes.json
    // support deferred to v0.5 (trivial std::ifstream + simple tokenizer).
    ::ndk::ScopedAStatus submitDetect(int64_t modelHandle,
                                      const std::string& imagePath,
                                      const std::shared_ptr<IOirWorkerBboxCallback>& cb,
                                      int64_t* _aidl_return) override;

    // v0.4 H4-B: vision.embed — AIDL + ORT session load + preprocess + pool
    // shipped real. SigLIP-base validated on cvd (768-dim output vector).
    ::ndk::ScopedAStatus loadVisionEmbed(const std::string& modelPath, int64_t* _aidl_return) override;

    // v0.4 H4-B: vision.embed real impl.
    // Decode → resize 224×224 (bilinear) → SigLIP normalize (mean=0.5 std=0.5)
    // → ORT Run → mean-pool if patch-tokens → L2 normalize → emit vector.
    ::ndk::ScopedAStatus submitVisionEmbed(int64_t modelHandle,
                                           const std::string& imagePath,
                                           const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                           int64_t* _aidl_return) override;

    // ========================================================================
    // v0.4 H6 — vision.describe via clip.cpp + LLaVA-family VLM.
    //
    // Two model files: CLIP mmproj (.gguf) + LLM (.gguf). Loaded together
    // as a single LoadedModel handle — both share budget, both evict as one.
    // ========================================================================

    ::ndk::ScopedAStatus loadVlm(const std::string& clipPath,
                                 const std::string& llmPath,
                                 int64_t* _aidl_return) override;

    // Describe an image. Uses LLaVA-style template:
    //   USER: <image-tokens>\n<prompt>\nASSISTANT:
    // Token streaming via cb->onToken(String, int outputIndex).
    ::ndk::ScopedAStatus submitDescribeImage(int64_t modelHandle,
                                             const std::string& imagePath,
                                             const std::string& prompt,
                                             const std::shared_ptr<IOirWorkerCallback>& cb,
                                             int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus unload(int64_t modelHandle) override;

    ::ndk::ScopedAStatus submit(int64_t modelHandle,
                                const std::string& prompt,
                                int32_t maxTokens,
                                float temperature,
                                const std::shared_ptr<IOirWorkerCallback>& callback,
                                int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus cancel(int64_t requestHandle) override;

    // ========================================================================
    // v0.5 V5: audio.vad via Silero VAD (ONNX, MIT, ~2 MB).
    //
    // Silero v4/v5 ONNX signature:
    //   inputs:  input=[1,N_samples] float  (N_samples=512 @ 16kHz = 32ms)
    //            state=[2,1,128] float      (LSTM hidden+cell state)
    //            sr=[1] int64               (16000)
    //   outputs: output=[1,1] float         (voice probability 0–1)
    //            stateN=[2,1,128] float     (updated state)
    //
    // Caller contract: submitVad consumes a PCM16 LE 16kHz mono file and emits
    // one onState(bool, timestampMs) per 32ms window. Threshold 0.5.
    // Live-mic streaming is an app-side composition — client feeds microphone
    // PCM through the same shape. Shape frozen in v0.5.
    // ========================================================================
    ::ndk::ScopedAStatus loadVad(const std::string& modelPath,
                                 int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus submitVad(int64_t modelHandle,
                                   const std::string& pcmPath,
                                   const std::shared_ptr<IOirWorkerRealtimeBooleanCallback>& cb,
                                   int64_t* _aidl_return) override;

    // v0.4 S2-B: configure budget + warm TTL. Called once by OIRService at attachWorker.
    ::ndk::ScopedAStatus setConfig(int32_t memoryBudgetMb, int32_t warmTtlSeconds) override;

    // v0.5 V7: per-capability tuning via (key, float-value) pushed by OIRService
    // at attachWorker. Unknown keys are logged and ignored so OEMs can add new
    // knobs in their config files ahead of runtime support without errors.
    ::ndk::ScopedAStatus setCapabilityFloat(const std::string& key, float value) override;

    // v0.5 V7 full: string-valued counterpart. Same routing pattern; unknown
    // keys logged and ignored.
    ::ndk::ScopedAStatus setCapabilityString(const std::string& key,
                                             const std::string& value) override;

    // v0.4 S3: mark a model as warm — unevictable for warm_ttl_seconds past this call.
    ::ndk::ScopedAStatus warm(int64_t modelHandle) override;

    // v0.6.3: per-handle runtime snapshot. Pairs with getMemoryStats
    // (model-level RSS totals) — this one exposes the live pool state
    // ContextPool + WhisperPool expose as getters. Returned as a
    // newline-separated TSV so `cmd oir memory` can print verbatim
    // without needing a new parcelable layout.
    ::ndk::ScopedAStatus dumpRuntimeStats(std::string* _aidl_return) override;

    // v0.6.3: worker-side file stat fallback used by OIRService when
    // system_server's SELinux scope can't `getattr` an `oir_model_*_file`
    // label directly. Rechecked through oird's domain which does have
    // the read rule platform-wide. Path resolution is native stat();
    // symlinks resolve; permission-denied maps to false.
    ::ndk::ScopedAStatus fileIsReadable(const std::string& path, bool* _aidl_return) override;

    // v0.4 S2: snapshot of worker memory state for cmd oir dumpsys memory.
    ::ndk::ScopedAStatus getMemoryStats(MemoryStats* _aidl_return) override;

private:
    // v0.7-post step 1: cross-cutting state (mutex, model handle table,
    // budget, scheduler, load-dedup registry, request handle counters,
    // mRt.mActiveRequests, mRt.mWarmTtlSeconds) moved into Runtime. Backends will
    // each gain a Runtime& reference in step 2+; until then OirdService
    // is the only thing that holds Runtime, and member methods access
    // its fields via mRt.X.
    Runtime mRt;

    void runInference(int64_t modelHandle,
                      int64_t handle,
                      std::string prompt,
                      int32_t maxTokens,
                      float temperature,
                      std::shared_ptr<IOirWorkerCallback> cb,
                      std::shared_ptr<std::atomic_bool> cancelled,
                      std::shared_ptr<InFlightGuard> guard);

    void cleanupRequest(int64_t handle);

    // v0.7-post step F: register a ModelResource for a freshly-loaded
    // handle. Caller holds mRt.mLock. The tear-down lambda is the
    // kitchen-sink (frees state across all backends) — when backends
    // are extracted (steps 3-5) this can specialize per-backend.
    void registerModelResourceLocked(int64_t handle);

    // v0.7-post step 2a: releaseInflight() and mRt.acquireInflightLocked()
    // moved to Runtime so backend classes can use them through their
    // Runtime& reference. See runtime/runtime.h.

    // v0.6.3: resolve the scheduler priority for a capability name.
    // Uses the existing per-capability mXxxPriority knobs; unknown caps
    // default to PRIO_NORMAL. Called from submit* methods when building
    // the scheduler enqueue.
    int32_t priorityForCapability(const std::string& cap);

    // v0.6 Phase A: per-model context pool for llama-backed capabilities
    // (text.complete / text.embed / vision.describe). Keyed by modelHandle.
    // Pool created at load time; destroyed on unload / LRU eviction.
    // v0.7-post step 2b1: pool map lives on LlamaBackend; OirdService methods access it via mLlama.mPools.
    LlamaBackend mLlama{mRt};

    // v0.7-post step 3a: whisper-backed capability (audio.transcribe)
    // owns its state via WhisperBackend. mWhisperPools became
    // mWhisper.mPools (still public for OirdService methods until
    // method-body migration in step 3b).
    WhisperBackend mWhisper{mRt};

    // v0.7-post step 4a: ORT-backed capabilities (vision.detect, .ocr,
    // .embed; audio.synthesize, .vad; text.classify, .rerank) own their
    // state via OrtBackend. mOcrRec became mOrt.mOcrRec.
    OrtBackend mOrt{mRt};

    // v0.7-post step 5a: VlmBackend placeholder. No unique per-handle
    // state today (VLM ContextPools live in mLlama.mPools); the slot
    // is in place for step 5b knob + method-body migration.
    VlmBackend mVlm{mRt};

    // v0.5 V7: per-capability tuning. Defaults match v0.4 hardcoded constants;
    // OEM overrides land via setCapabilityFloat() calls at worker attach time.
    float mDetectScoreThresh = 0.25f;
    float mDetectIouThresh   = 0.45f;
    // v0.5 V5 + V7: audio.vad tunables. Defaults match Silero v5 @ 16 kHz.
    // OEMs swapping to a different VAD model (or Silero 8 kHz) override
    // these via oir_config.xml. voice_threshold: prob cutoff. sample_rate_hz:
    // model's expected sample rate. window_samples + context_samples: the
    // model's required input layout (context prepended to the current
    // window; total input tensor length is context + window). Wrong values
    // for a given model silently break inference, so verify against the
    // model's signature before tuning.
    float mVadVoiceThreshold = 0.5f;
    int32_t mVadSampleRateHz = 16000;
    int32_t mVadWindowSamples  = 512;
    int32_t mVadContextSamples = 64;

    // v0.5 V7 full: remaining capability tuning knobs. All have defaults
    // matching today's hardcoded values; OEM overrides land via
    // setCapabilityFloat / setCapabilityString at attachWorker time.
    //
    // Sizing / latency:
    int32_t mVisionDescribeNCtx    = 4096;
    int32_t mVisionDescribeNBatch  = 2048;
    int32_t mVisionDescribeMaxTokens = 256;
    // v0.6 Phase A: per-capability pool configuration. Pool size drives
    // KV memory (pool_size × n_ctx × n_layer × head_dim × 4 bytes); OEMs
    // with tight budgets drop each below its default.
    int32_t mVisionDescribeContextsPerModel  = 1;  // VLMs are 4GB+; pool=1 by default
    // v0.6.2: whisper context pool default. Whisper-tiny is ~40MB; per-ctx
    // state is small (decoder buffers, sampling state) so 2 is the right
    // balance between memory cost and concurrent transcribe support
    // (dictation + one-shot audio upload is a common 2-stream scenario).
    int32_t mAudioTranscribeContextsPerModel = 2;
    // Acquire timeouts — max time a caller waits for a free pool slot.
    // Hitting the timeout returns OIRError::TIMEOUT to the app; apps retry
    // or back off. Protects against pool wedging on a stuck inference.
    int32_t mVisionDescribeAcquireTimeoutMs = 60000;
    // v0.6.2: transcribe can legitimately take 10-30 s for longer audio;
    // a 60 s acquire timeout is generous but matches the single-ctx
    // worst-case that v0.6 already tolerated.
    int32_t mAudioTranscribeAcquireTimeoutMs = 60000;
    // Priority bands — lower = higher priority (POSIX niceness convention).
    // 0 = audio realtime, 10 = normal (text/vision), 20 = low/batch.
    // When audio.* and text.* contend for the same pool, audio is granted
    // first by release() hand-off.
    int32_t mVisionDescribePriority   = ContextPool::PRIO_NORMAL;
    int32_t mAudioTranscribePriority  = ContextPool::PRIO_AUDIO_REALTIME;
    int32_t mAudioVadPriority         = ContextPool::PRIO_AUDIO_REALTIME;
    int32_t mAudioSynthesizePriority  = ContextPool::PRIO_AUDIO_REALTIME;
    // Sampling defaults — apps pass temperature via submit() AIDL; when
    // caller passes <0 we fall back to these defaults. top_p is not yet
    // exposed via AIDL so the default is the effective value.
    // Batch size for llama_batch_init — affects prefill speed. Higher = more
    // memory during prefill. Default 512 matches upstream llama.cpp recommended.
    // Model-geometry (wrong values silently break inference — OEMs verify
    // against the bundled model's ONNX/GGUF signature):
    int32_t mVisionEmbedInputSize  = 224;   // SigLIP-base
    float   mVisionEmbedNormMean   = 0.5f;
    float   mVisionEmbedNormStd    = 0.5f;
    int32_t mVisionDetectInputSize = 640;   // YOLOv8n / RT-DETR-R50 both 640
    size_t mImageMaxPixels = kDefaultMaxImagePixels;  // v0.7: image.max_pixels knob (0 = no cap)
    // v0.5: default flipped to rtdetr to match platform-default RT-DETR model
    // shipped in /product/etc/oir/. OEMs swapping to YOLOv8 set family=yolov8
    // explicitly in their /vendor/etc/oir/oir_config.xml fragment.
    std::string mVisionDetectFamily    = "rtdetr";  // yolov8 / yolov5 / detr / rtdetr
    std::string mVisionDetectNormalize = "0_to_1";   // "0_to_1" / "mean_std"
    std::string mAudioTranscribeLanguage = "en";
    int32_t mAudioSynthesizeSampleRate = 22050;
    float   mAudioSynthesizeLengthScale = 1.0f;
    float   mAudioSynthesizeNoiseScale  = 0.667f;

    // v0.4 H2/H3: ONNX Runtime env + session options, created lazily on first
    // loadOnnx call. One env per process is the ORT recommendation; all sessions
    // share it.
    std::unique_ptr<Ort::Env> mOrtEnv;
    Ort::SessionOptions makeOrtSessionOptions(bool isDetection) const;
    void ensureOrtEnv();
};

} // namespace oird
