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
#include "common/error_codes.h"
#include "common/json_util.h"
#include "pool/context_pool.h"
#include "pool/whisper_pool.h"
#include "runtime/budget.h"
#include "runtime/load_registry.h"
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

struct LoadedModel {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    // v0.6 Phase A: legacy single-ctx field retained for transition; v0.6
    // inference paths lease from the per-model pool in
    // OirdService::mLlamaPools instead. Kept nullptr for new loads; only
    // non-null if a caller explicitly uses the legacy non-pooled path (none
    // in v0.6 shipped code).
    llama_context* ctx = nullptr;
    int32_t context_size = 0;
    int32_t n_threads = 4;
    int64_t handle = 0;
    std::string path;
    int64_t sizeBytes = 0;     // v0.4 S2: resident size (file size; mmap pages)
    int64_t loadTimestampMs = 0; // v0.4 S2: when load() was called
    int64_t lastAccessMs = 0;   // v0.4 S2: last submit dispatch time
    int64_t warmUntilMs = 0;    // v0.4 S3: unevictable while now < warmUntilMs
    int32_t inFlightCount = 0;  // v0.4 S2-B: unevictable while > 0
    bool    isEmbedding = false; // v0.4 H4-A: context built with embeddings=true
    whisper_context* wctx = nullptr; // v0.4 H1: set when isWhisper (not llama_context)
    bool    isWhisper = false;       // v0.4 H1: route to whisper API
    // v0.4 H2/H3: ONNX Runtime session (owned by heap allocation so we can
    // null-check cheaply; the session's env + allocator live in the static
    // singletons defined in the OirdService class).
    Ort::Session* ortSession = nullptr;
    bool    isOnnx = false;         // v0.4 H2/H3: route to ORT
    bool    onnxIsDetection = false; // v0.4 H3: detection vs synthesis model
    bool    isVisionEmbed = false;  // v0.4 H4-B: ORT session is a vision encoder (SigLIP / CLIP)
    // v0.5 V1: VLM state migrated from libllava to libmtmd (PR #12849).
    // mtmdCtx replaces the v0.4 clipCtx — mtmd_context owns both the image
    // encoder (CLIP) and the bound text-model reference internally.
    // (model, ctx, vocab) still hold the text LLM separately because the
    // sampler chain + token loop run directly on llama_context.
    mtmd_context* mtmdCtx = nullptr;
    bool    isVlm = false;
    // v0.5 V8: OEM classes.json sidecar for detect models. Empty → fall back
    // to embedded COCO-80. Populated at loadOnnx time; immutable thereafter.
    std::vector<std::string> detectClassLabels;
    // v0.5 V5: audio.vad — Silero ONNX session. Separate flag from isOnnx so
    // the existing detect/synth dispatch stays untouched. LSTM state persists
    // across submitVad windows in mVadState (kept out of this struct so it
    // can live on a per-request basis rather than per-handle — see submitVad).
    bool    isVad = false;
    // v0.6 Phase A: true when this model has a pool in mLlamaPools. Set at
    // load time for llama-backed models (text.complete / text.embed /
    // vision.describe). Submit paths use mLlamaPools.find(handle) → lease.
    bool    hasLlamaPool = false;
};

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


// Forward decl so InFlightGuard can hold a back-pointer.
class OirdService;

// v0.7: RAII guard for LoadedModel::inFlightCount. Replaces the v0.4
// pattern of paired `it->second.inFlightCount++;` ... `releaseInflight(h);`
// calls — same invariant, but now enforced by C++ object lifetime instead
// of comments on every submit path.
//
// Acquired via OirdService::acquireInflightLocked(lm, handle) under mLock;
// the destructor (or explicit release()) calls releaseInflight() exactly
// once. Wrapped in std::shared_ptr at every site so the guard can be
// captured into Scheduler::Task (std::function<void()>, which requires the
// captured lambda to be copy-constructible).
//
// v0.6.8 ordering note: many submit paths must release the inflight BEFORE
// firing the terminal binder callback (onComplete/onError) so a stalled
// callback can't pin the model resident. Call guard->release() explicitly
// at that point; the destructor then becomes a no-op. If callers forget
// the explicit release, the destructor is the safety net — slightly
// later release, but still no leak.
class InFlightGuard {
public:
    InFlightGuard(OirdService* svc, int64_t modelHandle)
            : mSvc(svc), mModelHandle(modelHandle), mActive(true) {}
    ~InFlightGuard();

    InFlightGuard(const InFlightGuard&) = delete;
    InFlightGuard& operator=(const InFlightGuard&) = delete;
    InFlightGuard(InFlightGuard&&) = delete;
    InFlightGuard& operator=(InFlightGuard&&) = delete;

    // Decrement now (idempotent). Use to control v0.6.8 ordering of release
    // vs terminal callback. Subsequent destructor is a no-op.
    void release();

private:
    OirdService* mSvc;
    int64_t      mModelHandle;
    bool         mActive;
};

class OirdService : public BnOirWorker {
    // v0.7: InFlightGuard's release() calls the private releaseInflight();
    // friending lets us keep that member private without exposing it as a
    // public API.
    friend class InFlightGuard;
public:
    OirdService() {
        llama_backend_init();
        // v0.6.3: build the cross-backend scheduler. hardware_concurrency
        // clamped so single-core dev environments still get 4 workers
        // (avoids head-of-line blocking) and big servers don't spawn 64
        // worker threads unnecessarily.
        unsigned cores = std::thread::hardware_concurrency();
        int workers = static_cast<int>(std::clamp<unsigned>(cores, 4u, 16u));
        mScheduler = std::make_unique<Scheduler>(workers);
        LOG(INFO) << "oird: llama.cpp backend initialized; scheduler workers=" << workers;
    }

    ~OirdService() override {
        // Stop the scheduler BEFORE tearing down model state — otherwise
        // an in-flight worker task could deref a freed llama_model while
        // the destructor races it. Scheduler dtor joins all workers.
        mScheduler.reset();
        std::lock_guard<std::mutex> lk(mLock);
        // Mirror the LRU-eviction cleanup (see loadOnnx / load / loadEmbed
        // / loadWhisper / loadVlm eviction paths). Process-exit would
        // eventually reclaim these, but tests, restart sequences, and any
        // future non-exit teardown path all want symmetric destruction.
        mLlamaPools.clear();                         // ContextPool dtor frees every pooled ctx
        mWhisperPools.clear();                       // WhisperPool dtor runs whisper_free per slot
        for (auto& [_h, r] : mOcrRec) delete r.session;
        mOcrRec.clear();
        for (auto& [h, m] : mModels) {
            if (m.ctx) llama_free(m.ctx);
            if (m.model) llama_model_free(m.model);
            // m.wctx is the legacy single-whisper pointer. v0.6.2 moved
            // whisper state into mWhisperPools above; the pointer here is
            // a dangling-after-pool-clear reference in newer paths and
            // never set in older ones. Do NOT whisper_free it — would
            // double-free against the pool.
            delete m.ortSession;
            if (m.mtmdCtx) mtmd_free(m.mtmdCtx);
        }
        mModels.clear();
        llama_backend_free();
    }

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
    static bool readWav16(const std::string& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        char riff[12];
        f.read(riff, 12);
        if (f.gcount() != 12) return false;
        if (std::string(riff, 4) != "RIFF" || std::string(riff + 8, 4) != "WAVE") return false;
        int16_t channels = 0; int32_t sampleRate = 0; int16_t bits = 0;
        std::vector<char> data;
        while (f) {
            char id[4]; int32_t sz = 0;
            f.read(id, 4); if (f.gcount() != 4) break;
            f.read((char*)&sz, 4); if (f.gcount() != 4) break;
            std::string chunkId(id, 4);
            if (chunkId == "fmt ") {
                std::vector<char> fmt(sz);
                f.read(fmt.data(), sz);
                if (sz < 16) return false;
                channels   = *(int16_t*)(fmt.data() + 2);
                sampleRate = *(int32_t*)(fmt.data() + 4);
                bits       = *(int16_t*)(fmt.data() + 14);
            } else if (chunkId == "data") {
                data.resize(sz);
                f.read(data.data(), sz);
                break;
            } else {
                f.seekg(sz, std::ios::cur);
            }
        }
        if (channels != 1 || sampleRate != 16000 || bits != 16 || data.empty()) return false;
        const int16_t* pcm = (const int16_t*)data.data();
        size_t n = data.size() / 2;
        out.resize(n);
        for (size_t i = 0; i < n; ++i) out[i] = (float)pcm[i] / 32768.0f;
        return true;
    }

    ::ndk::ScopedAStatus loadWhisper(const std::string& modelPath, int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus submitTranscribe(int64_t modelHandle,
                                          const std::string& audioPath,
                                          const std::shared_ptr<IOirWorkerCallback>& cb,
                                          int64_t* _aidl_return) override;

    // ========================================================================
    // ONNX Runtime integration — state as of v0.6.
    //
    // loadOnnx() is fully implemented — it creates an ORT session from an
    // on-disk .onnx model and tracks it in mModels under the same budget/LRU
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

    ::ndk::ScopedAStatus unload(int64_t modelHandle) override {
        std::lock_guard<std::mutex> lk(mLock);
        auto it = mModels.find(modelHandle);
        if (it == mModels.end()) return ::ndk::ScopedAStatus::ok();
        // v0.6.2: whisper contexts are owned by the WhisperPool now —
        // erasing the pool runs its destructor which calls whisper_free
        // on each slot. lm.wctx points at one of those ctxs so we must
        // NOT call whisper_free on it directly or we'd double-free.
        mWhisperPools.erase(modelHandle);
        it->second.wctx = nullptr;
        // v0.6 Phase A: destroy the context pool (frees all pooled ctx)
        // in addition to the legacy single ctx pointer.
        mLlamaPools.erase(modelHandle);
        // v0.6 Phase B: drop cached OCR rec session for this handle.
        {
            auto oit = mOcrRec.find(modelHandle);
            if (oit != mOcrRec.end()) {
                delete oit->second.session;
                mOcrRec.erase(oit);
            }
        }
        if (it->second.ctx) llama_free(it->second.ctx);
        if (it->second.model) llama_model_free(it->second.model);
        delete it->second.ortSession;  // v0.4 H2/H3: safe on nullptr
        if (it->second.mtmdCtx) mtmd_free(it->second.mtmdCtx);  // v0.4 H6
        mBudget.subResident(it->second.sizeBytes);
        LOG(INFO) << "oird: unloaded handle=" << modelHandle << " path=" << it->second.path
                  << " resident=" << (mBudget.totalBytes() >> 20) << "MB";
        mModels.erase(it);
        return ::ndk::ScopedAStatus::ok();
    }

    ::ndk::ScopedAStatus submit(int64_t modelHandle,
                                const std::string& prompt,
                                int32_t maxTokens,
                                float temperature,
                                const std::shared_ptr<IOirWorkerCallback>& callback,
                                int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus cancel(int64_t requestHandle) override {
        std::lock_guard<std::mutex> lk(mLock);
        auto it = mActiveRequests.find(requestHandle);
        if (it != mActiveRequests.end()) {
            it->second->store(true);
            LOG(INFO) << "oird: cancel requested for handle=" << requestHandle;
        }
        return ::ndk::ScopedAStatus::ok();
    }

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
    ::ndk::ScopedAStatus setConfig(int32_t memoryBudgetMb, int32_t warmTtlSeconds) override {
        std::lock_guard<std::mutex> lk(mLock);
        mBudget.setBudgetMb(memoryBudgetMb);
        mWarmTtlSeconds = warmTtlSeconds;
        LOG(INFO) << "oird: config budget=" << memoryBudgetMb << "MB warm_ttl=" << warmTtlSeconds << "s";
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.5 V7: per-capability tuning via (key, float-value) pushed by OIRService
    // at attachWorker. Unknown keys are logged and ignored so OEMs can add new
    // knobs in their config files ahead of runtime support without errors.
    ::ndk::ScopedAStatus setCapabilityFloat(const std::string& key, float value) override {
        std::lock_guard<std::mutex> lk(mLock);
        if (key == "vision.detect.score_threshold") {
            mDetectScoreThresh = value;
        } else if (key == "vision.detect.iou_threshold") {
            mDetectIouThresh = value;
        } else if (key == "audio.vad.voice_threshold") {
            mVadVoiceThreshold = value;
        } else if (key == "audio.vad.sample_rate_hz") {
            // Float-typed AIDL; cast back to int.
            mVadSampleRateHz = (int32_t)value;
        } else if (key == "audio.vad.window_samples") {
            mVadWindowSamples = (int32_t)value;
        } else if (key == "audio.vad.context_samples") {
            mVadContextSamples = (int32_t)value;
        } else if (key == "text.complete.n_ctx") {
            mTextCompleteNCtx = (int32_t)value;
        } else if (key == "text.complete.max_tokens") {
            mTextCompleteMaxTokens = (int32_t)value;
        } else if (key == "text.embed.n_ctx") {
            mTextEmbedNCtx = (int32_t)value;
        } else if (key == "vision.describe.n_ctx") {
            mVisionDescribeNCtx = (int32_t)value;
        } else if (key == "vision.describe.n_batch") {
            mVisionDescribeNBatch = (int32_t)value;
        } else if (key == "vision.describe.max_tokens") {
            mVisionDescribeMaxTokens = (int32_t)value;
        } else if (key == "vision.embed.input_size") {
            mVisionEmbedInputSize = (int32_t)value;
        } else if (key == "vision.embed.normalize_mean") {
            mVisionEmbedNormMean = value;
        } else if (key == "vision.embed.normalize_std") {
            mVisionEmbedNormStd = value;
        } else if (key == "vision.detect.input_size") {
            mVisionDetectInputSize = (int32_t)value;
        } else if (key == "image.max_pixels") {
            // v0.7 hardening — cap on decoded JPEG/PNG pixel count to
            // protect oird from pathological untrusted images. 0 disables
            // the cap. Default kDefaultMaxImagePixels = 16M (~48 MB RGB).
            mImageMaxPixels = (size_t)value;
        } else if (key == "audio.synthesize.sample_rate_hz") {
            mAudioSynthesizeSampleRate = (int32_t)value;
        } else if (key == "audio.synthesize.length_scale") {
            mAudioSynthesizeLengthScale = value;
        } else if (key == "audio.synthesize.noise_scale") {
            mAudioSynthesizeNoiseScale = value;
        } else if (key == "text.complete.contexts_per_model") {
            // v0.6 Phase A: per-capability pool sizes. Clamped to [1,16]
            // to avoid runaway memory on config typos.
            int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
            mTextCompleteContextsPerModel = n;
        } else if (key == "text.embed.contexts_per_model") {
            int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
            mTextEmbedContextsPerModel = n;
        } else if (key == "vision.describe.contexts_per_model") {
            int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
            mVisionDescribeContextsPerModel = n;
        } else if (key == "text.complete.acquire_timeout_ms") {
            int32_t n = (int32_t)value; if (n < 100) n = 100;
            mTextCompleteAcquireTimeoutMs = n;
        } else if (key == "text.embed.acquire_timeout_ms") {
            int32_t n = (int32_t)value; if (n < 100) n = 100;
            mTextEmbedAcquireTimeoutMs = n;
        } else if (key == "vision.describe.acquire_timeout_ms") {
            int32_t n = (int32_t)value; if (n < 100) n = 100;
            mVisionDescribeAcquireTimeoutMs = n;
        } else if (key == "audio.transcribe.contexts_per_model") {
            int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 8) n = 8;
            mAudioTranscribeContextsPerModel = n;
        } else if (key == "audio.transcribe.acquire_timeout_ms") {
            int32_t n = (int32_t)value; if (n < 100) n = 100;
            mAudioTranscribeAcquireTimeoutMs = n;
        } else if (key == "text.complete.priority") {
            mTextCompletePriority = (int32_t)value;
        } else if (key == "text.embed.priority") {
            mTextEmbedPriority = (int32_t)value;
        } else if (key == "vision.describe.priority") {
            mVisionDescribePriority = (int32_t)value;
        } else if (key == "audio.transcribe.priority") {
            mAudioTranscribePriority = (int32_t)value;
        } else if (key == "audio.vad.priority") {
            mAudioVadPriority = (int32_t)value;
        } else if (key == "audio.synthesize.priority") {
            mAudioSynthesizePriority = (int32_t)value;
        } else if (key == "text.complete.temperature") {
            if (value >= 0.0f && value <= 2.0f) mTextCompleteTemperatureDefault = value;
        } else if (key == "text.complete.top_p") {
            if (value > 0.0f && value <= 1.0f) mTextCompleteTopP = value;
        } else if (key == "llama.batch_size") {
            int32_t n = (int32_t)value; if (n < 32) n = 32; if (n > 4096) n = 4096;
            mLlamaBatchSize = n;
        } else {
            LOG(WARNING) << "oird: unknown capability tuning key " << key
                         << " = " << value << " (ignored)";
            return ::ndk::ScopedAStatus::ok();
        }
        LOG(INFO) << "oird: tuning " << key << " = " << value;
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.5 V7 full: string-valued counterpart. Same routing pattern; unknown
    // keys logged and ignored.
    ::ndk::ScopedAStatus setCapabilityString(const std::string& key,
                                             const std::string& value) override {
        std::lock_guard<std::mutex> lk(mLock);
        if (key == "audio.transcribe.whisper_language") {
            mAudioTranscribeLanguage = value;
        } else if (key == "vision.detect.family") {
            mVisionDetectFamily = value;
        } else if (key == "vision.detect.normalize") {
            mVisionDetectNormalize = value;
        } else {
            LOG(WARNING) << "oird: unknown capability tuning string key " << key
                         << " = " << value << " (ignored)";
            return ::ndk::ScopedAStatus::ok();
        }
        LOG(INFO) << "oird: tuning " << key << " = \"" << value << "\"";
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.4 S3: mark a model as warm — unevictable for warm_ttl_seconds past this call.
    ::ndk::ScopedAStatus warm(int64_t modelHandle) override {
        std::lock_guard<std::mutex> lk(mLock);
        auto it = mModels.find(modelHandle);
        if (it != mModels.end()) {
            it->second.warmUntilMs = currentTimeMs() + (int64_t)mWarmTtlSeconds * 1000;
            LOG(INFO) << "oird: warm handle=" << modelHandle
                      << " until=" << it->second.warmUntilMs;
        }
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.6.3: per-handle runtime snapshot. Pairs with getMemoryStats
    // (model-level RSS totals) — this one exposes the live pool state
    // ContextPool + WhisperPool expose as getters. Returned as a
    // newline-separated TSV so `cmd oir memory` can print verbatim
    // without needing a new parcelable layout.
    ::ndk::ScopedAStatus dumpRuntimeStats(std::string* _aidl_return) override {
        std::lock_guard<std::mutex> lk(mLock);
        std::string out;
        const int64_t MB = 1024 * 1024;
        for (const auto& [h, m] : mModels) {
            int32_t poolSize = 0, busy = 0, waiting = 0;
            const char* backend = "?";
            if (m.isWhisper) {
                backend = "whisper";
                auto pit = mWhisperPools.find(h);
                if (pit != mWhisperPools.end() && pit->second) {
                    poolSize = static_cast<int32_t>(pit->second->size());
                    busy     = pit->second->busyCount();
                    waiting  = pit->second->waitingCount();
                }
            } else if (m.isVlm) {
                backend = "mtmd";
                auto pit = mLlamaPools.find(h);
                if (pit != mLlamaPools.end() && pit->second) {
                    poolSize = pit->second->size();
                    busy     = pit->second->busyCount();
                    waiting  = pit->second->waitingCount();
                }
            } else if (m.isOnnx || m.isVad) {
                backend = "ort";  // ORT has no pool abstraction by design
            } else if (m.isVisionEmbed) {
                backend = "ort";
            } else {
                // llama — text.complete / text.embed / text.translate.
                backend = m.isEmbedding ? "llama_embed" : "llama";
                auto pit = mLlamaPools.find(h);
                if (pit != mLlamaPools.end() && pit->second) {
                    poolSize = pit->second->size();
                    busy     = pit->second->busyCount();
                    waiting  = pit->second->waitingCount();
                }
            }
            out += std::to_string(h);
            out += '\t'; out += backend;
            out += '\t'; out += std::to_string(poolSize);
            out += '\t'; out += std::to_string(busy);
            out += '\t'; out += std::to_string(waiting);
            out += '\t'; out += std::to_string(m.sizeBytes / MB);
            out += '\t'; out += m.path;
            out += '\n';
        }
        *_aidl_return = std::move(out);
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.6.3: worker-side file stat fallback used by OIRService when
    // system_server's SELinux scope can't `getattr` an `oir_model_*_file`
    // label directly. Rechecked through oird's domain which does have
    // the read rule platform-wide. Path resolution is native stat();
    // symlinks resolve; permission-denied maps to false.
    ::ndk::ScopedAStatus fileIsReadable(const std::string& path, bool* _aidl_return) override {
        if (path.empty()) { *_aidl_return = false; return ::ndk::ScopedAStatus::ok(); }
        struct stat st{};
        if (::stat(path.c_str(), &st) != 0) {
            *_aidl_return = false;
            return ::ndk::ScopedAStatus::ok();
        }
        if (!S_ISREG(st.st_mode)) {
            *_aidl_return = false;
            return ::ndk::ScopedAStatus::ok();
        }
        // Regular file + stat succeeded. Check read-openable — the
        // caller's real question is "can inference load this?", which
        // requires open-for-read. Closing immediately; we're not
        // actually loading here.
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { *_aidl_return = false; return ::ndk::ScopedAStatus::ok(); }
        ::close(fd);
        *_aidl_return = true;
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.4 S2: snapshot of worker memory state for cmd oir dumpsys memory.
    ::ndk::ScopedAStatus getMemoryStats(MemoryStats* _aidl_return) override {
        std::lock_guard<std::mutex> lk(mLock);
        const int64_t MB = 1024 * 1024;
        int64_t totalBytes = 0;
        std::vector<std::string> paths;
        std::vector<int32_t> sizes;
        std::vector<int64_t> loadTimes;
        std::vector<int64_t> accessTimes;
        for (const auto& [h, m] : mModels) {
            paths.push_back(m.path);
            sizes.push_back(static_cast<int32_t>(m.sizeBytes / MB));
            loadTimes.push_back(m.loadTimestampMs);
            accessTimes.push_back(m.lastAccessMs);
            totalBytes += m.sizeBytes;
        }
        _aidl_return->modelCount = static_cast<int32_t>(mModels.size());
        _aidl_return->residentMb = static_cast<int32_t>(totalBytes / MB);
        _aidl_return->modelPaths = std::move(paths);
        _aidl_return->modelSizesMb = std::move(sizes);
        _aidl_return->loadTimestampMs = std::move(loadTimes);
        _aidl_return->lastAccessMs = std::move(accessTimes);
        return ::ndk::ScopedAStatus::ok();
    }

private:
    // v0.7: in-progress load registry moved to runtime/load_registry.{h,cpp}.
    // See LoadRegistry header for the v0.6.9 dedup rationale and the
    // mutex-borrowing contract (registry uses caller's mLock for cv.wait).
    LoadRegistry mLoadRegistry;

    void runInference(int64_t modelHandle,
                      int64_t handle,
                      std::string prompt,
                      int32_t maxTokens,
                      float temperature,
                      std::shared_ptr<IOirWorkerCallback> cb,
                      std::shared_ptr<std::atomic_bool> cancelled,
                      std::shared_ptr<InFlightGuard> guard);

    void cleanupRequest(int64_t handle) {
        std::lock_guard<std::mutex> lk(mLock);
        mActiveRequests.erase(handle);
    }

    // v0.4 S2-B: decrement inFlightCount for the model that ran this request.
    // Caller passes modelHandle since we don't track request->model otherwise.
    void releaseInflight(int64_t modelHandle) {
        std::lock_guard<std::mutex> lk(mLock);
        auto it = mModels.find(modelHandle);
        if (it != mModels.end() && it->second.inFlightCount > 0) {
            it->second.inFlightCount--;
        }
    }

    // v0.7: RAII helper. Caller MUST hold mLock and have already validated
    // that lm refers to a live model. Increments lm.inFlightCount and returns
    // a shared_ptr<InFlightGuard> that owns the matching decrement (via
    // releaseInflight() in its destructor). Wrapped in shared_ptr so it can
    // be captured into Scheduler::Task lambdas — std::function requires
    // copy-constructible captures.
    std::shared_ptr<InFlightGuard> acquireInflightLocked(LoadedModel& lm,
                                                          int64_t modelHandle) {
        lm.inFlightCount++;
        return std::make_shared<InFlightGuard>(this, modelHandle);
    }

    // v0.6.3: resolve the scheduler priority for a capability name.
    // Uses the existing per-capability mXxxPriority knobs; unknown caps
    // default to PRIO_NORMAL. Called from submit* methods when building
    // the scheduler enqueue.
    int32_t priorityForCapability(const std::string& cap) {
        std::lock_guard<std::mutex> lk(mLock);
        if (cap == "audio.transcribe")  return mAudioTranscribePriority;
        if (cap == "audio.vad")         return mAudioVadPriority;
        if (cap == "audio.synthesize")  return mAudioSynthesizePriority;
        if (cap == "text.complete"
                || cap == "text.translate") return mTextCompletePriority;
        if (cap == "text.embed"
                || cap == "text.classify"
                || cap == "text.rerank")    return mTextEmbedPriority;
        if (cap == "vision.describe")   return mVisionDescribePriority;
        if (cap == "vision.embed"
                || cap == "vision.detect"
                || cap == "vision.ocr")     return ContextPool::PRIO_NORMAL;
        return ContextPool::PRIO_NORMAL;
    }

    std::mutex mLock;
    // v0.6.2: mInferenceMutex (the global "one inference at a time" lock
    // from v0.4/v0.5) is fully retired. ContextPool serves llama + VLM;
    // WhisperPool serves whisper; ORT Run() is thread-safe by design. The
    // initial validation + handle bookkeeping still runs under mLock but
    // releases before any Run()/whisper_full()/llama_decode().
    std::unordered_map<int64_t, LoadedModel> mModels;
    // v0.6 Phase A: per-model context pool for llama-backed capabilities
    // (text.complete / text.embed / vision.describe). Keyed by modelHandle.
    // Pool created at load time; destroyed on unload / LRU eviction.
    std::unordered_map<int64_t, std::unique_ptr<ContextPool>> mLlamaPools;

    // v0.6.2: per-model whisper_context pool for audio.transcribe. Same
    // lifecycle as mLlamaPools — created at loadWhisper, destroyed on
    // unload / LRU eviction. Size controlled by
    // mAudioTranscribeContextsPerModel (default 2).
    std::unordered_map<int64_t, std::unique_ptr<WhisperPool>> mWhisperPools;

    // v0.6.3: cross-backend scheduler. Every submit* enqueues into here
    // with its capability's configured priority; the scheduler's worker
    // threads pull in priority order and run the inference body on
    // whatever backend the task targets. Built in the constructor;
    // destroyed first in the destructor so no worker thread outlives
    // model state.
    std::unique_ptr<Scheduler> mScheduler;

    // v0.6 Phase B: per-det-handle cache for the vision.ocr recognizer
    // ORT session + vocabulary. Lazy-loaded on first submitOcr after
    // sidecar existence check; stays resident for the lifetime of the
    // det model. Cleared when the det model evicts.
    struct OcrRec {
        Ort::Session* session = nullptr;
        std::vector<std::string> vocab;   // idx 0 = CTC blank
    };
    std::unordered_map<int64_t, OcrRec> mOcrRec;
    int64_t mNextModelHandle = 1;
    int64_t mNextRequestHandle = 1;

    // v0.4 S2-B/S3 scheduler config — set by OIRService.attachWorker via setConfig()
    // v0.7: budget fields (mBudgetMb / mTotalBytes / mEvictionCount) moved to
    // runtime/Budget. mWarmTtlSeconds stays here — it's a per-handle TTL knob,
    // not part of resident-memory accounting.
    Budget  mBudget;
    int32_t mWarmTtlSeconds = 60;
    std::unordered_map<int64_t, std::shared_ptr<std::atomic_bool>> mActiveRequests;

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
    int32_t mTextCompleteNCtx      = 2048;
    int32_t mTextCompleteMaxTokens = 256;
    int32_t mTextEmbedNCtx         = 512;
    int32_t mVisionDescribeNCtx    = 4096;
    int32_t mVisionDescribeNBatch  = 2048;
    int32_t mVisionDescribeMaxTokens = 256;
    // v0.6 Phase A: per-capability pool configuration. Pool size drives
    // KV memory (pool_size × n_ctx × n_layer × head_dim × 4 bytes); OEMs
    // with tight budgets drop each below its default.
    int32_t mTextCompleteContextsPerModel    = 4;  // common concurrent chat
    int32_t mTextEmbedContextsPerModel       = 2;  // fast; 2 is usually plenty
    int32_t mVisionDescribeContextsPerModel  = 1;  // VLMs are 4GB+; pool=1 by default
    // v0.6.2: whisper context pool default. Whisper-tiny is ~40MB; per-ctx
    // state is small (decoder buffers, sampling state) so 2 is the right
    // balance between memory cost and concurrent transcribe support
    // (dictation + one-shot audio upload is a common 2-stream scenario).
    int32_t mAudioTranscribeContextsPerModel = 2;
    // Acquire timeouts — max time a caller waits for a free pool slot.
    // Hitting the timeout returns OIRError::TIMEOUT to the app; apps retry
    // or back off. Protects against pool wedging on a stuck inference.
    int32_t mTextCompleteAcquireTimeoutMs   = 30000;
    int32_t mTextEmbedAcquireTimeoutMs      = 10000;
    int32_t mVisionDescribeAcquireTimeoutMs = 60000;
    // v0.6.2: transcribe can legitimately take 10-30 s for longer audio;
    // a 60 s acquire timeout is generous but matches the single-ctx
    // worst-case that v0.6 already tolerated.
    int32_t mAudioTranscribeAcquireTimeoutMs = 60000;
    // Priority bands — lower = higher priority (POSIX niceness convention).
    // 0 = audio realtime, 10 = normal (text/vision), 20 = low/batch.
    // When audio.* and text.* contend for the same pool, audio is granted
    // first by release() hand-off.
    int32_t mTextCompletePriority     = ContextPool::PRIO_NORMAL;
    int32_t mTextEmbedPriority        = ContextPool::PRIO_NORMAL;
    int32_t mVisionDescribePriority   = ContextPool::PRIO_NORMAL;
    int32_t mAudioTranscribePriority  = ContextPool::PRIO_AUDIO_REALTIME;
    int32_t mAudioVadPriority         = ContextPool::PRIO_AUDIO_REALTIME;
    int32_t mAudioSynthesizePriority  = ContextPool::PRIO_AUDIO_REALTIME;
    // Sampling defaults — apps pass temperature via submit() AIDL; when
    // caller passes <0 we fall back to these defaults. top_p is not yet
    // exposed via AIDL so the default is the effective value.
    float   mTextCompleteTemperatureDefault = 0.7f;
    float   mTextCompleteTopP               = 0.9f;
    // Batch size for llama_batch_init — affects prefill speed. Higher = more
    // memory during prefill. Default 512 matches upstream llama.cpp recommended.
    int32_t mLlamaBatchSize = 512;
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
    Ort::SessionOptions makeOrtSessionOptions(bool isDetection) const {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(std::max(2, (int)sysconf(_SC_NPROCESSORS_ONLN) / 2));
        so.SetGraphOptimizationLevel(
            isDetection ? GraphOptimizationLevel::ORT_ENABLE_ALL
                        : GraphOptimizationLevel::ORT_ENABLE_BASIC);
        return so;
    }
    void ensureOrtEnv() {
        if (!mOrtEnv) {
            mOrtEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "oird");
        }
    }
};

// InFlightGuard out-of-class definitions (need full OirdService for
// releaseInflight()).
inline InFlightGuard::~InFlightGuard() { release(); }

inline void InFlightGuard::release() {
    if (mActive && mSvc) mSvc->releaseInflight(mModelHandle);
    mActive = false;
}


} // namespace oird
