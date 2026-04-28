// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// runtime/runtime.h — cross-cutting daemon state shared by all backends.
//
// v0.7-post step 1 of backend decomposition: state that doesn't belong to
// any specific backend (mutex, model handle table, budget, scheduler,
// load-dedup registry, request handle counters, in-flight request set,
// warm TTL config) moves into one struct passed by reference to each
// backend.
//
// Future steps create LlamaBackend / WhisperBackend / OrtBackend /
// VlmBackend, each owning its per-handle pool maps + per-capability
// knobs, and each holding a Runtime& for the cross-cutting bits. The
// AIDL stub OirdService keeps a Runtime + the four backends as members
// and routes by capability.
//
// Thread safety: members are NOT individually synchronized. The daemon's
// `mLock` (the first member) is the single mutex protecting model state
// across all backends. Backends acquire `rt.mLock` before touching any
// shared field. Budget and LoadRegistry have their own contracts (see
// their headers); Scheduler is internally synchronized.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <llama.h>
#include <whisper.h>
#include <mtmd.h>
#include <onnxruntime_cxx_api.h>

#include "runtime/budget.h"
#include "runtime/load_registry.h"
#include "sched/scheduler.h"

namespace oird {

// LoadedModel struct: per-handle model state visible to all backends. Lives
// here (not in a backend) because the LRU evictor needs to walk all models
// regardless of which backend loaded them. Backend-specific pool/handle
// lookups are keyed by `handle` against the backend's own maps.
struct LoadedModel {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
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

struct Runtime {
    std::mutex mLock;
    std::unordered_map<int64_t, LoadedModel> mModels;
    Budget mBudget;
    LoadRegistry mLoadRegistry;
    std::unique_ptr<Scheduler> mScheduler;
    int64_t mNextModelHandle = 1;
    int64_t mNextRequestHandle = 1;
    std::unordered_map<int64_t, std::shared_ptr<std::atomic_bool>> mActiveRequests;
    // v0.4 S2-B/S3: warm TTL per-model after submit (ms = warmUntilMs).
    // Cross-capability config so it lives in Runtime, not a backend.
    int32_t mWarmTtlSeconds = 60;
};

} // namespace oird
