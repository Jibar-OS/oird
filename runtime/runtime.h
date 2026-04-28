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
// step 2a: InFlightGuard + releaseInflight + acquireInflightLocked move
// from OirdService to Runtime. This decouples the inflight machinery
// from the AIDL stub class so backend classes can acquire guards through
// their Runtime& reference without needing access to OirdService internals.
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

// Forward decl so Runtime can produce InFlightGuards holding back-pointers
// to itself; full class definition appears below Runtime.
class InFlightGuard;

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
    Ort::Session* ortSession = nullptr;
    bool    isOnnx = false;         // v0.4 H2/H3: route to ORT
    bool    onnxIsDetection = false; // v0.4 H3: detection vs synthesis model
    bool    isVisionEmbed = false;  // v0.4 H4-B: ORT session is a vision encoder (SigLIP / CLIP)
    mtmd_context* mtmdCtx = nullptr;
    bool    isVlm = false;
    std::vector<std::string> detectClassLabels;
    bool    isVad = false;
    bool    hasLlamaPool = false;
};

class Runtime {
public:
    // Cross-cutting daemon state. Public for direct backend access; backends
    // are part of the daemon's internal architecture, not external API.
    std::mutex mLock;
    std::unordered_map<int64_t, LoadedModel> mModels;
    Budget mBudget;
    LoadRegistry mLoadRegistry;
    std::unique_ptr<Scheduler> mScheduler;
    int64_t mNextModelHandle = 1;
    int64_t mNextRequestHandle = 1;
    std::unordered_map<int64_t, std::shared_ptr<std::atomic_bool>> mActiveRequests;
    int32_t mWarmTtlSeconds = 60;

    // v0.7: decrement inFlightCount for the model that ran a request.
    // Called from InFlightGuard's destructor (or explicit release()).
    // Internally locks mLock; safe to call from any thread.
    void releaseInflight(int64_t modelHandle);

    // v0.7: RAII-creating helper. Caller MUST hold mLock and have already
    // validated that lm refers to a live model. Increments lm.inFlightCount
    // and returns a shared_ptr<InFlightGuard> that owns the matching
    // decrement (via releaseInflight() in its destructor). Wrapped in
    // shared_ptr so the guard can be captured into Scheduler::Task lambdas
    // (std::function requires copy-constructible captures).
    std::shared_ptr<InFlightGuard> acquireInflightLocked(LoadedModel& lm,
                                                          int64_t modelHandle);
};

// v0.7: RAII guard for LoadedModel::inFlightCount. Replaces the v0.4
// pattern of paired `it->second.inFlightCount++;` ... `releaseInflight(h);`
// calls — same invariant, but enforced by C++ object lifetime instead of
// comments on every submit path.
//
// Acquired via Runtime::acquireInflightLocked(lm, handle); the destructor
// (or explicit release()) calls Runtime::releaseInflight() exactly once.
// Wrapped in std::shared_ptr at every site so the guard can be captured
// into Scheduler::Task (std::function<void()>, which requires the
// captured lambda to be copy-constructible).
//
// v0.6.8 ordering note: many submit paths must release the inflight BEFORE
// firing the terminal binder callback (onComplete/onError) so a stalled
// callback can't pin the model resident. Call guard->release() explicitly
// at that point; the destructor then becomes a no-op. If callers forget
// the explicit release, the destructor is the safety net — slightly later
// release, but still no leak.
class InFlightGuard {
public:
    InFlightGuard(Runtime* rt, int64_t modelHandle)
            : mRt(rt), mModelHandle(modelHandle), mActive(true) {}
    ~InFlightGuard() { release(); }

    InFlightGuard(const InFlightGuard&) = delete;
    InFlightGuard& operator=(const InFlightGuard&) = delete;
    InFlightGuard(InFlightGuard&&) = delete;
    InFlightGuard& operator=(InFlightGuard&&) = delete;

    // Decrement now (idempotent). Use to control v0.6.8 ordering of release
    // vs terminal callback. Subsequent destructor is a no-op.
    void release() {
        if (mActive && mRt) mRt->releaseInflight(mModelHandle);
        mActive = false;
    }

private:
    Runtime* mRt;
    int64_t  mModelHandle;
    bool     mActive;
};

// Inline definitions referencing InFlightGuard (must follow its declaration).
inline void Runtime::releaseInflight(int64_t modelHandle) {
    std::lock_guard<std::mutex> lk(mLock);
    auto it = mModels.find(modelHandle);
    if (it != mModels.end() && it->second.inFlightCount > 0) {
        it->second.inFlightCount--;
    }
}

inline std::shared_ptr<InFlightGuard> Runtime::acquireInflightLocked(
        LoadedModel& lm, int64_t modelHandle) {
    lm.inFlightCount++;
    return std::make_shared<InFlightGuard>(this, modelHandle);
}

} // namespace oird
