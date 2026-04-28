/*
 * Copyright (C) 2026 The OpenIntelligenceRuntime Project
 * Licensed under the Apache License, Version 2.0
 *
 * oird — OIR native inference worker. Runs in its own process (not
 * system_server) so native crashes can't take down the platform
 * (DESIGN.md §4.1). Registers the "oir_worker" binder service; only
 * system_server is expected to call in.
 *
 * Uses the AIDL NDK backend (std::string, std::shared_ptr, ndk::ScopedAStatus)
 * rather than the legacy CPP backend (String16, sp<>, android::binder::Status).
 *
 * Inference loop adapted from AAOSP's llm_jni.cpp (b4547 API). AAOSP ran
 * inside system_server via JNI; OIR inlines the same logic into this
 * native worker — no JNI needed.
 */

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
#include <fcntl.h>    // v0.6.3: open()/O_RDONLY for IOirWorker::fileIsReadable
#include <unordered_map>
#include <vector>
#include <functional>   // v0.6.3: std::function for Scheduler::Task
#include <queue>        // v0.6.3: std::priority_queue for Scheduler

#include <aidl/com/android/server/oir/BnOirWorker.h>
#include <aidl/com/android/server/oir/IOirWorkerCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerVectorCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerAudioCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerBboxCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerRealtimeBooleanCallback.h>
#include <aidl/com/android/server/oir/MemoryStats.h>

using aidl::com::android::server::oir::BnOirWorker;
using aidl::com::android::server::oir::IOirWorkerCallback;
using aidl::com::android::server::oir::IOirWorkerVectorCallback;
using aidl::com::android::server::oir::IOirWorkerAudioCallback;
using aidl::com::android::server::oir::IOirWorkerBboxCallback;
using aidl::com::android::server::oir::IOirWorkerRealtimeBooleanCallback;
using aidl::com::android::server::oir::MemoryStats;


#include "image_decode.h"

namespace {

constexpr int32_t W_MODEL_ERROR         = 1;
constexpr int32_t W_CANCELLED           = 2;
constexpr int32_t W_INVALID_INPUT       = 3;
constexpr int32_t W_INSUFFICIENT_MEMORY = 4;
constexpr int32_t W_TIMEOUT             = 5;
// v0.6 Phase A: OEM model shape mismatch (load-time). Distinct from
// W_MODEL_ERROR so SDK can map to typed OirModelIncompatibleException
// and apps can fail gracefully + fall back to OEM-bake guidance.
// Must match OIRError::MODEL_INCOMPATIBLE in the Java AIDL.
constexpr int32_t W_MODEL_INCOMPATIBLE  = 11;
// v0.6 Phase A: capability declared in capabilities.xml but runtime
// path isn't complete (e.g. audio.synthesize until G2P ships in
// v0.6 Phase C). OIRService translates to Java
// CAPABILITY_UNAVAILABLE_NO_MODEL=9.
constexpr int32_t W_CAPABILITY_UNSUPPORTED = 12;
// v0.6 Phase B: capability dispatches successfully, but the model /
// companion-sidecar the capability needs is not present on this device.
// OIRService maps to Java OIRError.CAPABILITY_UNAVAILABLE_NO_MODEL=9.
constexpr int32_t W_CAPABILITY_UNAVAILABLE_NO_MODEL = 9;

// Inline batch helpers lifted from AAOSP's llm_jni.cpp.
static void llama_batch_clear_local(struct llama_batch& batch) {
    batch.n_tokens = 0;
}

static void llama_batch_add_local(
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

// Forward declaration — full class defined below currentTimeMs().
class ContextPool;

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

static int64_t currentTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

// ============================================================================
// v0.6 Phase A: per-model context pool (priority-aware, bounded-wait)
//
// v0.5 and earlier serialized every llama-backed submit through a single
// global mInferenceMutex — two concurrent text.complete calls on the same
// model blocked each other. v0.6 replaces that with a pool of contexts per
// model so concurrent submits progress in parallel.
//
// Pool slot carries both llama_context and mtmd_context (non-null only for
// VLM pools) so vision.describe benefits from pooling too — v0.5 held a
// single mtmd+llama pair per VLM, which meant a second describe() call
// waited for the first.
//
// Scheduling choices made for platform use:
//   1. Priority-aware wait queue. Waiters are ordered by (priority, enqueue
//      time); when a slot frees, release() hands it DIRECTLY to the
//      highest-priority longest-waiting acquirer. This is non-preemptive —
//      an in-flight request runs to completion — but queue ordering makes
//      audio preempt text at the dispatcher level without starving anyone.
//      Priority: 0 = highest (realtime audio), 10 = normal (text/vision),
//      20 = background (batch). Matches POSIX niceness convention (lower
//      = higher priority).
//   2. Bounded acquire. Every caller passes a deadline. On timeout the
//      waiter removes itself from the queue and returns invalid Lease — no
//      caller hangs forever if the pool wedges.
//   3. Direct hand-off on release. release() transfers the slot to the
//      front-of-queue waiter without dropping the lock, which avoids the
//      "thundering herd" of notify_all + re-contend.
//   4. Atomic busy count for cheap observability. dumpsys reads without
//      taking the pool's main mutex.
//   5. Graceful shutdown. The destructor flips mShutdown, wakes every
//      waiter, and waits for the queue to drain before freeing contexts.
//      No dangling pointer races with in-flight acquire()s.
//
// Memory: each context carries its own KV cache. For a 2048-ctx text
// model ~8 MB per slot on a ~500 MB 0.5B model; ~50 MB per slot on a 7B
// model at fp16. MemoryManager accounts pool KV via
// estimateKvBytesPerContext() (see below) — mTotalBytes += kv_bytes × pool_size
// at load time.
//
// Lifecycle:
//   - Created at load time (load / loadEmbed / loadWithMmproj).
//   - Leased via ContextLease (RAII); release on scope exit.
//   - Destroyed on model unload or LRU eviction — all contexts freed.
// ============================================================================

// Payload carried by each pool slot. For text.complete / text.embed, only
// `ctx` is set. For vision.describe (VLM), both `ctx` and `mctx` are set —
// the pair must be used together because mtmd_context binds to a specific
// llama_context at init time.
struct PooledContext {
    llama_context* ctx = nullptr;
    mtmd_context*  mctx = nullptr;
};

class ContextPool {
public:
    // Standard priority bands. OEMs override per-capability via
    // `<capability>.priority` knob (integer in oir_config.xml).
    static constexpr int PRIO_AUDIO_REALTIME = 0;    // audio.vad, audio.transcribe
    static constexpr int PRIO_NORMAL         = 10;   // text.*, vision.*
    static constexpr int PRIO_LOW            = 20;   // batch/background

    // Takes ownership of each slot's contexts; frees on destruction.
    explicit ContextPool(std::vector<PooledContext> contexts)
        : mSlots(contexts.size()) {
        // v0.7: empty-pool rejection at construction. A 0-slot pool would
        // accept submits, find no free slot, and block waiters forever.
        // Load callers already check for individual init failures and return
        // MODEL_ERROR — this is defense-in-depth against future regressions.
        if (contexts.empty()) {
            LOG(FATAL) << "ContextPool: refusing to construct with zero contexts "
                          "(would deadlock first acquire). Caller must validate "
                          "pooledCtxs before construction.";
        }
        for (size_t i = 0; i < contexts.size(); ++i) {
            mSlots[i].payload = contexts[i];
        }
    }

    ~ContextPool() {
        shutdown();  // wakes all waiters, waits for in-flight releases
        std::lock_guard<std::mutex> lk(mMtx);
        for (auto& s : mSlots) {
            if (s.payload.mctx) mtmd_free(s.payload.mctx);
            if (s.payload.ctx)  llama_free(s.payload.ctx);
            s.payload = {};
        }
    }

    ContextPool(const ContextPool&) = delete;
    ContextPool& operator=(const ContextPool&) = delete;

    struct Lease {
        llama_context* ctx  = nullptr;
        mtmd_context*  mctx = nullptr;
        int            slotIdx = -1;
        bool valid() const { return ctx != nullptr; }
    };

    // Acquire a slot honoring priority + bounded wait.
    //   priority: lower = higher (0 = audio realtime, 10 = normal, 20 = low)
    //   timeout: maximum time to wait for a slot; returns invalid on expiry
    //
    // Ownership model: Waiter is stack-allocated in this function. The
    // queue stores non-owning `Waiter*` pointers. acquire() unconditionally
    // removes its Waiter from the queue before returning — safe because
    // release()'s hand-off pops the waiter (see handOffNextLocked_), and
    // if that already happened the erase is a no-op.
    //
    // On timeout, caller should surface OIRError::TIMEOUT to the app.
    // On shutdown (pool destroyed), returns invalid immediately — callers
    // must hold model.inFlightCount > 0 to prevent shutdown mid-wait.
    Lease acquire(int priority, std::chrono::milliseconds timeout) {
        using clock = std::chrono::steady_clock;
        const auto wait_start = clock::now();
        const auto deadline = wait_start + timeout;

        std::unique_lock<std::mutex> lk(mMtx);
        if (mShutdown.load()) return {};

        // Fast path: nobody ahead of us AND a slot is free.
        if (mQueue.empty()) {
            int idx = findFreeSlotLocked_();
            if (idx >= 0) {
                Lease l = takeSlotLocked_(idx);
                return l;  // no wait logged for fast path
            }
        }

        // Slow path: queue up. Waiter on our stack; queue stores a raw ptr.
        Waiter w;
        w.priority  = priority;
        w.enqueueMs = currentTimeMs();
        w.id        = ++mNextWaiterId;
        insertSortedLocked_(&w);

        // Wait for: my grantedSlot set by release(), OR pool shutdown, OR deadline.
        bool gotSlot = w.cv.wait_until(lk, deadline, [&] {
            return w.grantedSlot >= 0 || mShutdown.load();
        });

        // Always remove self from queue on exit. If handOff already popped
        // us, this is a no-op. Cheap.
        removeWaiterByPtrLocked_(&w);

        if (mShutdown.load()) return {};
        if (!gotSlot || w.grantedSlot < 0) return {};  // timeout

        // Granted cleanly. Slot inUse + busy count already set by handoff.
        int idx = w.grantedSlot;
        mSlots[idx].lastUsedMs = currentTimeMs();
        auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - wait_start).count();
        // Log contention events so OEMs can tune pool size.
        if (wait_ms > 100) {
            LOG(INFO) << "oird: pool acquire waited " << wait_ms << "ms"
                      << " (priority=" << priority
                      << " size=" << mSlots.size()
                      << " busy_now=" << mBusyCount.load(std::memory_order_relaxed) << ")";
        }
        return {mSlots[idx].payload.ctx, mSlots[idx].payload.mctx, idx};
    }

    void release(int slotIdx) {
        if (slotIdx < 0) return;
        std::unique_lock<std::mutex> lk(mMtx);
        if (slotIdx >= (int)mSlots.size()) return;
        mSlots[slotIdx].inUse = false;
        mSlots[slotIdx].lastUsedMs = currentTimeMs();
        mBusyCount.fetch_sub(1, std::memory_order_relaxed);
        handOffNextLocked_(slotIdx);
    }

    // Wake all waiters with invalid Lease; subsequent acquires return
    // invalid immediately. Blocks until queue drains so destructor can
    // safely free contexts without dangling pointers.
    void shutdown() {
        std::unique_lock<std::mutex> lk(mMtx);
        if (mShutdown.exchange(true)) return;
        // Wake every waiter — each one will observe mShutdown and bail.
        for (auto* w : mQueue) w->cv.notify_one();
        // Wait for queue to drain. Waiters remove themselves as they wake.
        while (!mQueue.empty()) {
            mDrainedCv.wait_for(lk, std::chrono::milliseconds(100));
        }
    }

    int32_t size() const { return (int32_t)mSlots.size(); }

    // Atomic read — no mutex. Safe to call from dumpsys / monitoring.
    int32_t busyCount() const {
        return mBusyCount.load(std::memory_order_relaxed);
    }

    int32_t waitingCount() const {
        std::lock_guard<std::mutex> lk(mMtx);
        return (int32_t)mQueue.size();
    }

    // For dumpsys — per-slot last-used-ms to see which contexts are warm.
    std::vector<int64_t> lastUsedSnapshot() const {
        std::lock_guard<std::mutex> lk(mMtx);
        std::vector<int64_t> out;
        out.reserve(mSlots.size());
        for (auto& s : mSlots) out.push_back(s.lastUsedMs);
        return out;
    }

private:
    struct Slot {
        PooledContext payload;
        bool inUse = false;
        int64_t lastUsedMs = 0;
    };
    // Stack-allocated in acquire() — NOT owned by the queue. Queue holds
    // non-owning pointers. This matches how kernel wait queues and
    // pthread cond_wait internals work: the waiter's lifetime is tied to
    // the waiting function's stack frame.
    struct Waiter {
        int      priority = 10;
        int64_t  enqueueMs = 0;
        int64_t  id = 0;
        int      grantedSlot = -1;  // set by release() on hand-off
        std::condition_variable cv;
    };

    mutable std::mutex mMtx;
    std::vector<Slot> mSlots;
    // Queue sorted by (priority asc, enqueueMs asc). Front = next to grant.
    // Non-owning — Waiter lives on acquire()'s stack.
    std::deque<Waiter*> mQueue;
    std::atomic<int32_t> mBusyCount{0};
    std::atomic<bool> mShutdown{false};
    int64_t mNextWaiterId = 0;
    std::condition_variable mDrainedCv;  // signaled when a waiter removes itself

    int findFreeSlotLocked_() const {
        for (size_t i = 0; i < mSlots.size(); ++i) {
            if (!mSlots[i].inUse && mSlots[i].payload.ctx) return (int)i;
        }
        return -1;
    }

    Lease takeSlotLocked_(int idx) {
        mSlots[idx].inUse = true;
        mSlots[idx].lastUsedMs = currentTimeMs();
        mBusyCount.fetch_add(1, std::memory_order_relaxed);
        return {mSlots[idx].payload.ctx, mSlots[idx].payload.mctx, idx};
    }

    void insertSortedLocked_(Waiter* w) {
        // v0.7: FIFO tiebreaker via monotonic id. Two waiters with identical
        // (priority, enqueueMs) — possible when many submits land in the same
        // millisecond — preserve insertion order instead of relying on the
        // unspecified order of std::lower_bound on equal keys.
        auto it = std::lower_bound(
                mQueue.begin(), mQueue.end(), w,
                [](const Waiter* a, const Waiter* b) {
                    if (a->priority != b->priority) return a->priority < b->priority;
                    if (a->enqueueMs != b->enqueueMs) return a->enqueueMs < b->enqueueMs;
                    return a->id < b->id;
                });
        mQueue.insert(it, w);
    }

    void removeWaiterByPtrLocked_(Waiter* w) {
        for (auto it = mQueue.begin(); it != mQueue.end(); ++it) {
            if (*it == w) {
                mQueue.erase(it);
                mDrainedCv.notify_all();
                return;
            }
        }
    }

    // Called from release() path: hand the slot DIRECTLY to the front-of-
    // queue waiter AND pop the waiter from the queue in one locked step.
    // Popping here (rather than letting the waiter pop itself on wake)
    // prevents double-hand-off races where a subsequent release could
    // overwrite grantedSlot on a waiter that hadn't run yet.
    void handOffNextLocked_(int slotIdx) {
        if (mQueue.empty()) return;
        Waiter* w = mQueue.front();
        mQueue.pop_front();
        mSlots[slotIdx].inUse = true;
        mBusyCount.fetch_add(1, std::memory_order_relaxed);
        w->grantedSlot = slotIdx;
        w->cv.notify_one();
        mDrainedCv.notify_all();
    }
};

// RAII lease wrapper — release on scope exit guarantees no slot leak even
// on exception / early-return paths.
class ContextLease {
public:
    ContextLease() = default;
    ContextLease(ContextPool& pool, int priority, std::chrono::milliseconds timeout)
            : mPool(&pool) {
        auto l = pool.acquire(priority, timeout);
        mCtx  = l.ctx;
        mMCtx = l.mctx;
        mSlot = l.slotIdx;
    }
    ~ContextLease() {
        if (mPool && mSlot >= 0) mPool->release(mSlot);
    }
    ContextLease(const ContextLease&) = delete;
    ContextLease& operator=(const ContextLease&) = delete;
    ContextLease(ContextLease&& other) noexcept
        : mPool(other.mPool), mCtx(other.mCtx), mMCtx(other.mMCtx), mSlot(other.mSlot) {
        other.mPool = nullptr; other.mCtx = nullptr; other.mMCtx = nullptr; other.mSlot = -1;
    }
    ContextLease& operator=(ContextLease&& other) noexcept {
        if (this != &other) {
            if (mPool && mSlot >= 0) mPool->release(mSlot);
            mPool = other.mPool; mCtx = other.mCtx; mMCtx = other.mMCtx; mSlot = other.mSlot;
            other.mPool = nullptr; other.mCtx = nullptr; other.mMCtx = nullptr; other.mSlot = -1;
        }
        return *this;
    }
    bool valid() const { return mCtx != nullptr; }
    llama_context* ctx()  const { return mCtx;  }
    mtmd_context*  mctx() const { return mMCtx; }

private:
    ContextPool* mPool = nullptr;
    llama_context* mCtx = nullptr;
    mtmd_context*  mMCtx = nullptr;
    int mSlot = -1;
};

// v0.6 Phase A: ORT load-time shape validation.
//
// v0.5 and earlier assumed OEM-supplied ONNX models matched the hardcoded
// shape contract for each capability. Mismatched models SIGSEGV'd at
// inference time or silently produced garbage. v0.6 validates at load
// time and returns MODEL_INCOMPATIBLE with a diagnostic message, so OEMs
// see "this model won't work" before they ship a device.
//
// Validates input count + optional per-input shape. -1 in expected shape
// = wildcard (batch / dynamic spatial dims). Returns empty string on
// success; diagnostic on mismatch.
static std::string validateOrtContract(Ort::Session* s,
                                       size_t expectedInputs,
                                       const std::vector<std::vector<int64_t>>& inputShapes,
                                       const char* capability) {
    if (!s) return std::string(capability) + ": null session";
    auto toStr = [](const std::vector<int64_t>& v) {
        std::string r = "[";
        for (size_t i = 0; i < v.size(); ++i) { if (i) r += ","; r += std::to_string(v[i]); }
        return r + "]";
    };
    size_t nIn = s->GetInputCount();
    if (nIn != expectedInputs) {
        return std::string(capability) + ": expected "
               + std::to_string(expectedInputs) + " inputs, got "
               + std::to_string(nIn);
    }
    for (size_t i = 0; i < inputShapes.size() && i < nIn; ++i) {
        const auto& expect = inputShapes[i];
        if (expect.empty()) continue;  // skip check for this input
        auto actual = s->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        if (expect.size() != actual.size()) {
            return std::string(capability) + ": input[" + std::to_string(i)
                   + "] rank mismatch: expected " + toStr(expect)
                   + " got " + toStr(actual);
        }
        for (size_t d = 0; d < expect.size(); ++d) {
            if (expect[d] == -1) continue;  // wildcard on expected side
            // v0.6.9: treat dynamic actual dim (-1) as compatible with any
            // concrete expected dim. ONNX models with dynamic H/W (e.g. RT-DETR
            // `pixel_values [batch,3,height,width]` where height/width are
            // ONNX symbolic names) accept the concrete size at Run() time —
            // they're not incompatible with our preprocess pipeline that
            // resizes to kIn×kIn, they just don't pre-fix the spatial dims
            // in the graph.
            if (actual[d] == -1) continue;
            if (expect[d] != actual[d]) {
                return std::string(capability) + ": input[" + std::to_string(i)
                       + "] shape mismatch: expected " + toStr(expect)
                       + " got " + toStr(actual);
            }
        }
    }
    return "";
}

// v0.6 Phase A: estimate KV cache bytes per llama_context so the
// MemoryManager can account pool overhead in the resident budget.
//
// Formula: n_ctx × n_layer × 2 (K+V) × n_kv_head × head_dim × bytes_per_elem
//
// Elements are fp16 by default in llama.cpp (2 bytes). This is an
// estimate — real KV allocation may differ by a few % due to alignment
// and per-layer tensor padding. Close enough for budget planning.
static int64_t estimateKvBytesPerContext(llama_model* m, int32_t n_ctx) {
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

static int64_t fileSizeBytes(const std::string& path) {
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
static std::vector<std::string> readDetectClassLabels(const std::string& modelPath) {
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

// v0.6 Phase B/C helpers — companion-file loaders for capabilities that need
// OEM-baked metadata (tokenizers, phoneme maps). Kept deliberately minimal:
// hand-rolled parsing, no third-party JSON dep, same style as
// readDetectClassLabels above. If the sidecar is missing or malformed, the
// returned object is "empty" and callers surface CAPABILITY_UNAVAILABLE_NO_MODEL.

static bool fileExists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Read a whole file into a string; returns false if unreadable.
static bool readFileToString(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    out.assign((std::istreambuf_iterator<char>(f)),
               std::istreambuf_iterator<char>());
    return true;
}

// Skip JSON whitespace + commas. Returns new position.
static size_t skipJsonWs(const std::string& s, size_t p, size_t end) {
    while (p < end && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n'
                       || s[p] == '\r' || s[p] == ',')) ++p;
    return p;
}

// Parse a JSON string token starting at s[p] (which must be '"'). Returns the
// position after the closing quote and writes the decoded string to `out`.
// Supports \", \\, \n, \t, \/, \uXXXX (BMP only). Returns npos on malformed.
static size_t parseJsonString(const std::string& s, size_t p, size_t end, std::string& out) {
    if (p >= end || s[p] != '"') return std::string::npos;
    ++p;
    out.clear();
    while (p < end && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < end) {
            const char n = s[p + 1];
            if (n == '"')  { out.push_back('"');  p += 2; }
            else if (n == '\\') { out.push_back('\\'); p += 2; }
            else if (n == '/')  { out.push_back('/');  p += 2; }
            else if (n == 'n')  { out.push_back('\n'); p += 2; }
            else if (n == 't')  { out.push_back('\t'); p += 2; }
            else if (n == 'r')  { out.push_back('\r'); p += 2; }
            else if (n == 'b')  { out.push_back('\b'); p += 2; }
            else if (n == 'f')  { out.push_back('\f'); p += 2; }
            else if (n == 'u' && p + 5 < end) {
                unsigned cp = 0;
                for (int i = 0; i < 4; ++i) {
                    char c = s[p + 2 + i];
                    cp <<= 4;
                    if (c >= '0' && c <= '9')       cp |= (c - '0');
                    else if (c >= 'a' && c <= 'f')  cp |= (10 + c - 'a');
                    else if (c >= 'A' && c <= 'F')  cp |= (10 + c - 'A');
                    else return std::string::npos;
                }
                // UTF-8 encode (BMP only; surrogate pairs not supported).
                if (cp < 0x80) {
                    out.push_back((char)cp);
                } else if (cp < 0x800) {
                    out.push_back((char)(0xC0 | (cp >> 6)));
                    out.push_back((char)(0x80 | (cp & 0x3F)));
                } else {
                    out.push_back((char)(0xE0 | (cp >> 12)));
                    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back((char)(0x80 | (cp & 0x3F)));
                }
                p += 6;
            } else {
                out.push_back(n); p += 2;
            }
        } else {
            out.push_back(s[p++]);
        }
    }
    if (p >= end) return std::string::npos;
    return p + 1;  // past closing quote
}

// ---- Phoneme map for audio.synthesize ----
// Sidecar schema: `<piper-model-path>.phonemes.json`
//   { "version": 1,
//     "phoneme_ids": { "<ph>": <int>, ... },      // not used by the simple
//                                                 //   word-level path below
//     "grapheme_map": { "<word-or-grapheme>": [<id>, <id>, ...], ... } }
// Lookup is word-level (case-insensitive, whitespace-tokenized). Unknown words
// fall back to per-character lookup using single-char keys if present in the
// map; otherwise emit the "<unk>" entry if one is declared, else drop the
// word silently. This is intentionally simple — OEM-supplied maps for real
// languages can be produced from espeak-ng dumps.
struct PhonemeMap {
    std::unordered_map<std::string, std::vector<int64_t>> graphemeToIds;
    std::vector<int64_t> unkIds;
    bool empty() const { return graphemeToIds.empty() && unkIds.empty(); }
};

static bool parseIdArray(const std::string& s, size_t& p, size_t end,
                         std::vector<int64_t>& out) {
    p = skipJsonWs(s, p, end);
    if (p >= end || s[p] != '[') return false;
    ++p;
    while (true) {
        p = skipJsonWs(s, p, end);
        if (p >= end) return false;
        if (s[p] == ']') { ++p; return true; }
        const size_t numStart = p;
        if (s[p] == '-' || s[p] == '+') ++p;
        while (p < end && s[p] >= '0' && s[p] <= '9') ++p;
        if (numStart == p) return false;
        try {
            out.push_back((int64_t)std::stoll(s.substr(numStart, p - numStart)));
        } catch (...) { return false; }
    }
}

static bool loadPhonemeSidecar(const std::string& path, PhonemeMap& out) {
    std::string content;
    if (!readFileToString(path, content)) return false;
    const size_t end = content.size();

    const size_t gmapKey = content.find("\"grapheme_map\"");
    if (gmapKey == std::string::npos) {
        LOG(WARNING) << "oird: phoneme sidecar " << path << " missing \"grapheme_map\"";
        return false;
    }
    size_t p = content.find('{', gmapKey);
    if (p == std::string::npos) return false;
    ++p;
    while (true) {
        p = skipJsonWs(content, p, end);
        if (p >= end) return false;
        if (content[p] == '}') { ++p; break; }
        std::string key;
        size_t nextP = parseJsonString(content, p, end, key);
        if (nextP == std::string::npos) return false;
        p = skipJsonWs(content, nextP, end);
        if (p >= end || content[p] != ':') return false;
        ++p;
        std::vector<int64_t> ids;
        if (!parseIdArray(content, p, end, ids)) return false;
        // Case-fold ASCII for stable lookup.
        for (auto& c : key) c = (char)tolower((unsigned char)c);
        if (key == "<unk>") out.unkIds = std::move(ids);
        else out.graphemeToIds[std::move(key)] = std::move(ids);
    }
    LOG(INFO) << "oird: loaded phoneme map from " << path
              << " entries=" << out.graphemeToIds.size()
              << " has_unk=" << (!out.unkIds.empty() ? "yes" : "no");
    return !out.empty();
}

// Split `text` into lowercase whitespace-delimited tokens and look up each
// in the phoneme map. On miss, try per-character; on total miss, append
// <unk> if defined. Concatenates the ID sequences.
static std::vector<int64_t> graphemesToPhonemeIds(const std::string& text,
                                                   const PhonemeMap& pm) {
    std::vector<int64_t> out;
    std::string tok;
    auto flush = [&]() {
        if (tok.empty()) return;
        auto it = pm.graphemeToIds.find(tok);
        if (it != pm.graphemeToIds.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
        } else {
            // Per-character fallback.
            bool anyHit = false;
            for (char c : tok) {
                std::string one(1, c);
                auto cit = pm.graphemeToIds.find(one);
                if (cit != pm.graphemeToIds.end()) {
                    anyHit = true;
                    out.insert(out.end(), cit->second.begin(), cit->second.end());
                }
            }
            if (!anyHit && !pm.unkIds.empty()) {
                out.insert(out.end(), pm.unkIds.begin(), pm.unkIds.end());
            }
        }
        tok.clear();
    };
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { flush(); continue; }
        tok.push_back((char)tolower((unsigned char)c));
    }
    flush();
    return out;
}

// ---- HuggingFace-style tokenizer for text.classify / text.rerank ----
// Sidecar schema: `<model-path>.tokenizer.json` — the standard HF tokenizer
// export, of which we consume a narrow subset:
//   { "model": { "vocab": { "<token>": <id>, ... } },
//     "added_tokens": [{ "id": N, "content": "[CLS]" | "[SEP]" | "[UNK]" | "[PAD]" }, ...] }
// Runtime does whitespace pre-tokenization + longest-prefix match against the
// vocab. This is NOT a full WordPiece/BPE implementation — it works cleanly
// for single-word lookups and falls back to <UNK> for subword splits that
// need real WordPiece. OEMs shipping a classifier that requires full
// WordPiece must either bake a pre-tokenized model or link a fuller
// tokenizer in v0.6.1.
class HfTokenizer {
public:
    bool loadFromSidecar(const std::string& path);
    std::vector<int64_t> encode(const std::string& text) const;
    std::vector<int64_t> encodePair(const std::string& a, const std::string& b) const;
    std::vector<int64_t> typeIdsForPair(const std::string& a, const std::string& b) const;
    bool valid() const { return !mVocab.empty(); }

private:
    std::unordered_map<std::string, int64_t> mVocab;
    int64_t mCls  = -1;
    int64_t mSep  = -1;
    int64_t mUnk  = -1;
    int64_t mPad  = -1;

    std::vector<int64_t> tokenize(const std::string& text) const;
};

bool HfTokenizer::loadFromSidecar(const std::string& path) {
    std::string content;
    if (!readFileToString(path, content)) return false;
    const size_t end = content.size();

    // Find "vocab" key (accept nested under "model.vocab" or top-level).
    size_t vkey = content.find("\"vocab\"");
    if (vkey == std::string::npos) return false;
    size_t p = content.find('{', vkey);
    if (p == std::string::npos) return false;
    ++p;
    while (true) {
        p = skipJsonWs(content, p, end);
        if (p >= end) return false;
        if (content[p] == '}') { ++p; break; }
        std::string token;
        size_t nextP = parseJsonString(content, p, end, token);
        if (nextP == std::string::npos) return false;
        p = skipJsonWs(content, nextP, end);
        if (p >= end || content[p] != ':') return false;
        ++p;
        p = skipJsonWs(content, p, end);
        const size_t numStart = p;
        if (p < end && (content[p] == '-' || content[p] == '+')) ++p;
        while (p < end && content[p] >= '0' && content[p] <= '9') ++p;
        if (numStart == p) return false;
        try {
            mVocab[std::move(token)] = std::stoll(content.substr(numStart, p - numStart));
        } catch (...) { return false; }
    }

    // Wire special tokens if present.
    auto look = [&](const char* s) -> int64_t {
        auto it = mVocab.find(s);
        return it == mVocab.end() ? -1 : it->second;
    };
    mCls = look("[CLS]");   if (mCls < 0) mCls = look("<s>");
    mSep = look("[SEP]");   if (mSep < 0) mSep = look("</s>");
    mUnk = look("[UNK]");   if (mUnk < 0) mUnk = look("<unk>");
    mPad = look("[PAD]");   if (mPad < 0) mPad = look("<pad>");

    LOG(INFO) << "oird: loaded tokenizer from " << path
              << " vocab=" << mVocab.size()
              << " cls=" << mCls << " sep=" << mSep
              << " unk=" << mUnk << " pad=" << mPad;
    return !mVocab.empty();
}

// Whitespace + longest-prefix match. Case-folds ASCII lowercase before
// lookup — enough for uncased BERT-family tokenizers; OEM tokenizers that
// are case-sensitive or require proper WordPiece need v0.6.1 or a pre-
// tokenized input flow.
std::vector<int64_t> HfTokenizer::tokenize(const std::string& text) const {
    std::vector<int64_t> out;
    std::string word;
    auto flushWord = [&]() {
        if (word.empty()) return;
        // Try whole-word first.
        auto it = mVocab.find(word);
        if (it != mVocab.end()) { out.push_back(it->second); word.clear(); return; }
        // Longest-prefix match walk — approximates WordPiece for single-piece vocab.
        size_t pos = 0;
        bool first = true;
        while (pos < word.size()) {
            size_t bestLen = 0;
            int64_t bestId = -1;
            for (size_t len = word.size() - pos; len > 0; --len) {
                std::string piece = (first ? "" : "##") + word.substr(pos, len);
                auto pit = mVocab.find(piece);
                if (pit != mVocab.end()) { bestLen = len; bestId = pit->second; break; }
            }
            if (bestId < 0) { out.push_back(mUnk >= 0 ? mUnk : 0); word.clear(); return; }
            out.push_back(bestId);
            pos += bestLen;
            first = false;
        }
        word.clear();
    };
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r'
                || c == '.' || c == ',' || c == '!' || c == '?' || c == ';') {
            flushWord();
        } else {
            word.push_back((char)tolower((unsigned char)c));
        }
    }
    flushWord();
    return out;
}

std::vector<int64_t> HfTokenizer::encode(const std::string& text) const {
    std::vector<int64_t> ids;
    if (mCls >= 0) ids.push_back(mCls);
    auto mid = tokenize(text);
    ids.insert(ids.end(), mid.begin(), mid.end());
    if (mSep >= 0) ids.push_back(mSep);
    return ids;
}

std::vector<int64_t> HfTokenizer::encodePair(const std::string& a,
                                              const std::string& b) const {
    std::vector<int64_t> ids;
    if (mCls >= 0) ids.push_back(mCls);
    auto aIds = tokenize(a);
    ids.insert(ids.end(), aIds.begin(), aIds.end());
    if (mSep >= 0) ids.push_back(mSep);
    auto bIds = tokenize(b);
    ids.insert(ids.end(), bIds.begin(), bIds.end());
    if (mSep >= 0) ids.push_back(mSep);
    return ids;
}

std::vector<int64_t> HfTokenizer::typeIdsForPair(const std::string& a,
                                                  const std::string& b) const {
    // Mirror the layout encodePair produces: [CLS] a [SEP] b [SEP]
    std::vector<int64_t> types;
    if (mCls >= 0) types.push_back(0);
    auto aIds = tokenize(a);
    for (size_t i = 0; i < aIds.size(); ++i) types.push_back(0);
    if (mSep >= 0) types.push_back(0);
    auto bIds = tokenize(b);
    for (size_t i = 0; i < bIds.size(); ++i) types.push_back(1);
    if (mSep >= 0) types.push_back(1);
    return types;
}

static bool loadHfTokenizerSidecar(const std::string& path, HfTokenizer& out) {
    return out.loadFromSidecar(path);
}

// v0.6.2: same-model concurrency for whisper (audio.transcribe).
//
// Whisper's whisper_context holds per-inference state — decoder buffers,
// timestamp mappings, sampling state. Two whisper_full() calls on one
// context corrupt that state, so v0.6's single-ctx load was a latent
// race for concurrent transcribe submits.
//
// Mirror the llama ContextPool pattern: load N whisper_context* per
// model, lease one per submit, release on complete. The kernel's COW
// mmap sharing means N contexts cost only per-ctx state (a few MB
// each) rather than N × weight-size.
//
// Priority queueing is not wired — audio.transcribe is a single-
// priority capability today (no mixed-urgency submits on the same
// model). If that changes a ContextPool-style priority-sorted queue
// drops in here verbatim.
class WhisperPool {
public:
    WhisperPool(std::vector<whisper_context*> ctxs, int acquireTimeoutMs)
        : mSlots(), mAcquireTimeoutMs(acquireTimeoutMs) {
        // v0.7: empty-pool rejection — same rationale as ContextPool.
        if (ctxs.empty()) {
            LOG(FATAL) << "WhisperPool: refusing to construct with zero contexts "
                          "(would deadlock first acquire). Caller must validate "
                          "ctxs before construction.";
        }
        mSlots.reserve(ctxs.size());
        for (auto* c : ctxs) mSlots.push_back({c, false});
    }

    ~WhisperPool() {
        for (auto& s : mSlots) if (s.wctx) whisper_free(s.wctx);
    }

    WhisperPool(const WhisperPool&) = delete;
    WhisperPool& operator=(const WhisperPool&) = delete;

    size_t size() const { return mSlots.size(); }

    // v0.6.3: observability — pairs with getRuntimeStats() so `cmd oir
    // memory` can show whisper pool state the same way it shows llama pool
    // state. Both are best-effort snapshots under the pool's own mutex.
    int32_t busyCount() const {
        std::unique_lock<std::mutex> lk(mMu);
        int32_t n = 0;
        for (const auto& s : mSlots) if (s.inUse) ++n;
        return n;
    }

    int32_t waitingCount() const {
        std::lock_guard<std::mutex> lk(mMu);
        return mWaiters;
    }

    // Blocks up to acquireTimeoutMs waiting for a free slot.
    // Returns nullptr on timeout; sets slotIdxOut to the leased slot on
    // success so the caller can release() on completion.
    whisper_context* acquire(int& slotIdxOut, int64_t& waitMsOut) {
        using namespace std::chrono;
        const auto t0 = steady_clock::now();
        std::unique_lock<std::mutex> lk(mMu);
        const auto deadline = t0 + milliseconds(mAcquireTimeoutMs);
        while (true) {
            for (size_t i = 0; i < mSlots.size(); ++i) {
                if (!mSlots[i].inUse) {
                    mSlots[i].inUse = true;
                    slotIdxOut = static_cast<int>(i);
                    waitMsOut = duration_cast<milliseconds>(steady_clock::now() - t0).count();
                    return mSlots[i].wctx;
                }
            }
            ++mWaiters;
            auto waitStatus = mCv.wait_until(lk, deadline);
            --mWaiters;
            if (waitStatus == std::cv_status::timeout) {
                slotIdxOut = -1;
                waitMsOut = duration_cast<milliseconds>(steady_clock::now() - t0).count();
                return nullptr;
            }
        }
    }

    void release(int slotIdx) {
        std::unique_lock<std::mutex> lk(mMu);
        if (slotIdx < 0 || slotIdx >= static_cast<int>(mSlots.size())) return;
        mSlots[slotIdx].inUse = false;
        mCv.notify_one();
    }

private:
    struct Slot { whisper_context* wctx = nullptr; bool inUse = false; };
    std::vector<Slot>         mSlots;
    mutable std::mutex        mMu;   // mutable: busyCount() + waitingCount() are const observers
    std::condition_variable   mCv;
    int                       mAcquireTimeoutMs;
    int32_t                   mWaiters = 0;   // guarded by mMu — observed by waitingCount()
};

// RAII lease so release is guaranteed even on early-return / exception.
class WhisperLease {
public:
    WhisperLease(WhisperPool* pool, whisper_context* wctx, int slotIdx, int64_t waitMs)
        : mPool(pool), mWctx(wctx), mSlotIdx(slotIdx), mWaitMs(waitMs) {}
    ~WhisperLease() { if (mPool && mSlotIdx >= 0) mPool->release(mSlotIdx); }
    WhisperLease(const WhisperLease&) = delete;
    WhisperLease& operator=(const WhisperLease&) = delete;

    whisper_context* ctx() const { return mWctx; }
    int64_t waitMs()       const { return mWaitMs; }
    bool    valid()        const { return mWctx != nullptr; }

private:
    WhisperPool*     mPool;
    whisper_context* mWctx;
    int              mSlotIdx;
    int64_t          mWaitMs;
};

// v0.6.3: cross-backend scheduler.
//
// Before v0.6.3, same-model concurrency was solid (llama ContextPool,
// whisper WhisperPool, ORT inherent thread-safety) but priority only
// applied *within* one backend's pool — a high-priority audio submit
// couldn't jump ahead of queued text submits because they ran on
// different queues.
//
// This scheduler puts every submit onto one global priority-sorted
// ready queue. M worker threads (default = hardware_concurrency()
// clamped to [4, 16]) pull tasks in priority order and run them. Each
// task internally still leases from its backend's pool / calls its
// backend's Run — same-model parallelism and pool-level priority are
// preserved beneath.
//
// Priority convention matches ContextPool::PRIO_* (POSIX niceness —
// lower = higher priority). audio.* submits arrive at PRIO_AUDIO_REALTIME
// (0); text/vision at PRIO_NORMAL (10). Within one priority level,
// submits run FIFO (monotonic seq tiebreaker).
//
// Worker-count sizing: we want more workers than the sum of all pool
// sizes so a pool-bound task never blocks a worker that could
// otherwise pull a runnable task from the queue (head-of-line
// avoidance). Default 8 covers the typical mix of a 4-ctx llama
// pool + 2-ctx whisper + 1-ctx VLM without contention.
//
// Shutdown: scheduler destructor stops the workers; tasks still
// in-flight finish their current iteration then observe mStop.
// Queued-but-not-started tasks are dropped silently — callers see
// the request as cancelled (which matches oird-process-exit semantics).
class Scheduler {
public:
    using Task = std::function<void()>;

    struct Entry {
        int64_t seq;
        int32_t priority;
        Task    task;
        bool operator<(const Entry& o) const {
            // std::priority_queue is a max-heap. Invert comparison so
            // lower priority value is popped first; FIFO within same
            // priority via seq.
            if (priority != o.priority) return priority > o.priority;
            return seq > o.seq;
        }
    };

    explicit Scheduler(int numWorkers) : mStop(false) {
        if (numWorkers < 1) numWorkers = 1;
        mWorkers.reserve(numWorkers);
        for (int i = 0; i < numWorkers; ++i) {
            mWorkers.emplace_back([this] { workerLoop(); });
        }
    }

    ~Scheduler() {
        {
            std::lock_guard<std::mutex> lk(mMu);
            mStop = true;
        }
        mCv.notify_all();
        for (auto& w : mWorkers) if (w.joinable()) w.join();
    }

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void enqueue(int32_t priority, Task task) {
        {
            std::lock_guard<std::mutex> lk(mMu);
            mQueue.push({mNextSeq++, priority, std::move(task)});
        }
        mCv.notify_one();
    }

    int32_t queueDepth() const {
        std::lock_guard<std::mutex> lk(mMu);
        return static_cast<int32_t>(mQueue.size());
    }

    int32_t workerCount() const { return static_cast<int32_t>(mWorkers.size()); }

private:
    void workerLoop() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lk(mMu);
                mCv.wait(lk, [this] { return mStop || !mQueue.empty(); });
                if (mStop && mQueue.empty()) return;
                // const_cast because priority_queue::top returns const ref
                // but we want to move out of it. std::priority_queue
                // doesn't expose a non-const top; the standard workaround
                // is this const_cast-then-pop pattern.
                task = std::move(const_cast<Entry&>(mQueue.top()).task);
                mQueue.pop();
            }
            try {
                task();
            } catch (const std::exception& e) {
                LOG(ERROR) << "scheduler task threw: " << e.what();
            } catch (...) {
                LOG(ERROR) << "scheduler task threw unknown exception";
            }
        }
    }

    mutable std::mutex              mMu;
    std::condition_variable         mCv;
    std::priority_queue<Entry>      mQueue;
    std::vector<std::thread>        mWorkers;
    int64_t                         mNextSeq = 0;
    std::atomic<bool>               mStop;
};

class OirdService : public BnOirWorker {
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

    ::ndk::ScopedAStatus load(const std::string& modelPath, int64_t* _aidl_return) override {
        // v0.6.9: mLock is dropped around the slow ctor. See LoadInProgress
        // comment above claimLoadSlot for why.
        const std::string key = "llama-gen:" + modelPath;
        std::unique_lock<std::mutex> lk(mLock);

        // v0.4 S2: idempotent same-path detect — return existing handle if
        // same model already loaded as a generation model.
        for (auto& [h, m] : mModels) {
            if (m.path == modelPath
                    && !m.isEmbedding && !m.isWhisper && !m.isOnnx && !m.isVlm) {
                LOG(INFO) << "oird: model already loaded path=" << modelPath << " handle=" << h;
                *_aidl_return = h;
                return ::ndk::ScopedAStatus::ok();
            }
        }

        // v0.6.9: concurrent-load dedup. If another thread is already loading
        // the same key, wait here instead of racing a duplicate slow ctor.
        LoadClaim claim = claimLoadSlot(lk, key);
        if (claim.waited) {
            if (claim.waited->errCode != 0) {
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        claim.waited->errCode, claim.waited->errMsg.c_str());
            }
            *_aidl_return = claim.waited->handle;
            return ::ndk::ScopedAStatus::ok();
        }
        auto slot = claim.slot;

        // v0.4 S2-B: budget check + LRU eviction. Eviction skips in-flight + warmed.
        const int64_t newSize = fileSizeBytes(modelPath);
        const int64_t MB = 1024 * 1024;
        if (mBudgetMb > 0 && (mTotalBytes + newSize) > (int64_t)mBudgetMb * MB) {
            const int64_t budgetBytes = (int64_t)mBudgetMb * MB;
            const int64_t now = currentTimeMs();
            std::vector<std::pair<int64_t, int64_t>> candidates;
            for (const auto& [h, m] : mModels) {
                if (m.inFlightCount > 0) continue;
                if (m.warmUntilMs > now) continue;
                candidates.emplace_back(m.lastAccessMs, h);
            }
            std::sort(candidates.begin(), candidates.end());
            int64_t freed = 0;
            for (const auto& [_ts, h] : candidates) {
                if (mTotalBytes + newSize - freed <= budgetBytes) break;
                auto it = mModels.find(h);
                if (it == mModels.end()) continue;
                mLlamaPools.erase(h);
                {
                    auto oit = mOcrRec.find(h);
                    if (oit != mOcrRec.end()) {
                        delete oit->second.session;
                        mOcrRec.erase(oit);
                    }
                }
                if (it->second.ctx) llama_free(it->second.ctx);
                if (it->second.model) llama_model_free(it->second.model);
                mWhisperPools.erase(h);
                it->second.wctx = nullptr;
                delete it->second.ortSession;
                if (it->second.mtmdCtx) mtmd_free(it->second.mtmdCtx);
                freed += it->second.sizeBytes;
                LOG(INFO) << "oird: evicted handle=" << h
                          << " path=" << it->second.path
                          << " freed=" << (it->second.sizeBytes >> 20) << "MB";
                mModels.erase(it);
                mEvictionCount++;
            }
            mTotalBytes -= freed;
            if (mTotalBytes + newSize > budgetBytes) {
                LOG(ERROR) << "oird: budget " << mBudgetMb
                           << "MB exceeded; resident=" << (mTotalBytes >> 20)
                           << " + new=" << (newSize >> 20)
                           << "MB; nothing more evictable";
                const std::string msg = "budget exceeded; nothing evictable";
                publishLoadResult(lk, key, slot, 0, W_INSUFFICIENT_MEMORY, msg);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_INSUFFICIENT_MEMORY, msg.c_str());
            }
        }

        // Snapshot tunables under lock so slow-init uses a consistent set.
        const int32_t kCtxSize = mTextCompleteNCtx;
        const int32_t poolSize = std::max(1, mTextCompleteContextsPerModel);

        // Reserve the file-size share of mTotalBytes up front so a concurrent
        // load of a *different* path sees our pending bytes. KV-cache bytes
        // are added once known (after slow ctor).
        mTotalBytes += newSize;

        lk.unlock();

        LOG(INFO) << "oird: loading " << modelPath << " ctx=" << kCtxSize;

        // --- slow ctor, mLock NOT held ---
        constexpr int32_t kGpuLayers = 0;
        const int32_t totalCores = std::max(2, (int32_t)sysconf(_SC_NPROCESSORS_ONLN));
        int32_t threads = std::max(1, totalCores / poolSize);

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = kGpuLayers;
        mparams.use_mmap = true;
        mparams.use_mlock = false;

        llama_model* model = llama_model_load_from_file(modelPath.c_str(), mparams);
        if (!model) {
            LOG(ERROR) << "oird: llama_model_load_from_file failed for " << modelPath;
            lk.lock();
            mTotalBytes -= newSize;
            publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, "model load failed");
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, "model load failed");
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = kCtxSize;
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;

        std::vector<PooledContext> pooledCtxs;
        pooledCtxs.reserve(poolSize);
        for (int32_t i = 0; i < poolSize; ++i) {
            llama_context* c = llama_init_from_model(model, cparams);
            if (!c) {
                LOG(ERROR) << "oird: llama_init_from_model failed at pool slot " << i;
                for (auto& p : pooledCtxs) llama_free(p.ctx);
                llama_model_free(model);
                lk.lock();
                mTotalBytes -= newSize;
                publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, "context init failed");
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_MODEL_ERROR, "context init failed");
            }
            pooledCtxs.push_back({c, nullptr});
        }

        const int64_t kvPerCtx = estimateKvBytesPerContext(model, kCtxSize);
        const int64_t poolKvBytes = kvPerCtx * poolSize;

        // --- re-lock to insert ---
        lk.lock();

        const int64_t handle = mNextModelHandle++;
        const int64_t now = currentTimeMs();
        LoadedModel lm;
        lm.model = model;
        lm.vocab = llama_model_get_vocab(model);
        lm.ctx = nullptr;
        lm.context_size = kCtxSize;
        lm.n_threads = threads;
        lm.handle = handle;
        lm.path = modelPath;
        lm.sizeBytes = newSize + poolKvBytes;
        lm.loadTimestampMs = now;
        lm.lastAccessMs = now;
        lm.hasLlamaPool = true;
        // newSize was already added to mTotalBytes at reservation above; only
        // the KV bytes are new here.
        mTotalBytes += poolKvBytes;
        mModels[handle] = std::move(lm);
        mLlamaPools[handle] = std::make_unique<ContextPool>(std::move(pooledCtxs));

        publishLoadResult(lk, key, slot, handle, 0, "");

        *_aidl_return = handle;
        LOG(INFO) << "oird: model loaded handle=" << handle << " path=" << modelPath
                  << " size=" << (newSize >> 20) << "MB"
                  << " pool=" << poolSize << " ctx × " << (kvPerCtx >> 20) << "MB KV"
                  << " = " << (mModels[handle].sizeBytes >> 20) << "MB total"
                  << " resident=" << (mTotalBytes >> 20) << "/" << mBudgetMb << "MB"
                  << " total_models=" << mModels.size();
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.4 H4-A: load a model in embedding mode.
    // Separate code path from load() because llama_context_params differ materially
    // (embeddings=true, pooling_type=MEAN, smaller n_ctx is fine for sentence-level).
    ::ndk::ScopedAStatus loadEmbed(const std::string& modelPath, int64_t* _aidl_return) override {
        // v0.6.9: mLock shrunk around slow ctor (see load() / claimLoadSlot).
        const std::string key = "llama-emb:" + modelPath;
        std::unique_lock<std::mutex> lk(mLock);

        // Idempotent same-path detect — same file loaded as generation model is a distinct handle.
        for (auto& [h, m] : mModels) {
            if (m.path == modelPath && m.isEmbedding) {
                LOG(INFO) << "oird: embed model already loaded path=" << modelPath << " handle=" << h;
                *_aidl_return = h;
                return ::ndk::ScopedAStatus::ok();
            }
        }

        LoadClaim claim = claimLoadSlot(lk, key);
        if (claim.waited) {
            if (claim.waited->errCode != 0) {
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        claim.waited->errCode, claim.waited->errMsg.c_str());
            }
            *_aidl_return = claim.waited->handle;
            return ::ndk::ScopedAStatus::ok();
        }
        auto slot = claim.slot;

        const int64_t newSize = fileSizeBytes(modelPath);
        const int64_t MB = 1024 * 1024;
        if (mBudgetMb > 0 && (mTotalBytes + newSize) > (int64_t)mBudgetMb * MB) {
            const int64_t budgetBytes = (int64_t)mBudgetMb * MB;
            const int64_t now = currentTimeMs();
            std::vector<std::pair<int64_t, int64_t>> candidates;
            for (const auto& [h, m] : mModels) {
                if (m.inFlightCount > 0) continue;
                if (m.warmUntilMs > now) continue;
                candidates.emplace_back(m.lastAccessMs, h);
            }
            std::sort(candidates.begin(), candidates.end());
            int64_t freed = 0;
            for (const auto& [_ts, h] : candidates) {
                if (mTotalBytes + newSize - freed <= budgetBytes) break;
                auto it = mModels.find(h);
                if (it == mModels.end()) continue;
                mLlamaPools.erase(h);
                {
                    auto oit = mOcrRec.find(h);
                    if (oit != mOcrRec.end()) {
                        delete oit->second.session;
                        mOcrRec.erase(oit);
                    }
                }
                if (it->second.ctx) llama_free(it->second.ctx);
                if (it->second.model) llama_model_free(it->second.model);
                mWhisperPools.erase(h);
                it->second.wctx = nullptr;
                delete it->second.ortSession;
                if (it->second.mtmdCtx) mtmd_free(it->second.mtmdCtx);
                freed += it->second.sizeBytes;
                LOG(INFO) << "oird: evicted handle=" << h << " path=" << it->second.path
                          << " freed=" << (it->second.sizeBytes >> 20) << "MB";
                mModels.erase(it);
                mEvictionCount++;
            }
            mTotalBytes -= freed;
            if (mTotalBytes + newSize > budgetBytes) {
                const std::string msg = "budget exceeded; nothing evictable";
                publishLoadResult(lk, key, slot, 0, W_INSUFFICIENT_MEMORY, msg);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_INSUFFICIENT_MEMORY, msg.c_str());
            }
        }

        // Snapshot tunables + reserve budget under lock.
        const int32_t embNCtx = mTextEmbedNCtx;
        const int32_t poolSize = std::max(1, mTextEmbedContextsPerModel);
        mTotalBytes += newSize;

        lk.unlock();

        LOG(INFO) << "oird: loading (embed mode) " << modelPath;

        const int32_t totalCores = std::max(2, (int32_t)sysconf(_SC_NPROCESSORS_ONLN));
        int32_t threads = std::max(1, totalCores / poolSize);

        llama_model_params mparams = llama_model_default_params();
        mparams.use_mmap = true;
        llama_model* model = llama_model_load_from_file(modelPath.c_str(), mparams);
        if (!model) {
            LOG(ERROR) << "oird: llama_model_load_from_file failed for " << modelPath;
            lk.lock();
            mTotalBytes -= newSize;
            publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, "embed model load failed");
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, "embed model load failed");
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = embNCtx;
        cparams.embeddings = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;

        std::vector<PooledContext> pooledCtxs;
        pooledCtxs.reserve(poolSize);
        for (int32_t i = 0; i < poolSize; ++i) {
            llama_context* c = llama_init_from_model(model, cparams);
            if (!c) {
                LOG(ERROR) << "oird: llama_init_from_model (embed) failed at pool slot " << i;
                for (auto& p : pooledCtxs) llama_free(p.ctx);
                llama_model_free(model);
                lk.lock();
                mTotalBytes -= newSize;
                publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, "embed context init failed");
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_MODEL_ERROR, "embed context init failed");
            }
            pooledCtxs.push_back({c, nullptr});
        }

        const int64_t kvPerCtx = estimateKvBytesPerContext(model, embNCtx);
        const int64_t poolKvBytes = kvPerCtx * poolSize;
        const int32_t nEmbd = llama_n_embd(model);

        lk.lock();

        const int64_t handle = mNextModelHandle++;
        const int64_t now = currentTimeMs();
        LoadedModel lm;
        lm.model = model;
        lm.vocab = llama_model_get_vocab(model);
        lm.ctx = nullptr;
        lm.context_size = embNCtx;
        lm.n_threads = threads;
        lm.handle = handle;
        lm.path = modelPath;
        lm.sizeBytes = newSize + poolKvBytes;
        lm.loadTimestampMs = now;
        lm.lastAccessMs = now;
        lm.isEmbedding = true;
        lm.hasLlamaPool = true;
        mTotalBytes += poolKvBytes;
        mModels[handle] = std::move(lm);
        mLlamaPools[handle] = std::make_unique<ContextPool>(std::move(pooledCtxs));

        publishLoadResult(lk, key, slot, handle, 0, "");

        *_aidl_return = handle;
        LOG(INFO) << "oird: embed model loaded handle=" << handle << " path=" << modelPath
                  << " size=" << (newSize >> 20) << "MB"
                  << " n_embd=" << nEmbd
                  << " pool=" << poolSize << " ctx × " << (kvPerCtx >> 20) << "MB KV"
                  << " resident=" << (mTotalBytes >> 20) << "/" << mBudgetMb << "MB";
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.4 H4-A: run text through an embedding model; callback receives one pooled vector.
    // Synchronous in current v0.4 (no async thread) since embedding is ~1ms for MiniLM.
    ::ndk::ScopedAStatus submitEmbed(int64_t modelHandle,
                                     const std::string& text,
                                     const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                     int64_t* _aidl_return) override {
        LoadedModel* lmPtr = nullptr;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isEmbedding) {
                cb->onError(W_INVALID_INPUT, "handle not an embedding model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
            lmPtr = &it->second;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue inference body on the cross-backend scheduler.
        // lmPtr stays valid because inFlightCount++ above blocks LRU
        // eviction until releaseInflight runs in the lambda.
        mScheduler->enqueue(priorityForCapability("text.embed"),
            [this, modelHandle, text, cb, lmPtr]() {
                // v0.6.8: hold lease + inflight across inference ONLY; fire
                // terminal callback AFTER releasing both. Prior revisions
                // called cb->onVector / cb->onError while the ContextLease
                // was still in scope — a stalled binder callback (dead app,
                // binder thread contention) would pin the pool slot forever,
                // and any subsequent load() needing to evict would hang on
                // ContextPool::shutdown() waiting for the unreleasable lease.
                std::function<void()> terminal;
                {
                    // Tokenize — BOS=true, add_special=true (BERT-style)
                    std::vector<llama_token> tokens(text.size() + 8);
                    int n = llama_tokenize(lmPtr->vocab, text.c_str(), (int)text.size(),
                                           tokens.data(), (int)tokens.size(), true, true);
                    if (n < 0) {
                        tokens.resize(-n);
                        n = llama_tokenize(lmPtr->vocab, text.c_str(), (int)text.size(),
                                           tokens.data(), (int)tokens.size(), true, true);
                    }
                    if (n <= 0) {
                        terminal = [cb]() { cb->onError(W_INVALID_INPUT, "tokenize failed"); };
                        goto done;
                    }
                    tokens.resize(n);

                    ContextPool* pool = nullptr;
                    int priority = ContextPool::PRIO_NORMAL;
                    std::chrono::milliseconds timeout{10000};
                    {
                        std::lock_guard<std::mutex> lk(mLock);
                        auto pit = mLlamaPools.find(modelHandle);
                        if (pit == mLlamaPools.end()) {
                            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "embed model has no context pool"); };
                            goto done;
                        }
                        pool = pit->second.get();
                        priority = mTextEmbedPriority;
                        timeout = std::chrono::milliseconds(mTextEmbedAcquireTimeoutMs);
                    }
                    ContextLease lease(*pool, priority, timeout);
                    llama_context* ectx = lease.ctx();
                    if (!ectx) {
                        terminal = [cb]() { cb->onError(W_TIMEOUT, "embed pool acquire timed out"); };
                        goto done;
                    }

                    llama_memory_clear(llama_get_memory(ectx), true);
                    llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
                    for (int i = 0; i < (int)tokens.size(); ++i) {
                        batch.token[i] = tokens[i];
                        batch.pos[i] = i;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i] = (i == (int)tokens.size() - 1);
                    }
                    batch.n_tokens = (int)tokens.size();

                    int rc = llama_decode(ectx, batch);
                    if (rc != 0) {
                        llama_batch_free(batch);
                        terminal = [cb]() { cb->onError(W_MODEL_ERROR, "embed decode failed"); };
                        goto done;
                    }

                    const float* embeds = llama_get_embeddings_seq(ectx, 0);
                    if (!embeds) {
                        embeds = llama_get_embeddings_ith(ectx, (int)tokens.size() - 1);
                    }
                    if (!embeds) {
                        llama_batch_free(batch);
                        terminal = [cb]() { cb->onError(W_MODEL_ERROR, "no embeddings returned"); };
                        goto done;
                    }

                    int n_embd = llama_n_embd(lmPtr->model);
                    std::vector<float> vec(embeds, embeds + n_embd);

                    double sum2 = 0.0;
                    for (float v : vec) sum2 += (double)v * (double)v;
                    float norm = (float)std::sqrt(sum2);
                    if (norm > 1e-8f) {
                        for (float& v : vec) v /= norm;
                    }

                    llama_batch_free(batch);
                    terminal = [cb, vec = std::move(vec)]() { cb->onVector(vec); };

                    LOG(INFO) << "oird: embed handle=" << modelHandle << " text_tokens=" << tokens.size()
                              << " n_embd=" << n_embd;
                }
            done:
                // Lease dtor fires at the closing brace above (pool slot released).
                releaseInflight(modelHandle);
                if (terminal) terminal();
            });
        return ::ndk::ScopedAStatus::ok();
    }

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

    ::ndk::ScopedAStatus loadWhisper(const std::string& modelPath, int64_t* _aidl_return) override {
        // v0.6.9: mLock shrunk around slow ctor.
        const std::string key = "whisper:" + modelPath;
        std::unique_lock<std::mutex> lk(mLock);
        for (auto& [h, m] : mModels) {
            if (m.path == modelPath && m.isWhisper) {
                LOG(INFO) << "oird: whisper already loaded path=" << modelPath << " handle=" << h;
                *_aidl_return = h;
                return ::ndk::ScopedAStatus::ok();
            }
        }

        LoadClaim claim = claimLoadSlot(lk, key);
        if (claim.waited) {
            if (claim.waited->errCode != 0) {
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        claim.waited->errCode, claim.waited->errMsg.c_str());
            }
            *_aidl_return = claim.waited->handle;
            return ::ndk::ScopedAStatus::ok();
        }
        auto slot = claim.slot;

        const int64_t newSize = fileSizeBytes(modelPath);
        const int64_t MB = 1024 * 1024;
        if (mBudgetMb > 0 && (mTotalBytes + newSize) > (int64_t)mBudgetMb * MB) {
            const int64_t budgetBytes = (int64_t)mBudgetMb * MB;
            const int64_t now = currentTimeMs();
            std::vector<std::pair<int64_t, int64_t>> candidates;
            for (const auto& [h, m] : mModels) {
                if (m.inFlightCount > 0) continue;
                if (m.warmUntilMs > now) continue;
                candidates.emplace_back(m.lastAccessMs, h);
            }
            std::sort(candidates.begin(), candidates.end());
            int64_t freed = 0;
            for (const auto& [_ts, h] : candidates) {
                if (mTotalBytes + newSize - freed <= budgetBytes) break;
                auto it = mModels.find(h);
                if (it == mModels.end()) continue;
                mLlamaPools.erase(h);
                {
                    auto oit = mOcrRec.find(h);
                    if (oit != mOcrRec.end()) {
                        delete oit->second.session;
                        mOcrRec.erase(oit);
                    }
                }
                if (it->second.ctx) llama_free(it->second.ctx);
                if (it->second.model) llama_model_free(it->second.model);
                mWhisperPools.erase(h);
                it->second.wctx = nullptr;
                freed += it->second.sizeBytes;
                LOG(INFO) << "oird: evicted handle=" << h << " path=" << it->second.path;
                mModels.erase(it);
                mEvictionCount++;
            }
            mTotalBytes -= freed;
            if (mTotalBytes + newSize > budgetBytes) {
                const std::string msg = "budget exceeded; nothing evictable";
                publishLoadResult(lk, key, slot, 0, W_INSUFFICIENT_MEMORY, msg);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_INSUFFICIENT_MEMORY, msg.c_str());
            }
        }

        const int32_t poolSize = std::max(1, mAudioTranscribeContextsPerModel);
        const int32_t acquireTimeoutMs = mAudioTranscribeAcquireTimeoutMs;
        mTotalBytes += newSize;

        lk.unlock();

        // v0.6.2: build a whisper_context pool. Each context mmaps the same
        // weights file — kernel COW-shares pages so actual extra cost is
        // per-ctx state (tens of MB each for whisper-tiny).
        whisper_context_params wp = whisper_context_default_params();
        wp.use_gpu = false;

        std::vector<whisper_context*> ctxs;
        ctxs.reserve(poolSize);
        for (int i = 0; i < poolSize; ++i) {
            whisper_context* c = whisper_init_from_file_with_params(modelPath.c_str(), wp);
            if (!c) {
                LOG(ERROR) << "oird: whisper_init_from_file_with_params failed for "
                           << modelPath << " (pool ctx " << i << "/" << poolSize << ")";
                for (auto* prev : ctxs) whisper_free(prev);
                lk.lock();
                mTotalBytes -= newSize;
                publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, "whisper load failed");
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_MODEL_ERROR, "whisper load failed");
            }
            ctxs.push_back(c);
        }

        lk.lock();

        const int64_t handle = mNextModelHandle++;
        const int64_t now = currentTimeMs();
        LoadedModel lm;
        // Legacy lm.wctx kept pointing at the first ctx for any v0.5-era
        // non-pooled path; submitTranscribe leases from the pool, not this.
        lm.wctx = ctxs.front();
        lm.handle = handle;
        lm.path = modelPath;
        lm.sizeBytes = newSize;
        lm.loadTimestampMs = now;
        lm.lastAccessMs = now;
        lm.isWhisper = true;
        // newSize already reserved pre-slow-ctor.
        mModels[handle] = std::move(lm);

        mWhisperPools[handle] = std::make_unique<WhisperPool>(
                std::move(ctxs), acquireTimeoutMs);

        publishLoadResult(lk, key, slot, handle, 0, "");

        *_aidl_return = handle;
        LOG(INFO) << "oird: whisper model loaded handle=" << handle << " path=" << modelPath
                  << " size=" << (newSize >> 20) << "MB"
                  << " pool=" << poolSize
                  << " resident=" << (mTotalBytes >> 20) << "/" << mBudgetMb << "MB";
        return ::ndk::ScopedAStatus::ok();
    }

    ::ndk::ScopedAStatus submitTranscribe(int64_t modelHandle,
                                          const std::string& audioPath,
                                          const std::shared_ptr<IOirWorkerCallback>& cb,
                                          int64_t* _aidl_return) override {
        WhisperPool* pool = nullptr;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isWhisper) {
                cb->onError(W_INVALID_INPUT, "handle not a whisper model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            auto pit = mWhisperPools.find(modelHandle);
            if (pit == mWhisperPools.end() || !pit->second) {
                cb->onError(W_MODEL_ERROR, "whisper pool missing for handle");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            pool = pit->second.get();
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.3: enqueue the inference body onto the cross-backend
        // scheduler instead of running it inline — unblocks the binder
        // thread and lets audio-priority submits cut ahead of lower-
        // priority ones queued for other backends.
        mScheduler->enqueue(priorityForCapability("audio.transcribe"),
            [this, modelHandle, audioPath, cb, pool]() {
                // v0.6.8: release WhisperLease + inflight BEFORE firing the
                // terminal callback. cb->onToken mid-flight must stay inside
                // the lease scope (whisper's segment callback drives it);
                // only onComplete/onError are deferred.
                std::function<void()> terminal;
                int64_t waitMs = 0;
                size_t pcmSize = 0;
                int n_segments = 0;
                int64_t wall = 0;
                {
                    std::vector<float> pcm;
                    if (!readWav16(audioPath, pcm)) {
                        terminal = [cb, audioPath]() {
                            cb->onError(W_INVALID_INPUT, "WAV must be 16-bit mono 16 kHz: " + audioPath);
                        };
                        goto done;
                    }
                    pcmSize = pcm.size();

                    int slotIdx = -1;
                    whisper_context* wctx = pool->acquire(slotIdx, waitMs);
                    if (!wctx) {
                        const int timeoutMs = mAudioTranscribeAcquireTimeoutMs;
                        terminal = [cb, timeoutMs]() {
                            cb->onError(W_TIMEOUT,
                                "whisper pool acquire timed out after "
                                + std::to_string(timeoutMs) + "ms");
                        };
                        goto done;
                    }
                    WhisperLease lease(pool, wctx, slotIdx, waitMs);
                    if (waitMs > 100) {
                        LOG(INFO) << "oird: transcribe lease wait=" << waitMs << "ms";
                    }

                    std::string whisperLang;
                    {
                        std::lock_guard<std::mutex> lk(mLock);
                        whisperLang = mAudioTranscribeLanguage;
                    }
                    whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                    wp.print_progress = false;
                    wp.print_special = false;
                    wp.print_realtime = false;
                    wp.translate = false;
                    wp.language = whisperLang.c_str();
                    wp.single_segment = false;

                    struct CbCtx { std::shared_ptr<IOirWorkerCallback> cb; };
                    CbCtx ctx = { cb };
                    wp.new_segment_callback = [](whisper_context* c, whisper_state* /*unused*/, int n_new, void* user) {
                        auto* ctx = (CbCtx*)user;
                        int n_segments = whisper_full_n_segments(c);
                        for (int i = n_segments - n_new; i < n_segments; i++) {
                            const char* text = whisper_full_get_segment_text(c, i);
                            if (text && *text) ctx->cb->onToken(std::string(text), i);
                        }
                    };
                    wp.new_segment_callback_user_data = &ctx;

                    int64_t t0 = currentTimeMs();
                    int rc = whisper_full(wctx, wp, pcm.data(), (int)pcm.size());
                    int64_t t1 = currentTimeMs();
                    wall = t1 - t0;

                    if (rc != 0) {
                        terminal = [cb, rc]() {
                            cb->onError(W_MODEL_ERROR, "whisper_full failed rc=" + std::to_string(rc));
                        };
                        goto done;
                    }

                    n_segments = whisper_full_n_segments(wctx);
                    terminal = [cb, n_segments, wall]() {
                        cb->onComplete(n_segments, 0, wall);
                    };
                }
            done:
                // WhisperLease dtor fired at the scope close above.
                releaseInflight(modelHandle);
                if (terminal) terminal();

                LOG(INFO) << "oird: transcribe handle=" << modelHandle
                          << " audio_s=" << (pcmSize / 16000)
                          << " segments=" << n_segments
                          << " wait_ms=" << waitMs
                          << " wall_ms=" << wall;
            });
        return ::ndk::ScopedAStatus::ok();
    }

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
                                  int64_t* _aidl_return) override {
        // v0.6.9: mLock shrunk around slow ctor. This method had a latent
        // self-deadlock before: line ~2041 re-locked mLock inside the
        // detection branch while already holding the outer lock_guard.
        // std::mutex is non-recursive → futex_wait hang. The refactor
        // snapshots mVisionDetectInputSize under the initial lock and
        // releases before Ort::Session ctor.
        const std::string key = std::string(isDetection ? "onnx-det:" : "onnx-synth:") + modelPath;
        std::unique_lock<std::mutex> lk(mLock);
        for (auto& [h, m] : mModels) {
            if (m.path == modelPath && m.isOnnx && m.onnxIsDetection == isDetection) {
                LOG(INFO) << "oird: onnx model already loaded path=" << modelPath << " handle=" << h;
                *_aidl_return = h;
                return ::ndk::ScopedAStatus::ok();
            }
        }

        LoadClaim claim = claimLoadSlot(lk, key);
        if (claim.waited) {
            if (claim.waited->errCode != 0) {
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        claim.waited->errCode, claim.waited->errMsg.c_str());
            }
            *_aidl_return = claim.waited->handle;
            return ::ndk::ScopedAStatus::ok();
        }
        auto slot = claim.slot;

        const int64_t newSize = fileSizeBytes(modelPath);
        const int64_t MB = 1024 * 1024;
        if (mBudgetMb > 0 && (mTotalBytes + newSize) > (int64_t)mBudgetMb * MB) {
            const int64_t budgetBytes = (int64_t)mBudgetMb * MB;
            const int64_t now = currentTimeMs();
            std::vector<std::pair<int64_t, int64_t>> candidates;
            for (const auto& [h, m] : mModels) {
                if (m.inFlightCount > 0) continue;
                if (m.warmUntilMs > now) continue;
                candidates.emplace_back(m.lastAccessMs, h);
            }
            std::sort(candidates.begin(), candidates.end());
            int64_t freed = 0;
            for (const auto& [_ts, h] : candidates) {
                if (mTotalBytes + newSize - freed <= budgetBytes) break;
                auto it = mModels.find(h);
                if (it == mModels.end()) continue;
                mLlamaPools.erase(h);
                {
                    auto oit = mOcrRec.find(h);
                    if (oit != mOcrRec.end()) {
                        delete oit->second.session;
                        mOcrRec.erase(oit);
                    }
                }
                if (it->second.ctx) llama_free(it->second.ctx);
                if (it->second.model) llama_model_free(it->second.model);
                mWhisperPools.erase(h);
                it->second.wctx = nullptr;
                delete it->second.ortSession;
                if (it->second.mtmdCtx) mtmd_free(it->second.mtmdCtx);
                freed += it->second.sizeBytes;
                LOG(INFO) << "oird: evicted handle=" << h << " path=" << it->second.path;
                mModels.erase(it);
                mEvictionCount++;
            }
            mTotalBytes -= freed;
            if (mTotalBytes + newSize > budgetBytes) {
                const std::string msg = "budget exceeded; nothing evictable";
                publishLoadResult(lk, key, slot, 0, W_INSUFFICIENT_MEMORY, msg);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_INSUFFICIENT_MEMORY, msg.c_str());
            }
        }

        // Snapshot tunables under lock.
        const int32_t kIn = mVisionDetectInputSize;
        mTotalBytes += newSize;

        lk.unlock();

        // --- slow ctor: Ort env + Session, mLock NOT held ---
        ensureOrtEnv();
        Ort::SessionOptions so = makeOrtSessionOptions(isDetection);
        Ort::Session* session = nullptr;
        try {
            session = new Ort::Session(*mOrtEnv, modelPath.c_str(), so);
        } catch (const Ort::Exception& e) {
            LOG(ERROR) << "oird: Ort::Session failed for " << modelPath << ": " << e.what();
            const std::string msg = std::string("onnx load failed: ") + e.what();
            lk.lock();
            mTotalBytes -= newSize;
            publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, msg);
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, msg.c_str());
        }
        // Validate the ONNX shape contract — see above for per-kind rules.
        if (isDetection) {
            std::vector<std::vector<int64_t>> inShapes = {
                {-1, 3, kIn, kIn},  // batch wildcard, spatial fixed
            };
            std::string err = validateOrtContract(session, 1, inShapes, "vision.detect");
            if (!err.empty()) {
                LOG(ERROR) << "oird: " << err;
                delete session;
                lk.lock();
                mTotalBytes -= newSize;
                publishLoadResult(lk, key, slot, 0, W_MODEL_INCOMPATIBLE, err);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_MODEL_INCOMPATIBLE, err.c_str());
            }
        } else {
            // Infer "this is Piper" from `!isDetection` + input-name sniff.
            // Other single-Session ONNX (classify/rerank) skip Piper's
            // 3-input contract so they don't get wrongly rejected.
            bool looksLikePiper =
                session->GetInputCount() == 3
                && session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get() == std::string("input")
                && session->GetInputNameAllocated(1, Ort::AllocatorWithDefaultOptions()).get() == std::string("input_lengths")
                && session->GetInputNameAllocated(2, Ort::AllocatorWithDefaultOptions()).get() == std::string("scales");
            if (looksLikePiper) {
                std::vector<std::vector<int64_t>> inShapes = {
                    {1, -1},  // input: batch 1, phLen dynamic
                    {1},      // input_lengths
                    {3},      // scales
                };
                std::string err = validateOrtContract(session, 3, inShapes, "audio.synthesize");
                if (!err.empty()) {
                    LOG(ERROR) << "oird: " << err;
                    delete session;
                    lk.lock();
                    mTotalBytes -= newSize;
                    publishLoadResult(lk, key, slot, 0, W_MODEL_INCOMPATIBLE, err);
                    return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                            W_MODEL_INCOMPATIBLE, err.c_str());
                }
            }
        }

        // Per-handle detect labels are a best-effort sidecar read (file I/O
        // only, not held locked).
        std::vector<std::string> detectLabels;
        if (isDetection) {
            detectLabels = readDetectClassLabels(modelPath);
        }

        lk.lock();

        const int64_t handle = mNextModelHandle++;
        const int64_t now = currentTimeMs();
        LoadedModel lm;
        lm.ortSession = session;
        lm.handle = handle;
        lm.path = modelPath;
        lm.sizeBytes = newSize;
        lm.loadTimestampMs = now;
        lm.lastAccessMs = now;
        lm.isOnnx = true;
        lm.onnxIsDetection = isDetection;
        if (isDetection) {
            lm.detectClassLabels = std::move(detectLabels);
        }
        mModels[handle] = std::move(lm);

        publishLoadResult(lk, key, slot, handle, 0, "");

        *_aidl_return = handle;
        LOG(INFO) << "oird: onnx model loaded handle=" << handle << " path=" << modelPath
                  << " kind=" << (isDetection ? "detect" : "synth")
                  << " size=" << (newSize >> 20) << "MB"
                  << " resident=" << (mTotalBytes >> 20) << "/" << mBudgetMb << "MB";
        return ::ndk::ScopedAStatus::ok();
    }

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
                                          int64_t* _aidl_return) override {
        Ort::Session* session = nullptr;
        std::string sidecarPath;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isOnnx || it->second.onnxIsDetection) {
                cb->onError(W_INVALID_INPUT, "handle not an onnx synthesis model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            session     = it->second.ortSession;
            sidecarPath = it->second.path + ".phonemes.json";
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue on the cross-backend scheduler at audio-realtime
        // priority (matches audio.transcribe / audio.vad). Synthesize's
        // wall time is usually short, but routing through the scheduler
        // means a text.complete backlog doesn't delay TTS.
        mScheduler->enqueue(priorityForCapability("audio.synthesize"),
            [this, modelHandle, text, cb, session, sidecarPath]() {
                // v0.6.8: terminal cb (onComplete/onError) fires AFTER
                // releaseInflight. Streaming onChunk stays inline because
                // PCM is produced incrementally; if one onChunk binder call
                // stalls, the in-flight ref is the only resource held —
                // that's the minimum achievable for a streaming shape.
                std::function<void()> terminal;
                size_t nSamples = 0;
                size_t phCount = 0;
                int64_t totalMs = 0;
                {
                    if (text.empty()) {
                        terminal = [cb]() { cb->onError(W_INVALID_INPUT, "text is empty"); };
                        goto done;
                    }

                    PhonemeMap phonemes;
                    if (!loadPhonemeSidecar(sidecarPath, phonemes)) {
                        std::string msg = "no G2P sidecar at " + sidecarPath
                                + " — OEM must bake phonemes.json next to the Piper model";
                        terminal = [cb, msg]() {
                            cb->onError(W_CAPABILITY_UNAVAILABLE_NO_MODEL, msg.c_str());
                        };
                        goto done;
                    }

                    std::vector<int64_t> phIds = graphemesToPhonemeIds(text, phonemes);
                    if (phIds.empty()) {
                        terminal = [cb]() {
                            cb->onError(W_INVALID_INPUT, "G2P produced empty phoneme sequence");
                        };
                        goto done;
                    }
                    phCount = phIds.size();

                    Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(
                            OrtArenaAllocator, OrtMemTypeDefault);
                    const int64_t phLen = (int64_t)phIds.size();
                    std::array<int64_t, 2> inputShape{1, phLen};
                    std::array<int64_t, 1> lenShape{1};
                    std::array<int64_t, 1> scalesShape{3};
                    std::array<int64_t, 1> lenData{phLen};
                    // Piper scales[] is [noise_scale, length_scale, noise_w].
                    // First two are OEM-tunable via audio.synthesize.noise_scale
                    // and audio.synthesize.length_scale (applied through
                    // setCapabilityFloat). noise_w has no knob yet — keeping
                    // the standard Piper default of 0.8.
                    std::array<float, 3> scales{
                        mAudioSynthesizeNoiseScale,
                        mAudioSynthesizeLengthScale,
                        0.8f,
                    };

                    auto inputT = Ort::Value::CreateTensor<int64_t>(
                            meminfo, phIds.data(), phIds.size(),
                            inputShape.data(), inputShape.size());
                    auto lenT = Ort::Value::CreateTensor<int64_t>(
                            meminfo, lenData.data(), lenData.size(),
                            lenShape.data(), lenShape.size());
                    auto scalesT = Ort::Value::CreateTensor<float>(
                            meminfo, scales.data(), scales.size(),
                            scalesShape.data(), scalesShape.size());

                    const char* inputNames[]  = {"input", "input_lengths", "scales"};
                    const char* outputNames[] = {"output"};
                    Ort::Value inputs[] = {std::move(inputT), std::move(lenT), std::move(scalesT)};

                    std::vector<Ort::Value> outputs;
                    try {
                        outputs = session->Run(Ort::RunOptions{nullptr},
                                                inputNames, inputs, 3,
                                                outputNames, 1);
                    } catch (const std::exception& e) {
                        std::string msg = std::string("Piper ORT Run failed: ") + e.what();
                        terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                        goto done;
                    }
                    if (outputs.empty()) {
                        terminal = [cb]() { cb->onError(W_MODEL_ERROR, "Piper produced no output"); };
                        goto done;
                    }

                    const float* pcmF32 = outputs[0].GetTensorData<float>();
                    auto outInfo = outputs[0].GetTensorTypeAndShapeInfo();
                    nSamples = outInfo.GetElementCount();
                    // OEM-tunable via audio.synthesize.sample_rate_hz. Voices
                    // ship at fixed rates (Piper en-US/lessac is 22050) — this
                    // knob is for OEMs bundling non-default voices.
                    const int32_t kSampleRateHz = mAudioSynthesizeSampleRate;
                    constexpr int32_t kChannels = 1;
                    constexpr int32_t kEncodingPcmFloat = 4;

                    const size_t kChunkSamples = (size_t)kSampleRateHz / 10;
                    totalMs = (int64_t)((double)nSamples / (double)kSampleRateHz * 1000.0);
                    size_t emitted = 0;
                    while (emitted < nSamples) {
                        const size_t remaining = nSamples - emitted;
                        const size_t thisChunk = std::min(kChunkSamples, remaining);
                        const bool last = (emitted + thisChunk == nSamples);
                        std::vector<uint8_t> pcmBytes(thisChunk * sizeof(float));
                        std::memcpy(pcmBytes.data(), pcmF32 + emitted, pcmBytes.size());
                        cb->onChunk(pcmBytes, kSampleRateHz, kChannels, kEncodingPcmFloat, last);
                        emitted += thisChunk;
                    }
                    terminal = [cb, totalMs]() { cb->onComplete((int32_t)totalMs); };
                }
            done:
                releaseInflight(modelHandle);
                if (terminal) terminal();

                LOG(INFO) << "oird: submitSynthesize handle=" << modelHandle
                          << " text.len=" << text.size()
                          << " phonemes=" << phCount
                          << " samples=" << nSamples
                          << " ms=" << totalMs;
            });
        return ::ndk::ScopedAStatus::ok();
    }

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
                                        int64_t* _aidl_return) override {
        Ort::Session* session = nullptr;
        std::string sidecarPath;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isOnnx || it->second.onnxIsDetection) {
                cb->onError(W_INVALID_INPUT, "handle not an onnx text-classifier model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            session     = it->second.ortSession;
            sidecarPath = it->second.path + ".tokenizer.json";
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue on the cross-backend scheduler at text-normal
        // priority. ORT Run() is thread-safe so no pool needed — the
        // scheduler worker runs it directly. mLock never held across Run().
        mScheduler->enqueue(priorityForCapability("text.classify"),
            [this, modelHandle, text, cb, session, sidecarPath]() {
                // v0.6.8: terminal cb fires after releaseInflight.
                std::function<void()> terminal;
                size_t nTokens = 0;
                size_t nLabels = 0;
                {
                    if (text.empty()) {
                        terminal = [cb]() { cb->onError(W_INVALID_INPUT, "text is empty"); };
                        goto done;
                    }

                    HfTokenizer tok;
                    if (!loadHfTokenizerSidecar(sidecarPath, tok)) {
                        std::string msg = "no tokenizer sidecar at " + sidecarPath
                                + " — OEM must bake tokenizer.json next to the classifier";
                        terminal = [cb, msg]() {
                            cb->onError(W_CAPABILITY_UNAVAILABLE_NO_MODEL, msg.c_str());
                        };
                        goto done;
                    }

                    std::vector<int64_t> inputIds     = tok.encode(text);
                    std::vector<int64_t> attentionMask(inputIds.size(), 1);
                    const int64_t nTok = (int64_t)inputIds.size();
                    nTokens = inputIds.size();

                    Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(
                            OrtArenaAllocator, OrtMemTypeDefault);
                    std::array<int64_t, 2> shape{1, nTok};
                    auto idsT = Ort::Value::CreateTensor<int64_t>(
                            meminfo, inputIds.data(), inputIds.size(),
                            shape.data(), shape.size());
                    auto maskT = Ort::Value::CreateTensor<int64_t>(
                            meminfo, attentionMask.data(), attentionMask.size(),
                            shape.data(), shape.size());

                    const char* inputNames[]  = {"input_ids", "attention_mask"};
                    const char* outputNames[] = {"logits"};
                    Ort::Value inputs[] = {std::move(idsT), std::move(maskT)};

                    std::vector<Ort::Value> outputs;
                    try {
                        outputs = session->Run(Ort::RunOptions{nullptr},
                                                inputNames, inputs, 2,
                                                outputNames, 1);
                    } catch (const std::exception& e) {
                        std::string msg = std::string("classifier ORT Run failed: ") + e.what();
                        terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                        goto done;
                    }

                    const float* logits = outputs[0].GetTensorData<float>();
                    nLabels = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
                    std::vector<float> scores(nLabels);
                    float maxLogit = -std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < nLabels; ++i) maxLogit = std::max(maxLogit, logits[i]);
                    float sumExp = 0.f;
                    for (size_t i = 0; i < nLabels; ++i) { scores[i] = std::exp(logits[i] - maxLogit); sumExp += scores[i]; }
                    for (size_t i = 0; i < nLabels; ++i) scores[i] /= sumExp;

                    terminal = [cb, scores = std::move(scores)]() { cb->onVector(scores); };
                }
            done:
                releaseInflight(modelHandle);
                if (terminal) terminal();

                LOG(INFO) << "oird: submitClassify handle=" << modelHandle
                          << " text.len=" << text.size()
                          << " tokens=" << nTokens
                          << " labels=" << nLabels;
            });
        return ::ndk::ScopedAStatus::ok();
    }

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
                                       int64_t* _aidl_return) override {
        Ort::Session* session = nullptr;
        std::string sidecarPath;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isOnnx || it->second.onnxIsDetection) {
                cb->onError(W_INVALID_INPUT, "handle not an onnx reranker model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            session     = it->second.ortSession;
            sidecarPath = it->second.path + ".tokenizer.json";
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue. Rerank loops Run() once per candidate so it
        // can be chunky; scheduler dispatch keeps the binder thread free.
        mScheduler->enqueue(priorityForCapability("text.rerank"),
            [this, modelHandle, query, candidates, cb, session, sidecarPath]() {
                // v0.6.8: terminal cb deferred past releaseInflight.
                std::function<void()> terminal;
                {
                    if (query.empty() || candidates.empty()) {
                        terminal = [cb]() { cb->onError(W_INVALID_INPUT, "query/candidates empty"); };
                        goto done;
                    }

                    HfTokenizer tok;
                    if (!loadHfTokenizerSidecar(sidecarPath, tok)) {
                        std::string msg = "no tokenizer sidecar at " + sidecarPath
                                + " — OEM must bake tokenizer.json next to the reranker";
                        terminal = [cb, msg]() {
                            cb->onError(W_CAPABILITY_UNAVAILABLE_NO_MODEL, msg.c_str());
                        };
                        goto done;
                    }

                    std::vector<float> scores;
                    scores.reserve(candidates.size());
                    bool runFailed = false;
                    std::string runErr;

                    for (const auto& cand : candidates) {
                        std::vector<int64_t> inputIds     = tok.encodePair(query, cand);
                        std::vector<int64_t> typeIds      = tok.typeIdsForPair(query, cand);
                        std::vector<int64_t> attentionMask(inputIds.size(), 1);
                        const int64_t nTok = (int64_t)inputIds.size();

                        Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(
                                OrtArenaAllocator, OrtMemTypeDefault);
                        std::array<int64_t, 2> shape{1, nTok};
                        auto idsT  = Ort::Value::CreateTensor<int64_t>(
                                meminfo, inputIds.data(), inputIds.size(),
                                shape.data(), shape.size());
                        auto maskT = Ort::Value::CreateTensor<int64_t>(
                                meminfo, attentionMask.data(), attentionMask.size(),
                                shape.data(), shape.size());
                        auto typeT = Ort::Value::CreateTensor<int64_t>(
                                meminfo, typeIds.data(), typeIds.size(),
                                shape.data(), shape.size());

                        const char* inputNames[]  = {"input_ids", "attention_mask", "token_type_ids"};
                        const char* outputNames[] = {"logits"};
                        Ort::Value inputs[] = {std::move(idsT), std::move(maskT), std::move(typeT)};

                        std::vector<Ort::Value> outputs;
                        try {
                            outputs = session->Run(Ort::RunOptions{nullptr},
                                                    inputNames, inputs, 3,
                                                    outputNames, 1);
                        } catch (const std::exception& e) {
                            runErr = std::string("reranker ORT Run failed: ") + e.what();
                            runFailed = true;
                            break;
                        }
                        const float* logits = outputs[0].GetTensorData<float>();
                        scores.push_back(logits[0]);
                    }
                    if (runFailed) {
                        terminal = [cb, runErr]() { cb->onError(W_MODEL_ERROR, runErr.c_str()); };
                        goto done;
                    }

                    terminal = [cb, scores = std::move(scores)]() { cb->onVector(scores); };
                }
            done:
                releaseInflight(modelHandle);
                if (terminal) terminal();

                LOG(INFO) << "oird: submitRerank handle=" << modelHandle
                          << " query.len=" << query.size()
                          << " candidates=" << candidates.size();
            });
        return ::ndk::ScopedAStatus::ok();
    }

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
                                          int64_t* _aidl_return) override {
        if (maxTokens <= 0) maxTokens = 512;
        // Low temperature gives stable, close-to-greedy translations.
        return submit(modelHandle, prompt, maxTokens, /*temperature=*/0.2f, cb, _aidl_return);
    }

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
                                    int64_t* _aidl_return) override {
        Ort::Session* detSession = nullptr;
        std::string basePath;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isOnnx || !it->second.onnxIsDetection) {
                cb->onError(W_INVALID_INPUT, "handle not an onnx OCR-detection model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            detSession = it->second.ortSession;
            basePath   = it->second.path;
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue on scheduler.
        mScheduler->enqueue(priorityForCapability("vision.ocr"),
            [this, modelHandle, imagePath, cb, detSession, basePath]() {
        // v0.6.8: terminal cb deferred past releaseInflight.
        std::function<void()> terminal;
        size_t candCount = 0, keptCount = 0;
        int imgW = 0, imgH = 0;
        {
        // Require both sidecars up-front; partial OCR isn't meaningful.
        const std::string recPath   = basePath + ".rec.onnx";
        const std::string vocabPath = basePath + ".rec.vocab.txt";
        if (!fileExists(recPath) || !fileExists(vocabPath)) {
            std::string msg = "OCR requires det+rec+vocab triplet; missing "
                              + (fileExists(recPath) ? vocabPath : recPath);
            terminal = [cb, msg]() {
                cb->onError(W_CAPABILITY_UNAVAILABLE_NO_MODEL, msg.c_str());
            };
            goto done;
        }

        // Lazy-load rec session + vocab (first submitOcr for this handle).
        // Cached in mOcrRec; released when the det model evicts.
        Ort::Session* recSession = nullptr;
        std::vector<std::string> vocab;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto oit = mOcrRec.find(modelHandle);
            if (oit == mOcrRec.end()) {
                // Load rec ONNX session. Reuse the static ORT env from mOrtEnv.
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(2);
                opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                Ort::Session* rs = nullptr;
                try {
                    rs = new Ort::Session(*mOrtEnv, recPath.c_str(), opts);
                } catch (const std::exception& e) {
                    std::string msg = std::string("rec ONNX load failed: ") + e.what();
                    terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                    goto done;
                }
                // Load vocab file (one UTF-8 token per line; index 0 = CTC blank).
                std::ifstream vf(vocabPath);
                std::vector<std::string> v;
                std::string line;
                while (std::getline(vf, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    v.push_back(std::move(line));
                }
                if (v.size() < 2) {
                    delete rs;
                    terminal = [cb]() {
                        cb->onError(W_MODEL_ERROR, "rec vocab < 2 entries (need blank + ≥1 char)");
                    };
                    goto done;
                }
                mOcrRec[modelHandle] = OcrRec{rs, std::move(v)};
                oit = mOcrRec.find(modelHandle);
                LOG(INFO) << "oird: loaded OCR rec handle=" << modelHandle
                          << " rec=" << recPath << " vocab_sz=" << oit->second.vocab.size();
            }
            recSession = oit->second.session;
            vocab      = oit->second.vocab;  // copy under lock, immutable after load
        }

        // Decode source image.
        RgbImage img;
        bool ok = false;
        std::string ext4 = imagePath.size() >= 4 ? imagePath.substr(imagePath.size() - 4) : "";
        std::string ext5 = imagePath.size() >= 5 ? imagePath.substr(imagePath.size() - 5) : "";
        for (auto& c : ext4) c = (char)tolower((unsigned char)c);
        for (auto& c : ext5) c = (char)tolower((unsigned char)c);
        if (ext4 == ".jpg" || ext5 == ".jpeg") ok = decodeJpeg(imagePath, img, mImageMaxPixels);
        else if (ext4 == ".png") ok = decodePng(imagePath, img, mImageMaxPixels);
        if (!ok) {
            std::string msg = "image decode failed (need .jpg/.jpeg/.png): " + imagePath;
            terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
            goto done;
        }
        imgW = img.w;
        imgH = img.h;

        // --- Stage 1: Run detection via the loaded det model ---
        // Expected output shape: either [1, 5, N] (YOLO with 1 class) or
        // [1, N, 6] (DETR). The detection family knob (vision.detect.family)
        // controls parsing — OCR uses the same knob, defaulted per
        // oir_config.xml. Low-confidence / overlapping regions NMS-pruned
        // using the existing detect thresholds.
        int32_t detectInputSize;
        float   scoreThresh;
        float   iouThresh;
        std::string detectFamily;
        {
            std::lock_guard<std::mutex> lk(mLock);
            detectInputSize = mVisionDetectInputSize;
            scoreThresh     = mDetectScoreThresh;
            iouThresh       = mDetectIouThresh;
            detectFamily    = mVisionDetectFamily;
        }

        const int kIn = detectInputSize;
        const float detScale = std::min((float)kIn / img.w, (float)kIn / img.h);
        const int newW = (int)std::round(img.w * detScale);
        const int newH = (int)std::round(img.h * detScale);
        const int padX = (kIn - newW) / 2;
        const int padY = (kIn - newH) / 2;
        std::vector<float> detInput(3 * kIn * kIn, 114.0f / 255.0f);
        for (int y = 0; y < newH; ++y) {
            float fy = (y + 0.5f) / detScale - 0.5f;
            int y0 = std::max(0, (int)std::floor(fy));
            int y1 = std::min(img.h - 1, y0 + 1);
            float wy = fy - y0;
            for (int x = 0; x < newW; ++x) {
                float fx = (x + 0.5f) / detScale - 0.5f;
                int x0 = std::max(0, (int)std::floor(fx));
                int x1 = std::min(img.w - 1, x0 + 1);
                float wx = fx - x0;
                for (int c = 0; c < 3; ++c) {
                    float p00 = img.px[(y0 * img.w + x0) * 3 + c];
                    float p01 = img.px[(y0 * img.w + x1) * 3 + c];
                    float p10 = img.px[(y1 * img.w + x0) * 3 + c];
                    float p11 = img.px[(y1 * img.w + x1) * 3 + c];
                    float a = p00 * (1 - wx) + p01 * wx;
                    float b = p10 * (1 - wx) + p11 * wx;
                    float v = (a * (1 - wy) + b * wy) / 255.0f;
                    detInput[c * kIn * kIn + (y + padY) * kIn + (x + padX)] = v;
                }
            }
        }

        std::array<int64_t, 4> detShape{1, 3, kIn, kIn};
        Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
        auto detT = Ort::Value::CreateTensor<float>(
                meminfo, detInput.data(), detInput.size(),
                detShape.data(), detShape.size());

        // Output-node names vary per exporter; probe common alternatives.
        Ort::AllocatorWithDefaultOptions alloc;
        auto detInputName  = detSession->GetInputNameAllocated(0, alloc);
        auto detOutputName = detSession->GetOutputNameAllocated(0, alloc);
        const char* detInNames[]  = {detInputName.get()};
        const char* detOutNames[] = {detOutputName.get()};
        Ort::Value detInputs[] = {std::move(detT)};

        std::vector<Ort::Value> detOutputs;
        try {
            detOutputs = detSession->Run(Ort::RunOptions{nullptr},
                                          detInNames, detInputs, 1,
                                          detOutNames, 1);
        } catch (const std::exception& e) {
            std::string msg = std::string("OCR det ORT Run failed: ") + e.what();
            terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
            goto done;
        }

        // Parse detections as axis-aligned text boxes. Output layout:
        //   YOLO-1-class:  [1, 5, N]  (cx, cy, w, h, score)  per-column N-wise
        //   DETR-style:    [1, N, 6]  (x1, y1, x2, y2, score, class)
        const bool useDetr = (detectFamily == "rtdetr" || detectFamily == "detr");
        const float* detData = detOutputs[0].GetTensorData<float>();
        auto detInfo = detOutputs[0].GetTensorTypeAndShapeInfo();
        auto detShapeOut = detInfo.GetShape();
        struct TextBox { float x1, y1, x2, y2, score; };
        std::vector<TextBox> cands;
        if (useDetr && detShapeOut.size() == 3) {
            const int64_t N = detShapeOut[1];
            const int64_t F = detShapeOut[2];
            for (int64_t i = 0; i < N; ++i) {
                float score = detData[i * F + 4];
                if (score < scoreThresh) continue;
                cands.push_back({detData[i*F+0], detData[i*F+1],
                                 detData[i*F+2], detData[i*F+3], score});
            }
        } else if (detShapeOut.size() == 3) {
            const int64_t C = detShapeOut[1];   // 5 for 1-class YOLO text detector
            const int64_t N = detShapeOut[2];
            for (int64_t i = 0; i < N; ++i) {
                float score = (C >= 5) ? detData[4 * N + i] : 0.f;
                if (score < scoreThresh) continue;
                float cx = detData[0 * N + i];
                float cy = detData[1 * N + i];
                float w  = detData[2 * N + i];
                float h  = detData[3 * N + i];
                cands.push_back({cx - w/2, cy - h/2, cx + w/2, cy + h/2, score});
            }
        } else {
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "unsupported OCR det output shape"); };
            goto done;
        }
        candCount = cands.size();

        // Inverse letterbox: map back to original image coords, clip.
        for (auto& c : cands) {
            c.x1 = std::max(0.f, (c.x1 - padX) / detScale);
            c.y1 = std::max(0.f, (c.y1 - padY) / detScale);
            c.x2 = std::min((float)img.w, (c.x2 - padX) / detScale);
            c.y2 = std::min((float)img.h, (c.y2 - padY) / detScale);
        }

        // Simple IoU-NMS — reuses the same thresholds as vision.detect.
        std::sort(cands.begin(), cands.end(),
                  [](const TextBox& a, const TextBox& b) { return a.score > b.score; });
        std::vector<bool> keep(cands.size(), true);
        for (size_t i = 0; i < cands.size(); ++i) {
            if (!keep[i]) continue;
            for (size_t j = i + 1; j < cands.size(); ++j) {
                if (!keep[j]) continue;
                float xx1 = std::max(cands[i].x1, cands[j].x1);
                float yy1 = std::max(cands[i].y1, cands[j].y1);
                float xx2 = std::min(cands[i].x2, cands[j].x2);
                float yy2 = std::min(cands[i].y2, cands[j].y2);
                float inter = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);
                float a = std::max(0.f, cands[i].x2 - cands[i].x1)
                        * std::max(0.f, cands[i].y2 - cands[i].y1);
                float b = std::max(0.f, cands[j].x2 - cands[j].x1)
                        * std::max(0.f, cands[j].y2 - cands[j].y1);
                if (inter / (a + b - inter + 1e-6f) > iouThresh) keep[j] = false;
            }
        }

        // --- Stage 2: For each kept box, crop → rec → CTC decode ---
        // PaddleOCR-compatible rec input: 3-channel float32 [1, 3, 48, W]
        // where W scales with aspect ratio (quantized to a multiple of 8).
        // Pixel normalization: (x/255 - 0.5) / 0.5 per channel.
        constexpr int kRecH = 48;
        constexpr int kRecWMin = 48;
        constexpr int kRecWMax = 640;
        std::vector<int> xs, ys, widths, heights, labelsPerBox;
        std::vector<std::string> labelsFlat;
        std::vector<float> scoresFlat;

        for (size_t i = 0; i < cands.size(); ++i) {
            if (!keep[i]) continue;
            const auto& c = cands[i];
            int cx1 = (int)std::max(0.f, std::floor(c.x1));
            int cy1 = (int)std::max(0.f, std::floor(c.y1));
            int cx2 = (int)std::min((float)img.w, std::ceil(c.x2));
            int cy2 = (int)std::min((float)img.h, std::ceil(c.y2));
            int cw = cx2 - cx1;
            int ch = cy2 - cy1;
            if (cw < 4 || ch < 4) continue;

            // Resize crop to (kRecH, recW) preserving aspect.
            int recW = std::max(kRecWMin,
                    std::min(kRecWMax, (int)std::round((float)cw * kRecH / ch)));
            recW = ((recW + 7) / 8) * 8;  // multiple of 8 for kernel alignment

            std::vector<float> recInput(3 * kRecH * recW, 0.f);
            const float sx = (float)cw / recW;
            const float sy = (float)ch / kRecH;
            for (int y = 0; y < kRecH; ++y) {
                float fy = (y + 0.5f) * sy - 0.5f;
                int y0 = std::max(0, (int)std::floor(fy));
                int y1 = std::min(ch - 1, y0 + 1);
                float wy = fy - y0;
                for (int x = 0; x < recW; ++x) {
                    float fx = (x + 0.5f) * sx - 0.5f;
                    int x0 = std::max(0, (int)std::floor(fx));
                    int x1 = std::min(cw - 1, x0 + 1);
                    float wx = fx - x0;
                    for (int chn = 0; chn < 3; ++chn) {
                        int srcYX0 = (cy1 + y0) * img.w + (cx1 + x0);
                        int srcYX1 = (cy1 + y0) * img.w + (cx1 + x1);
                        int srcY1X0 = (cy1 + y1) * img.w + (cx1 + x0);
                        int srcY1X1 = (cy1 + y1) * img.w + (cx1 + x1);
                        float p00 = img.px[srcYX0  * 3 + chn];
                        float p01 = img.px[srcYX1  * 3 + chn];
                        float p10 = img.px[srcY1X0 * 3 + chn];
                        float p11 = img.px[srcY1X1 * 3 + chn];
                        float a = p00 * (1 - wx) + p01 * wx;
                        float b = p10 * (1 - wx) + p11 * wx;
                        float v = (a * (1 - wy) + b * wy) / 255.0f;
                        // PaddleOCR normalization: (x - 0.5) / 0.5
                        recInput[chn * kRecH * recW + y * recW + x] = (v - 0.5f) / 0.5f;
                    }
                }
            }

            // Rec ORT run.
            std::array<int64_t, 4> recShape{1, 3, kRecH, recW};
            auto recT = Ort::Value::CreateTensor<float>(
                    meminfo, recInput.data(), recInput.size(),
                    recShape.data(), recShape.size());
            auto recInName  = recSession->GetInputNameAllocated(0, alloc);
            auto recOutName = recSession->GetOutputNameAllocated(0, alloc);
            const char* recInNames[]  = {recInName.get()};
            const char* recOutNames[] = {recOutName.get()};
            Ort::Value recInputs[] = {std::move(recT)};

            std::vector<Ort::Value> recOutputs;
            try {
                recOutputs = recSession->Run(Ort::RunOptions{nullptr},
                                              recInNames, recInputs, 1,
                                              recOutNames, 1);
            } catch (const std::exception&) {
                continue;  // skip this region; don't fail the whole request
            }

            // Rec output: [1, T, C]  (logits or softmax-probs, per-timestep classes)
            const float* logits = recOutputs[0].GetTensorData<float>();
            auto recInfo = recOutputs[0].GetTensorTypeAndShapeInfo();
            auto recOutShape = recInfo.GetShape();
            if (recOutShape.size() != 3) continue;
            const int64_t T = recOutShape[1];
            const int64_t C = recOutShape[2];

            // CTC greedy decode: argmax per timestep, collapse repeats, drop blank(0).
            std::string text;
            int64_t prevId = -1;
            float scoreSum = 0.f;
            int   scoreN   = 0;
            for (int64_t t = 0; t < T; ++t) {
                int64_t bestId = 0;
                float bestVal = logits[t * C + 0];
                for (int64_t k = 1; k < C; ++k) {
                    float v = logits[t * C + k];
                    if (v > bestVal) { bestVal = v; bestId = k; }
                }
                if (bestId != 0 && bestId != prevId
                        && (size_t)bestId < vocab.size()) {
                    text += vocab[bestId];
                    scoreSum += bestVal;
                    scoreN++;
                }
                prevId = bestId;
            }
            if (text.empty()) continue;  // no characters decoded

            // Emit this region.
            xs.push_back((int)std::round(c.x1));
            ys.push_back((int)std::round(c.y1));
            widths.push_back((int)std::round(c.x2 - c.x1));
            heights.push_back((int)std::round(c.y2 - c.y1));
            labelsPerBox.push_back(1);
            labelsFlat.push_back(std::move(text));
            scoresFlat.push_back(scoreN > 0 ? scoreSum / scoreN : c.score);
        }

        keptCount = labelsFlat.size();
        terminal = [cb,
                    xs = std::move(xs),
                    ys = std::move(ys),
                    widths = std::move(widths),
                    heights = std::move(heights),
                    labelsPerBox = std::move(labelsPerBox),
                    labelsFlat = std::move(labelsFlat),
                    scoresFlat = std::move(scoresFlat)]() {
            cb->onBoundingBoxes(xs, ys, widths, heights,
                                labelsPerBox, labelsFlat, scoresFlat);
        };
        }
    done:
        releaseInflight(modelHandle);
        if (terminal) terminal();

        LOG(INFO) << "oird: submitOcr handle=" << modelHandle
                  << " img=" << imgW << "x" << imgH
                  << " candidates=" << candCount
                  << " kept=" << keptCount;
            });  // v0.6.4: close mScheduler->enqueue lambda
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.4 H3: vision.detect real impl.
    // Decode → letterbox 640×640 → YOLOv8 inference → parse (1, 84, 8400)
    // → NMS → map coords back to source image → bboxes callback.
    // Default class labels = COCO-80 inline. Sidecar <model>.classes.json
    // support deferred to v0.5 (trivial std::ifstream + simple tokenizer).
    ::ndk::ScopedAStatus submitDetect(int64_t modelHandle,
                                      const std::string& imagePath,
                                      const std::shared_ptr<IOirWorkerBboxCallback>& cb,
                                      int64_t* _aidl_return) override {
        Ort::Session* session = nullptr;
        std::vector<std::string> classLabels; // v0.5 V8: per-model sidecar, empty → COCO-80 fallback
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isOnnx || !it->second.onnxIsDetection) {
                cb->onError(W_INVALID_INPUT, "handle not an onnx detection model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            session = it->second.ortSession;
            classLabels = it->second.detectClassLabels; // copy under lock (labels are immutable post-load)
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue on scheduler. classLabels was already copied
        // under lock above; capture by move so the lambda owns it.
        mScheduler->enqueue(priorityForCapability("vision.detect"),
            [this, modelHandle, imagePath, cb, session,
             classLabels = std::move(classLabels)]() mutable {
        // v0.6.8: terminal cb deferred past releaseInflight.
        std::function<void()> terminal;
        int imgW = 0, imgH = 0;
        size_t candCount = 0, keptCount = 0;
        int64_t t0 = 0, t1 = 0;
        {
        // v0.5 V7: snapshot detect-tuning knobs under lock.
        int32_t detectInputSize;
        std::string detectFamily;
        {
            std::lock_guard<std::mutex> lk(mLock);
            detectInputSize = mVisionDetectInputSize;
            detectFamily    = mVisionDetectFamily;
        }

        // Decode
        RgbImage img;
        bool ok = false;
        std::string ext4 = imagePath.size() >= 4 ? imagePath.substr(imagePath.size() - 4) : "";
        std::string ext5 = imagePath.size() >= 5 ? imagePath.substr(imagePath.size() - 5) : "";
        for (auto& c : ext4) c = (char)tolower((unsigned char)c);
        for (auto& c : ext5) c = (char)tolower((unsigned char)c);
        if (ext4 == ".jpg" || ext5 == ".jpeg") ok = decodeJpeg(imagePath, img, mImageMaxPixels);
        else if (ext4 == ".png") ok = decodePng(imagePath, img, mImageMaxPixels);
        if (!ok) {
            std::string msg = "image decode failed (need .jpg/.jpeg/.png): " + imagePath;
            terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
            goto done;
        }
        imgW = img.w; imgH = img.h;

        // Letterbox to kIn×kIn (OEM-tunable, default 640 for YOLOv8n),
        // preserving aspect ratio, padding gray (114/255).
        // Record scale + pad so bboxes can be mapped back to source coords.
        const int kIn = detectInputSize;
        const float scale = std::min((float)kIn / img.w, (float)kIn / img.h);
        const int newW = (int)std::round(img.w * scale);
        const int newH = (int)std::round(img.h * scale);
        const int padX = (kIn - newW) / 2;
        const int padY = (kIn - newH) / 2;
        std::vector<float> input(3 * kIn * kIn, 114.0f / 255.0f);  // gray fill

        for (int y = 0; y < newH; ++y) {
            float fy = (y + 0.5f) / scale - 0.5f;
            int y0 = std::max(0, (int)std::floor(fy));
            int y1 = std::min(img.h - 1, y0 + 1);
            float wy = fy - y0;
            for (int x = 0; x < newW; ++x) {
                float fx = (x + 0.5f) / scale - 0.5f;
                int x0 = std::max(0, (int)std::floor(fx));
                int x1 = std::min(img.w - 1, x0 + 1);
                float wx = fx - x0;
                for (int c = 0; c < 3; ++c) {
                    float p00 = img.px[(y0 * img.w + x0) * 3 + c];
                    float p01 = img.px[(y0 * img.w + x1) * 3 + c];
                    float p10 = img.px[(y1 * img.w + x0) * 3 + c];
                    float p11 = img.px[(y1 * img.w + x1) * 3 + c];
                    float a = p00 * (1 - wx) + p01 * wx;
                    float b = p10 * (1 - wx) + p11 * wx;
                    float v = (a * (1 - wy) + b * wy) / 255.0f;
                    int outY = y + padY;
                    int outX = x + padX;
                    input[c * kIn * kIn + outY * kIn + outX] = v;
                }
            }
        }

        // ORT Run
        try {
            Ort::AllocatorWithDefaultOptions alloc;
            if (session->GetInputCount() < 1 || session->GetOutputCount() < 1) {
                terminal = [cb]() { cb->onError(W_MODEL_ERROR, "ORT session has 0 inputs or outputs"); };
                goto done;
            }
            // v0.5: pull all output names so the family-dispatched parsers can
            // pick the tensors they need by name (RT-DETR returns 3, YOLOv8 returns 1).
            const size_t nIn  = session->GetInputCount();
            const size_t nOut = session->GetOutputCount();
            std::vector<Ort::AllocatedStringPtr> inNamePtrs;
            std::vector<Ort::AllocatedStringPtr> outNamePtrs;
            std::vector<const char*> inNames;
            std::vector<const char*> outNames;
            for (size_t i = 0; i < nIn; ++i) {
                inNamePtrs.push_back(session->GetInputNameAllocated(i, alloc));
                inNames.push_back(inNamePtrs.back().get());
            }
            for (size_t i = 0; i < nOut; ++i) {
                outNamePtrs.push_back(session->GetOutputNameAllocated(i, alloc));
                outNames.push_back(outNamePtrs.back().get());
            }

            std::array<int64_t, 4> inShape = {1, 3, kIn, kIn};
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                    memInfo, input.data(), input.size(), inShape.data(), inShape.size());

            t0 = currentTimeMs();
            std::vector<Ort::Value> outputs = session->Run(
                    Ort::RunOptions{nullptr},
                    inNames.data(), &inputTensor, 1,
                    outNames.data(), outNames.size());
            t1 = currentTimeMs();

            if (outputs.empty()) {
                terminal = [cb]() { cb->onError(W_MODEL_ERROR, "ORT Run returned no outputs"); };
                goto done;
            }

            // v0.5: dispatch on family. Default-fill in CapabilityRegistry +
            // OEM knob (vision.detect.family). "yolov8" / "yolov5" use the
            // anchor-based YOLO parser; "rtdetr" / "detr" use the query-based
            // DETR parser. Anything else falls back to YOLOv8 with a warning.
            const bool useDetr = (detectFamily == "rtdetr" || detectFamily == "detr");
            Ort::Value& outTensor = outputs[0];
            auto info = outTensor.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape();
            float* outData = outTensor.GetTensorMutableData<float>();

            // v0.5 V7: NMS thresholds are OEM-tunable via <capability_tuning>
            // in oir_config.xml. Defaults (0.25/0.45) match v0.4 behavior.
            const float kScoreThresh = mDetectScoreThresh;
            const float kIouThresh   = mDetectIouThresh;

            struct Candidate {
                float x1, y1, x2, y2;
                int classIdx;
                float score;
            };
            std::vector<Candidate> cands;
            cands.reserve(256);

            auto unletterbox = [&](float& x1, float& y1, float& x2, float& y2) {
                x1 = std::max(0.0f, std::min((x1 - padX) / scale, (float)img.w));
                y1 = std::max(0.0f, std::min((y1 - padY) / scale, (float)img.h));
                x2 = std::max(0.0f, std::min((x2 - padX) / scale, (float)img.w));
                y2 = std::max(0.0f, std::min((y2 - padY) / scale, (float)img.h));
            };

            if (!useDetr) {
                // ---- YOLOv8 parser (anchor-based, single output tensor) ----
                // Output shape: (1, 84, 8400) where 84 = 4 bbox + 80 classes.
                // Some exporters produce (1, 8400, 84) transposed — detect both.
                int nPred = 0;
                int nAttr = 0;
                bool transposed = false;
                if (shape.size() == 3 && shape[0] == 1) {
                    if (shape[1] == 84 && shape[2] > 100) {
                        nAttr = (int)shape[1]; nPred = (int)shape[2];
                    } else if (shape[2] == 84 && shape[1] > 100) {
                        nPred = (int)shape[1]; nAttr = (int)shape[2]; transposed = true;
                    }
                }
                if (nAttr != 84 || nPred < 1) {
                    std::string msg = "unexpected YOLO output shape; need (1,84,N) or (1,N,84)";
                    terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                    goto done;
                }
                const int nClasses = nAttr - 4;

                auto at = [&](int anchor, int attr) -> float {
                    return transposed ? outData[anchor * nAttr + attr]
                                       : outData[attr * nPred + anchor];
                };

                for (int a = 0; a < nPred; ++a) {
                    int bestCls = -1;
                    float bestScore = 0.0f;
                    for (int c = 0; c < nClasses; ++c) {
                        float s = at(a, 4 + c);
                        if (s > bestScore) { bestScore = s; bestCls = c; }
                    }
                    if (bestScore < kScoreThresh) continue;

                    // YOLOv8 bbox is (cx, cy, w, h) in kIn × kIn space.
                    float cx = at(a, 0), cy = at(a, 1);
                    float w  = at(a, 2), h  = at(a, 3);
                    float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
                    float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;
                    unletterbox(x1, y1, x2, y2);
                    cands.push_back({x1, y1, x2, y2, bestCls, bestScore});
                }
            } else {
                // ---- RT-DETR / DETR parser (query-based) ----
                // Accepts two export conventions:
                //   (a) PaddlePaddle/NVIDIA triple: "boxes" [B,N,4], "scores" [B,N],
                //       "labels" [B,N int64].
                //   (b) HuggingFace transformers pair: "pred_boxes" [B,N,4],
                //       "logits" [B,N,C] — scores/labels derived via argmax per query.
                // Boxes are typically cxcywh; some exports use xyxy. Heuristic:
                // if any value > 2.0 → already in pixel space (kIn-scale); else
                // assume normalized 0-1 and multiply by kIn.
                const float* boxesData  = nullptr;
                const float* scoresData = nullptr;
                const int64_t* labelsData = nullptr;
                const float* logitsData = nullptr;
                int logitsNumClasses = 0;
                int nQueries = 0;
                int boxStride = 4;
                bool xyxy = false;
                for (size_t i = 0; i < outputs.size(); ++i) {
                    auto si = outputs[i].GetTensorTypeAndShapeInfo();
                    auto sh = si.GetShape();
                    std::string nm = (i < outNames.size()) ? std::string(outNames[i]) : "";
                    auto elemType = si.GetElementType();
                    if (nm == "boxes" || nm == "pred_boxes"
                        || (sh.size() == 3 && sh.back() == 4 && boxesData == nullptr)) {
                        boxesData = outputs[i].GetTensorData<float>();
                        if (sh.size() == 3) nQueries = (int)sh[1];
                        if (nm.find("xyxy") != std::string::npos) xyxy = true;
                    } else if (nm == "logits"
                               || (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                   && sh.size() == 3 && logitsData == nullptr)) {
                        // HF transformers: [B, N, C] class logits per query.
                        logitsData = outputs[i].GetTensorData<float>();
                        logitsNumClasses = (int)sh[2];
                        if (nQueries == 0) nQueries = (int)sh[1];
                    } else if (nm == "scores" || (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                                  && sh.size() == 2 && scoresData == nullptr
                                                  && (int)sh[1] == nQueries)) {
                        scoresData = outputs[i].GetTensorData<float>();
                    } else if (nm == "labels" || elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                        labelsData = outputs[i].GetTensorData<int64_t>();
                    }
                }
                // Derive scores/labels from logits if the model uses the HF
                // (logits + pred_boxes) export convention.
                std::vector<float>   derivedScores;
                std::vector<int64_t> derivedLabels;
                if (logitsData && nQueries > 0 && logitsNumClasses > 0
                    && (!scoresData || !labelsData)) {
                    derivedScores.resize(nQueries);
                    derivedLabels.resize(nQueries);
                    for (int q = 0; q < nQueries; ++q) {
                        const float* row = logitsData + (size_t)q * logitsNumClasses;
                        int bestC = 0;
                        float bestL = row[0];
                        for (int c = 1; c < logitsNumClasses; ++c) {
                            if (row[c] > bestL) { bestL = row[c]; bestC = c; }
                        }
                        // Sigmoid for RT-DETR (which uses focal-loss-style per-class
                        // sigmoid scores, not softmax). Good enough as a scalar
                        // confidence signal downstream of NMS.
                        derivedScores[q] = 1.0f / (1.0f + std::exp(-bestL));
                        derivedLabels[q] = bestC;
                    }
                    scoresData = derivedScores.data();
                    labelsData = derivedLabels.data();
                }
                if (!boxesData || !scoresData || !labelsData || nQueries < 1) {
                    terminal = [cb]() {
                        cb->onError(W_MODEL_ERROR,
                                    "RT-DETR output mismatch: need boxes+scores+labels "
                                    "(triple) or pred_boxes+logits (HF)");
                    };
                    goto done;
                }
                // Box-space heuristic: scan first 16 values for any > 2.0.
                bool inPixelSpace = false;
                for (int i = 0; i < std::min(16, nQueries * 4); ++i) {
                    if (std::fabs(boxesData[i]) > 2.0f) { inPixelSpace = true; break; }
                }
                const float boxScale = inPixelSpace ? 1.0f : (float)kIn;
                for (int q = 0; q < nQueries; ++q) {
                    float s = scoresData[q];
                    if (s < kScoreThresh) continue;
                    int cls = (int)labelsData[q];
                    float a = boxesData[q * 4 + 0] * boxScale;
                    float b = boxesData[q * 4 + 1] * boxScale;
                    float c = boxesData[q * 4 + 2] * boxScale;
                    float d = boxesData[q * 4 + 3] * boxScale;
                    float x1, y1, x2, y2;
                    if (xyxy) {
                        x1 = a; y1 = b; x2 = c; y2 = d;
                    } else {
                        // cxcywh
                        x1 = a - c * 0.5f; y1 = b - d * 0.5f;
                        x2 = a + c * 0.5f; y2 = b + d * 0.5f;
                    }
                    unletterbox(x1, y1, x2, y2);
                    cands.push_back({x1, y1, x2, y2, cls, s});
                }
            }

            // Sort by score descending.
            std::sort(cands.begin(), cands.end(),
                      [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

            // NMS per class.
            std::vector<bool> keep(cands.size(), true);
            auto iou = [](const Candidate& a, const Candidate& b) -> float {
                float ix1 = std::max(a.x1, b.x1);
                float iy1 = std::max(a.y1, b.y1);
                float ix2 = std::min(a.x2, b.x2);
                float iy2 = std::min(a.y2, b.y2);
                float iw = std::max(0.0f, ix2 - ix1);
                float ih = std::max(0.0f, iy2 - iy1);
                float inter = iw * ih;
                float aArea = std::max(0.0f, (a.x2 - a.x1)) * std::max(0.0f, (a.y2 - a.y1));
                float bArea = std::max(0.0f, (b.x2 - b.x1)) * std::max(0.0f, (b.y2 - b.y1));
                float uni = aArea + bArea - inter;
                return uni > 0.0f ? inter / uni : 0.0f;
            };
            for (size_t i = 0; i < cands.size(); ++i) {
                if (!keep[i]) continue;
                for (size_t j = i + 1; j < cands.size(); ++j) {
                    if (!keep[j]) continue;
                    if (cands[j].classIdx != cands[i].classIdx) continue;
                    if (iou(cands[i], cands[j]) > kIouThresh) keep[j] = false;
                }
            }

            // Embedded COCO-80 fallback (standard YOLOv8 training set). OEMs
            // override by shipping <model>.classes.json alongside the .onnx —
            // parsed at load time and resolved in the labels block below (v0.5 V8).
            static const char* kCoco80[] = {
                "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
                "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
                "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
                "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
                "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
                "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
                "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
                "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
            };

            // Build flattened arrays for the callback.
            std::vector<int> xs, ys, widths, heights, labelsPerBox;
            std::vector<std::string> labelsFlat;
            std::vector<float> scoresFlat;
            int nKept = 0;
            for (size_t i = 0; i < cands.size(); ++i) {
                if (!keep[i]) continue;
                const auto& c = cands[i];
                int cls = c.classIdx;
                // v0.5 V8: prefer OEM sidecar labels; fall back to embedded COCO-80.
                std::string name;
                if (cls >= 0 && cls < (int)classLabels.size()) {
                    name = classLabels[cls];
                } else if (cls >= 0 && cls < (int)(sizeof(kCoco80)/sizeof(kCoco80[0]))) {
                    name = kCoco80[cls];
                } else {
                    name = "unknown";
                }
                xs.push_back((int)std::round(c.x1));
                ys.push_back((int)std::round(c.y1));
                widths.push_back((int)std::round(c.x2 - c.x1));
                heights.push_back((int)std::round(c.y2 - c.y1));
                labelsPerBox.push_back(1);
                labelsFlat.push_back(std::move(name));
                scoresFlat.push_back(c.score);
                ++nKept;
            }

            candCount = cands.size();
            keptCount = nKept;
            terminal = [cb,
                        xs = std::move(xs),
                        ys = std::move(ys),
                        widths = std::move(widths),
                        heights = std::move(heights),
                        labelsPerBox = std::move(labelsPerBox),
                        labelsFlat = std::move(labelsFlat),
                        scoresFlat = std::move(scoresFlat)]() {
                cb->onBoundingBoxes(xs, ys, widths, heights,
                                    labelsPerBox, labelsFlat, scoresFlat);
            };
        } catch (const Ort::Exception& e) {
            std::string msg = std::string("ORT inference failed: ") + e.what();
            terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
        }
        }
    done:
        releaseInflight(modelHandle);
        if (terminal) terminal();

        LOG(INFO) << "oird: detect handle=" << modelHandle
                  << " img=" << imgW << "x" << imgH
                  << " candidates=" << candCount
                  << " kept=" << keptCount
                  << " wall_ms=" << (t1 - t0);
            });  // v0.6.4: close mScheduler->enqueue lambda
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.4 H4-B: vision.embed — AIDL + ORT session load + preprocess + pool
    // shipped real. SigLIP-base validated on cvd (768-dim output vector).
    ::ndk::ScopedAStatus loadVisionEmbed(const std::string& modelPath, int64_t* _aidl_return) override {
        // v0.6.9: mLock shrunk. Original code had a nested lock_guard on
        // mLock inside the validate block (line ~3364) while already
        // holding the outer lock_guard → non-recursive self-deadlock.
        // The snapshot now happens before the slow ctor and after lock
        // release.
        const std::string key = "onnx-ve:" + modelPath;
        std::unique_lock<std::mutex> lk(mLock);
        for (auto& [h, m] : mModels) {
            if (m.path == modelPath && m.isVisionEmbed) {
                *_aidl_return = h;
                return ::ndk::ScopedAStatus::ok();
            }
        }

        LoadClaim claim = claimLoadSlot(lk, key);
        if (claim.waited) {
            if (claim.waited->errCode != 0) {
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        claim.waited->errCode, claim.waited->errMsg.c_str());
            }
            *_aidl_return = claim.waited->handle;
            return ::ndk::ScopedAStatus::ok();
        }
        auto slot = claim.slot;

        const int64_t newSize = fileSizeBytes(modelPath);
        const int32_t kTarget = mVisionEmbedInputSize;
        mTotalBytes += newSize;

        lk.unlock();

        ensureOrtEnv();
        Ort::SessionOptions so = makeOrtSessionOptions(false);
        Ort::Session* session = nullptr;
        try {
            session = new Ort::Session(*mOrtEnv, modelPath.c_str(), so);
        } catch (const Ort::Exception& e) {
            LOG(ERROR) << "oird: Ort::Session (vision embed) failed: " << e.what();
            const std::string msg = std::string("vision embed load failed: ") + e.what();
            lk.lock();
            mTotalBytes -= newSize;
            publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, msg);
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, msg.c_str());
        }
        // v0.6 Phase A: SigLIP-style shape contract — [1, 3, kTarget, kTarget]
        // with batch wildcard. Output dim varies by model (768 base / 1024
        // large), not constrained here.
        {
            std::vector<std::vector<int64_t>> inShapes = {
                {-1, 3, kTarget, kTarget},
            };
            std::string err = validateOrtContract(session, 1, inShapes, "vision.embed");
            if (!err.empty()) {
                LOG(ERROR) << "oird: " << err;
                delete session;
                lk.lock();
                mTotalBytes -= newSize;
                publishLoadResult(lk, key, slot, 0, W_MODEL_INCOMPATIBLE, err);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_MODEL_INCOMPATIBLE, err.c_str());
            }
        }

        lk.lock();

        const int64_t handle = mNextModelHandle++;
        const int64_t now = currentTimeMs();
        LoadedModel lm;
        lm.ortSession = session;
        lm.handle = handle;
        lm.path = modelPath;
        lm.sizeBytes = newSize;
        lm.loadTimestampMs = now;
        lm.lastAccessMs = now;
        lm.isOnnx = true;
        lm.isVisionEmbed = true;
        mModels[handle] = std::move(lm);

        publishLoadResult(lk, key, slot, handle, 0, "");

        *_aidl_return = handle;
        LOG(INFO) << "oird: vision embed model loaded handle=" << handle << " path=" << modelPath;
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.4 H4-B: vision.embed real impl.
    // Decode → resize 224×224 (bilinear) → SigLIP normalize (mean=0.5 std=0.5)
    // → ORT Run → mean-pool if patch-tokens → L2 normalize → emit vector.
    ::ndk::ScopedAStatus submitVisionEmbed(int64_t modelHandle,
                                           const std::string& imagePath,
                                           const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                           int64_t* _aidl_return) override {
        Ort::Session* session = nullptr;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isVisionEmbed) {
                cb->onError(W_INVALID_INPUT, "handle not a vision embed model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            session = it->second.ortSession;
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue on scheduler.
        mScheduler->enqueue(priorityForCapability("vision.embed"),
            [this, modelHandle, imagePath, cb, session]() {
        // v0.6.8: terminal cb deferred past releaseInflight.
        std::function<void()> terminal;
        int imgW = 0, imgH = 0;
        size_t vecDim = 0;
        int64_t t0 = 0, t1 = 0;
        {
        // Decode — sniff extension (jpg/jpeg/png).
        RgbImage img;
        bool ok = false;
        std::string ext4 = imagePath.size() >= 4 ? imagePath.substr(imagePath.size() - 4) : "";
        std::string ext5 = imagePath.size() >= 5 ? imagePath.substr(imagePath.size() - 5) : "";
        for (auto& c : ext4) c = (char)tolower((unsigned char)c);
        for (auto& c : ext5) c = (char)tolower((unsigned char)c);
        if (ext4 == ".jpg" || ext5 == ".jpeg") ok = decodeJpeg(imagePath, img, mImageMaxPixels);
        else if (ext4 == ".png") ok = decodePng(imagePath, img, mImageMaxPixels);
        if (!ok) {
            std::string msg = "image decode failed (need .jpg/.jpeg/.png): " + imagePath;
            terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
            goto done;
        }
        imgW = img.w; imgH = img.h;

        // v0.5 V7: OEM-tunable input size + normalization (defaults: SigLIP-base
        // 224 w/ mean=0.5, std=0.5 → [-1, 1] range).
        int32_t kTarget;
        float normMean;
        float normStd;
        {
            std::lock_guard<std::mutex> lk(mLock);
            kTarget  = mVisionEmbedInputSize;
            normMean = mVisionEmbedNormMean;
            normStd  = mVisionEmbedNormStd;
        }
        // Preprocess: bilinear resize kTarget×kTarget, CHW float, encoder normalization.
        std::vector<float> input(3 * kTarget * kTarget);
        const float sx = (float)img.w / kTarget;
        const float sy = (float)img.h / kTarget;
        for (int y = 0; y < kTarget; ++y) {
            float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = std::max(0, (int)std::floor(fy));
            int y1 = std::min(img.h - 1, y0 + 1);
            float wy = fy - y0;
            for (int x = 0; x < kTarget; ++x) {
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = std::max(0, (int)std::floor(fx));
                int x1 = std::min(img.w - 1, x0 + 1);
                float wx = fx - x0;
                for (int c = 0; c < 3; ++c) {
                    float p00 = img.px[(y0 * img.w + x0) * 3 + c];
                    float p01 = img.px[(y0 * img.w + x1) * 3 + c];
                    float p10 = img.px[(y1 * img.w + x0) * 3 + c];
                    float p11 = img.px[(y1 * img.w + x1) * 3 + c];
                    float a = p00 * (1 - wx) + p01 * wx;
                    float b = p10 * (1 - wx) + p11 * wx;
                    float v = (a * (1 - wy) + b * wy) / 255.0f;
                    v = (v - normMean) / normStd;
                    input[c * kTarget * kTarget + y * kTarget + x] = v;
                }
            }
        }

        // ORT Run.
        try {
            Ort::AllocatorWithDefaultOptions alloc;
            if (session->GetInputCount() < 1 || session->GetOutputCount() < 1) {
                terminal = [cb]() { cb->onError(W_MODEL_ERROR, "ORT session has 0 inputs or outputs"); };
                goto done;
            }
            Ort::AllocatedStringPtr inNamePtr  = session->GetInputNameAllocated(0, alloc);
            Ort::AllocatedStringPtr outNamePtr = session->GetOutputNameAllocated(0, alloc);
            const char* inName  = inNamePtr.get();
            const char* outName = outNamePtr.get();

            std::array<int64_t, 4> inShape = {1, 3, kTarget, kTarget};
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                    memInfo, input.data(), input.size(), inShape.data(), inShape.size());

            t0 = currentTimeMs();
            std::vector<Ort::Value> outputs = session->Run(
                    Ort::RunOptions{nullptr},
                    &inName, &inputTensor, 1,
                    &outName, 1);
            t1 = currentTimeMs();

            if (outputs.empty()) {
                terminal = [cb]() { cb->onError(W_MODEL_ERROR, "ORT Run returned no outputs"); };
                goto done;
            }
            Ort::Value& outTensor = outputs[0];
            auto info = outTensor.GetTensorTypeAndShapeInfo();
            size_t n_elems = info.GetElementCount();
            float* outData = outTensor.GetTensorMutableData<float>();
            std::vector<int64_t> shape = info.GetShape();

            std::vector<float> vec;
            if (shape.size() == 2 && shape[0] == 1) {
                // (1, dim) — already pooled.
                vec.assign(outData, outData + shape[1]);
            } else if (shape.size() == 3 && shape[0] == 1) {
                // (1, n_patches, dim) — mean-pool over patches.
                int n_patches = (int)shape[1];
                int dim = (int)shape[2];
                vec.assign(dim, 0.0f);
                for (int p = 0; p < n_patches; ++p)
                    for (int d = 0; d < dim; ++d)
                        vec[d] += outData[p * dim + d];
                for (float& v : vec) v /= n_patches;
            } else {
                // Unknown shape — return raw and let caller deal.
                vec.assign(outData, outData + n_elems);
            }

            // L2-normalize for cosine-similarity convention.
            double sum2 = 0.0;
            for (float v : vec) sum2 += (double)v * v;
            float norm = (float)std::sqrt(sum2);
            if (norm > 1e-8f) for (float& v : vec) v /= norm;

            vecDim = vec.size();
            terminal = [cb, vec = std::move(vec)]() { cb->onVector(vec); };
        } catch (const Ort::Exception& e) {
            std::string msg = std::string("ORT inference failed: ") + e.what();
            terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
        }
        }
    done:
        releaseInflight(modelHandle);
        if (terminal) terminal();
        LOG(INFO) << "oird: vision embed handle=" << modelHandle
                  << " img=" << imgW << "x" << imgH
                  << " dim=" << vecDim
                  << " wall_ms=" << (t1 - t0);
            });  // v0.6.4: close mScheduler->enqueue lambda
        return ::ndk::ScopedAStatus::ok();
    }

    // ========================================================================
    // v0.4 H6 — vision.describe via clip.cpp + LLaVA-family VLM.
    //
    // Two model files: CLIP mmproj (.gguf) + LLM (.gguf). Loaded together
    // as a single LoadedModel handle — both share budget, both evict as one.
    // ========================================================================

    ::ndk::ScopedAStatus loadVlm(const std::string& clipPath,
                                 const std::string& llmPath,
                                 int64_t* _aidl_return) override {
        // v0.6.9: mLock shrunk around slow ctor (LLaVA-1.5-7b llama_model_load
        // + pool of mtmd_init is ~5-8s on cvd CPU; blocking mLock that long
        // stalled every other OIRService submit path during H6 validation).
        const std::string combined = clipPath + "|" + llmPath;
        const std::string key = "vlm:" + combined;
        std::unique_lock<std::mutex> lk(mLock);

        for (auto& [h, m] : mModels) {
            if (m.path == combined && m.isVlm) {
                LOG(INFO) << "oird: VLM already loaded handle=" << h;
                *_aidl_return = h;
                return ::ndk::ScopedAStatus::ok();
            }
        }

        LoadClaim claim = claimLoadSlot(lk, key);
        if (claim.waited) {
            if (claim.waited->errCode != 0) {
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        claim.waited->errCode, claim.waited->errMsg.c_str());
            }
            *_aidl_return = claim.waited->handle;
            return ::ndk::ScopedAStatus::ok();
        }
        auto slot = claim.slot;

        // Budget check (sum of both files)
        const int64_t newSize = fileSizeBytes(clipPath) + fileSizeBytes(llmPath);
        const int64_t MB = 1024 * 1024;
        if (mBudgetMb > 0 && (mTotalBytes + newSize) > (int64_t)mBudgetMb * MB) {
            const int64_t budgetBytes = (int64_t)mBudgetMb * MB;
            const int64_t now = currentTimeMs();
            std::vector<std::pair<int64_t, int64_t>> candidates;
            for (const auto& [h, m] : mModels) {
                if (m.inFlightCount > 0) continue;
                if (m.warmUntilMs > now) continue;
                candidates.emplace_back(m.lastAccessMs, h);
            }
            std::sort(candidates.begin(), candidates.end());
            int64_t freed = 0;
            for (const auto& [_ts, h] : candidates) {
                if (mTotalBytes + newSize - freed <= budgetBytes) break;
                auto it = mModels.find(h);
                if (it == mModels.end()) continue;
                mLlamaPools.erase(h);
                {
                    auto oit = mOcrRec.find(h);
                    if (oit != mOcrRec.end()) {
                        delete oit->second.session;
                        mOcrRec.erase(oit);
                    }
                }
                if (it->second.ctx) llama_free(it->second.ctx);
                if (it->second.model) llama_model_free(it->second.model);
                mWhisperPools.erase(h);
                it->second.wctx = nullptr;
                delete it->second.ortSession;
                if (it->second.mtmdCtx) mtmd_free(it->second.mtmdCtx);
                freed += it->second.sizeBytes;
                LOG(INFO) << "oird: evicted handle=" << h;
                mModels.erase(it);
                mEvictionCount++;
            }
            mTotalBytes -= freed;
            if (mTotalBytes + newSize > budgetBytes) {
                const std::string msg = "budget exceeded; nothing evictable";
                publishLoadResult(lk, key, slot, 0, W_INSUFFICIENT_MEMORY, msg);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_INSUFFICIENT_MEMORY, msg.c_str());
            }
        }

        // Snapshot tunables + reserve budget under lock.
        const int32_t vlmNCtx   = mVisionDescribeNCtx;
        const int32_t vlmNBatch = mVisionDescribeNBatch;
        const int32_t poolSize  = std::max(1, mVisionDescribeContextsPerModel);
        mTotalBytes += newSize;

        lk.unlock();

        LOG(INFO) << "oird: loading VLM clip=" << clipPath << " llm=" << llmPath;

        const int32_t totalCores = std::max(2, (int32_t)sysconf(_SC_NPROCESSORS_ONLN));
        int32_t threads = std::max(1, totalCores / poolSize);

        // v0.5 V1: llm-first because mtmd_init_from_file binds the image
        // encoder against the llama_model.
        llama_model_params mparams = llama_model_default_params();
        mparams.use_mmap = true;
        llama_model* model = llama_model_load_from_file(llmPath.c_str(), mparams);
        if (!model) {
            LOG(ERROR) << "oird: llama_model_load_from_file failed for " << llmPath;
            lk.lock();
            mTotalBytes -= newSize;
            publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, "VLM text-model load failed");
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, "VLM text-model load failed");
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx   = vlmNCtx;
        cparams.n_batch = vlmNBatch;
        cparams.n_ubatch = vlmNBatch;
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;

        mtmd_context_params mtmdParams = mtmd_context_params_default();
        mtmdParams.use_gpu = false;
        mtmdParams.print_timings = false;
        mtmdParams.n_threads = threads;

        std::vector<PooledContext> pooledCtxs;
        pooledCtxs.reserve(poolSize);
        auto cleanup_and_fail = [&](const char* what, int32_t code, const std::string& msg) {
            for (auto& p : pooledCtxs) {
                if (p.mctx) mtmd_free(p.mctx);
                if (p.ctx)  llama_free(p.ctx);
            }
            llama_model_free(model);
            lk.lock();
            mTotalBytes -= newSize;
            publishLoadResult(lk, key, slot, 0, code, msg);
            LOG(ERROR) << "oird: VLM " << what << " failed";
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(code, msg.c_str());
        };
        for (int32_t i = 0; i < poolSize; ++i) {
            llama_context* c = llama_init_from_model(model, cparams);
            if (!c) {
                return cleanup_and_fail("llama_init_from_model", W_MODEL_ERROR,
                        "VLM context init failed");
            }
            mtmd_context* mctx = mtmd_init_from_file(clipPath.c_str(), model, mtmdParams);
            if (!mctx) {
                llama_free(c);
                return cleanup_and_fail("mtmd_init_from_file", W_MODEL_ERROR, "mtmd init failed");
            }
            pooledCtxs.push_back({c, mctx});
        }

        const int64_t kvPerCtx = estimateKvBytesPerContext(model, vlmNCtx);
        const int64_t poolKvBytes = kvPerCtx * poolSize;

        lk.lock();

        const int64_t handle = mNextModelHandle++;
        const int64_t now = currentTimeMs();
        LoadedModel lm;
        lm.model = model;
        lm.vocab = llama_model_get_vocab(model);
        lm.ctx = nullptr;
        lm.context_size = vlmNCtx;
        lm.n_threads = threads;
        lm.mtmdCtx = nullptr;
        lm.handle = handle;
        lm.path = combined;
        lm.sizeBytes = newSize + poolKvBytes;
        lm.loadTimestampMs = now;
        lm.lastAccessMs = now;
        lm.isVlm = true;
        lm.hasLlamaPool = true;
        mTotalBytes += poolKvBytes;
        mModels[handle] = std::move(lm);
        mLlamaPools[handle] = std::make_unique<ContextPool>(std::move(pooledCtxs));

        publishLoadResult(lk, key, slot, handle, 0, "");

        *_aidl_return = handle;
        LOG(INFO) << "oird: VLM loaded handle=" << handle
                  << " size=" << (newSize >> 20) << "MB"
                  << " pool=" << poolSize << " (ctx+mtmd) × "
                  << (kvPerCtx >> 20) << "MB KV"
                  << " resident=" << (mTotalBytes >> 20) << "/" << mBudgetMb << "MB";
        return ::ndk::ScopedAStatus::ok();
    }

    // Describe an image. Uses LLaVA-style template:
    //   USER: <image-tokens>\n<prompt>\nASSISTANT:
    // Token streaming via cb->onToken(String, int outputIndex).
    ::ndk::ScopedAStatus submitDescribeImage(int64_t modelHandle,
                                             const std::string& imagePath,
                                             const std::string& prompt,
                                             const std::shared_ptr<IOirWorkerCallback>& cb,
                                             int64_t* _aidl_return) override {
        LoadedModel* lmPtr = nullptr;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isVlm) {
                cb->onError(W_INVALID_INPUT, "handle not a VLM");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
            lmPtr = &it->second;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue on scheduler.
        mScheduler->enqueue(priorityForCapability("vision.describe"),
            [this, modelHandle, imagePath, prompt, cb, lmPtr]() {
        // v0.6.8: onToken stream stays inside the lease (incremental); only
        // onComplete / onError are deferred past lease + releaseInflight.
        std::function<void()> terminal;
        int outputIndex = 0;
        int64_t totalMs = 0;
        {
        // v0.6 Phase A: lease a VLM slot (pair of llama_ctx + mtmd_ctx).
        // Priority + timeout per vision.describe knobs. Default pool=1 so
        // concurrent describe calls on the same VLM queue up honoring
        // priority order, rather than all piling on a single context.
        ContextPool* pool = nullptr;
        int priority = ContextPool::PRIO_NORMAL;
        std::chrono::milliseconds timeout{60000};
        const llama_vocab* vocab = lmPtr->vocab;
        const int n_batch = mVisionDescribeNBatch;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto pit = mLlamaPools.find(modelHandle);
            if (pit == mLlamaPools.end()) {
                terminal = [cb]() { cb->onError(W_MODEL_ERROR, "VLM has no context pool"); };
                goto done;
            }
            pool = pit->second.get();
            priority = mVisionDescribePriority;
            timeout = std::chrono::milliseconds(mVisionDescribeAcquireTimeoutMs);
        }
        {
        ContextLease lease(*pool, priority, timeout);
        llama_context* ctx = lease.ctx();
        mtmd_context* mctx = lease.mctx();
        if (!ctx || !mctx) {
            terminal = [cb]() { cb->onError(W_TIMEOUT, "vision.describe pool acquire timed out"); };
            goto done;
        }

        int64_t t0 = currentTimeMs();
        int64_t firstTokenMs = -1;

        llama_memory_clear(llama_get_memory(ctx), true);

        // v0.5 V1: VLM input pipeline rebuilt around libmtmd. Build a
        // composite prompt with the default media marker so the tokenizer
        // splits text + image into separate chunks; mtmd_helper_eval_chunks
        // runs both encode + decode passes against the llama_context.
        const std::string userPrompt = prompt.empty()
                ? std::string("Describe this image in detail.")
                : prompt;
        const std::string composed = std::string("USER: ") + mtmd_default_marker()
                + "\n" + userPrompt + "\nASSISTANT:";

        mtmd_input_text textInput;
        textInput.text          = composed.c_str();
        textInput.add_special   = true;
        textInput.parse_special = true;

        mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(mctx, imagePath.c_str());
        if (!bmp) {
            std::string msg = "mtmd_helper_bitmap_init_from_file failed: " + imagePath;
            terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
            goto done;
        }

        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        if (!chunks) {
            mtmd_bitmap_free(bmp);
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "mtmd_input_chunks_init failed"); };
            goto done;
        }

        const mtmd_bitmap* bitmaps[1] = { bmp };
        int32_t tkRes = mtmd_tokenize(mctx, chunks, &textInput, bitmaps, 1);
        if (tkRes != 0) {
            std::string msg = "mtmd_tokenize failed: " + std::to_string(tkRes);
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
            terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
            goto done;
        }

        llama_pos n_past_after = 0;
        int32_t evalRes = mtmd_helper_eval_chunks(
                mctx, ctx, chunks,
                /*n_past=*/0,
                /*seq_id=*/0,
                /*n_batch=*/n_batch,
                /*logits_last=*/true,
                &n_past_after);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        if (evalRes != 0) {
            std::string msg = "mtmd_helper_eval_chunks failed: " + std::to_string(evalRes);
            terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
            goto done;
        }
        // n_past is now tracked internally by mtmd via the seq_id; the
        // generation loop below pulls logits via llama_sampler_sample(ctx,-1)
        // and feeds tokens back through llama_decode without us tracking
        // an explicit position counter.
        (void)n_past_after;

        // Generation loop
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        llama_sampler* sampler = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        // v0.5 V7: OEM-tunable via vision.describe.max_tokens.
        int kMaxTokens;
        { std::lock_guard<std::mutex> lk(mLock); kMaxTokens = mVisionDescribeMaxTokens; }
        for (int i = 0; i < kMaxTokens; ++i) {
            llama_token tok = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) break;

            char buf[128];
            int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            if (n > 0) {
                if (firstTokenMs < 0) firstTokenMs = currentTimeMs() - t0;
                cb->onToken(std::string(buf, n), outputIndex++);
            }

            llama_batch b = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, b) != 0) break;
        }

        llama_sampler_free(sampler);
        totalMs = currentTimeMs() - t0;
        const int32_t ftMs = firstTokenMs < 0 ? 0 : (int32_t)firstTokenMs;
        const int capturedIdx = outputIndex;
        const int64_t capturedTotal = totalMs;
        terminal = [cb, capturedIdx, ftMs, capturedTotal]() {
            cb->onComplete(capturedIdx, ftMs, capturedTotal);
        };
        }  // ContextLease released
        }
    done:
        releaseInflight(modelHandle);
        if (terminal) terminal();

        LOG(INFO) << "oird: describe handle=" << modelHandle
                  << " tokens=" << outputIndex
                  << " totalMs=" << totalMs;
            });  // v0.6.4: close mScheduler->enqueue lambda
        return ::ndk::ScopedAStatus::ok();
    }

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
        mTotalBytes -= it->second.sizeBytes;
        LOG(INFO) << "oird: unloaded handle=" << modelHandle << " path=" << it->second.path
                  << " resident=" << (mTotalBytes >> 20) << "MB";
        mModels.erase(it);
        return ::ndk::ScopedAStatus::ok();
    }

    ::ndk::ScopedAStatus submit(int64_t modelHandle,
                                const std::string& prompt,
                                int32_t maxTokens,
                                float temperature,
                                const std::shared_ptr<IOirWorkerCallback>& callback,
                                int64_t* _aidl_return) override {
        if (callback == nullptr) {
            return ::ndk::ScopedAStatus::fromExceptionCode(EX_NULL_POINTER);
        }

        int64_t handle;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            // v0.6 Phase A: check for a live pool (was: !it->second.ctx).
            if (it == mModels.end() || !it->second.model
                    || !it->second.hasLlamaPool
                    || mLlamaPools.find(modelHandle) == mLlamaPools.end()) {
                callback->onError(W_MODEL_ERROR, "unknown modelHandle");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;  // v0.4 S2-B: protect from eviction
            handle = mNextRequestHandle++;
            mActiveRequests[handle] = std::make_shared<std::atomic_bool>(false);
        }

        if (maxTokens <= 0) maxTokens = 256;
        if (temperature < 0.0f) temperature = 0.7f;

        LOG(INFO) << "oird: submit handle=" << handle
                  << " prompt.len=" << prompt.size()
                  << " maxTokens=" << maxTokens << " temp=" << temperature;

        auto cancelled = mActiveRequests[handle];
        // v0.6.3: cross-backend scheduler. Was a bare `std::thread(...).detach()`;
        // now enqueues at the capability's configured priority so audio-
        // priority submits on a different backend can still jump ahead.
        const int32_t pri = priorityForCapability("text.complete");
        mScheduler->enqueue(pri,
            [this, modelHandle, handle, prompt, maxTokens, temperature,
             cb = callback, cancelled]() {
                runInference(modelHandle, handle, prompt, maxTokens, temperature, cb, cancelled);
            });

        *_aidl_return = handle;
        return ::ndk::ScopedAStatus::ok();
    }

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
                                 int64_t* _aidl_return) override {
        // v0.6.9: mLock shrunk. Original code had a nested lock_guard on
        // mLock (line ~4036) while already holding the outer lock_guard →
        // non-recursive self-deadlock the moment loadVad was called.
        // Snapshot VAD tunables under the initial lock before releasing.
        const std::string key = "vad:" + modelPath;
        std::unique_lock<std::mutex> lk(mLock);
        for (auto& [h, m] : mModels) {
            if (m.path == modelPath && m.isVad) {
                *_aidl_return = h;
                return ::ndk::ScopedAStatus::ok();
            }
        }

        LoadClaim claim = claimLoadSlot(lk, key);
        if (claim.waited) {
            if (claim.waited->errCode != 0) {
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        claim.waited->errCode, claim.waited->errMsg.c_str());
            }
            *_aidl_return = claim.waited->handle;
            return ::ndk::ScopedAStatus::ok();
        }
        auto slot = claim.slot;

        const int64_t newSize = fileSizeBytes(modelPath);
        const int32_t wSamples = mVadWindowSamples;
        const int32_t cSamples = mVadContextSamples;
        mTotalBytes += newSize;

        lk.unlock();

        ensureOrtEnv();
        Ort::SessionOptions so = makeOrtSessionOptions(false);
        Ort::Session* session = nullptr;
        try {
            session = new Ort::Session(*mOrtEnv, modelPath.c_str(), so);
        } catch (const Ort::Exception& e) {
            LOG(ERROR) << "oird: Ort::Session (vad) failed: " << e.what();
            const std::string msg = std::string("vad load failed: ") + e.what();
            lk.lock();
            mTotalBytes -= newSize;
            publishLoadResult(lk, key, slot, 0, W_MODEL_ERROR, msg);
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, msg.c_str());
        }
        // v0.6 Phase A: Silero shape contract — audio input, LSTM state, sr.
        // Wildcards on runtime-dependent dims so OEMs running 8 kHz or
        // adjusted window still load clean.
        {
            std::vector<std::vector<int64_t>> inShapes = {
                {1, (int64_t)(cSamples + wSamples)},
                {2, 1, 128},
                {1},
            };
            std::string err = validateOrtContract(session, 3, inShapes, "audio.vad");
            if (!err.empty()) {
                LOG(ERROR) << "oird: " << err;
                delete session;
                lk.lock();
                mTotalBytes -= newSize;
                publishLoadResult(lk, key, slot, 0, W_MODEL_INCOMPATIBLE, err);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_MODEL_INCOMPATIBLE, err.c_str());
            }
        }

        lk.lock();

        const int64_t handle = mNextModelHandle++;
        const int64_t now = currentTimeMs();
        LoadedModel lm;
        lm.ortSession = session;
        lm.handle = handle;
        lm.path = modelPath;
        lm.sizeBytes = newSize;
        lm.loadTimestampMs = now;
        lm.lastAccessMs = now;
        lm.isOnnx = true;
        lm.isVad = true;
        mModels[handle] = std::move(lm);

        publishLoadResult(lk, key, slot, handle, 0, "");

        *_aidl_return = handle;
        LOG(INFO) << "oird: vad model loaded handle=" << handle << " path=" << modelPath
                  << " size=" << (newSize >> 10) << "KB";
        return ::ndk::ScopedAStatus::ok();
    }

    ::ndk::ScopedAStatus submitVad(int64_t modelHandle,
                                   const std::string& pcmPath,
                                   const std::shared_ptr<IOirWorkerRealtimeBooleanCallback>& cb,
                                   int64_t* _aidl_return) override {
        Ort::Session* session = nullptr;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end() || !it->second.isVad) {
                cb->onError(W_INVALID_INPUT, "handle not a vad model");
                *_aidl_return = 0;
                return ::ndk::ScopedAStatus::ok();
            }
            session = it->second.ortSession;
            it->second.lastAccessMs = currentTimeMs();
            it->second.inFlightCount++;
        }
        const int64_t reqHandle = mNextRequestHandle++;
        *_aidl_return = reqHandle;

        // v0.6.4: enqueue on the cross-backend scheduler at audio-realtime
        // priority so a VAD request on a shared hardware queue jumps ahead
        // of queued text/vision submits. ORT Run() is thread-safe so no
        // pool needed — the scheduler worker drives the per-window loop
        // directly against the cached session pointer.
        mScheduler->enqueue(priorityForCapability("audio.vad"),
            [this, modelHandle, pcmPath, cb, session]() {
                // v0.6.8: onState streams inline; onComplete / onError
                // deferred past releaseInflight.
                std::function<void()> terminal;
                size_t sampleCount = 0;
                int64_t windowsProcessed = 0;
                {
                    std::ifstream f(pcmPath, std::ios::binary);
                    if (!f.is_open()) {
                        std::string msg = "pcm open failed: " + pcmPath;
                        terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
                        goto done;
                    }
                    f.seekg(0, std::ios::end);
                    const std::streamsize byteCount = f.tellg();
                    f.seekg(0, std::ios::beg);
                    sampleCount = byteCount / 2;
                    std::vector<int16_t> samples(sampleCount);
                    f.read(reinterpret_cast<char*>(samples.data()), sampleCount * 2);
                    f.close();

                    int32_t sampleRateHz;
                    int32_t windowSamples;
                    int32_t contextSamples;
                    float voiceThreshold;
                    {
                        std::lock_guard<std::mutex> lk(mLock);
                        sampleRateHz   = mVadSampleRateHz;
                        windowSamples  = mVadWindowSamples;
                        contextSamples = mVadContextSamples;
                        voiceThreshold = mVadVoiceThreshold;
                    }
                    const int kWindow  = windowSamples;
                    const int kContext = contextSamples;
                    const int kInputLen = kContext + kWindow;
                    const int64_t kSr = sampleRateHz;
                    std::vector<float> state(2 * 1 * 128, 0.0f);
                    std::vector<float> context(kContext, 0.0f);
                    std::vector<float> inputBuf(kInputLen, 0.0f);
                    std::vector<int64_t> srBuf{kSr};

                    const std::array<int64_t, 2> inputShape  = {1, kInputLen};
                    const std::array<int64_t, 3> stateShape  = {2, 1, 128};
                    const int64_t* srShapePtr = nullptr;

                    const char* inputNames[]  = {"input", "state", "sr"};
                    const char* outputNames[] = {"output", "stateN"};

                    const auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

                    try {
                        for (size_t off = 0; off + kWindow <= samples.size(); off += kWindow) {
                            std::copy(context.begin(), context.end(), inputBuf.begin());
                            for (int i = 0; i < kWindow; ++i) {
                                inputBuf[kContext + i] = samples[off + i] / 32768.0f;
                            }

                            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                                    memInfo, inputBuf.data(), inputBuf.size(),
                                    inputShape.data(), inputShape.size());
                            Ort::Value stateTensor = Ort::Value::CreateTensor<float>(
                                    memInfo, state.data(), state.size(),
                                    stateShape.data(), stateShape.size());
                            Ort::Value srTensor = Ort::Value::CreateTensor<int64_t>(
                                    memInfo, srBuf.data(), srBuf.size(),
                                    srShapePtr, 0);

                            std::array<Ort::Value, 3> ins = {
                                std::move(inputTensor),
                                std::move(stateTensor),
                                std::move(srTensor),
                            };
                            auto outs = session->Run(Ort::RunOptions{nullptr},
                                                     inputNames, ins.data(), ins.size(),
                                                     outputNames, 2);
                            const float prob = *outs[0].GetTensorData<float>();
                            const float* newState = outs[1].GetTensorData<float>();
                            std::copy(newState, newState + state.size(), state.begin());
                            std::copy(inputBuf.end() - kContext, inputBuf.end(), context.begin());

                            const bool isVoice = prob > voiceThreshold;
                            const int64_t timestampMs = (off * 1000) / kSr;
                            if (windowsProcessed < 10 || windowsProcessed % 25 == 0) {
                                LOG(INFO) << "oird: vad window=" << windowsProcessed
                                          << " t=" << timestampMs << "ms prob=" << prob;
                            }
                            cb->onState(isVoice, timestampMs);
                            ++windowsProcessed;
                        }
                        terminal = [cb]() { cb->onComplete(); };
                    } catch (const Ort::Exception& e) {
                        std::string msg = std::string("vad inference failed: ") + e.what();
                        terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                    }
                }
            done:
                releaseInflight(modelHandle);
                if (terminal) terminal();
                LOG(INFO) << "oird: vad handle=" << modelHandle
                          << " samples=" << sampleCount
                          << " windows=" << windowsProcessed;
            });
        return ::ndk::ScopedAStatus::ok();
    }

    // v0.4 S2-B: configure budget + warm TTL. Called once by OIRService at attachWorker.
    ::ndk::ScopedAStatus setConfig(int32_t memoryBudgetMb, int32_t warmTtlSeconds) override {
        std::lock_guard<std::mutex> lk(mLock);
        mBudgetMb = memoryBudgetMb;
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
    // v0.6.9: in-progress load registry. Every load*() method that was
    // previously holding mLock across a multi-second model ctor (reported
    // on cvd as 'oird has 8 threads: 7 in futex_wait, 1 idle binder')
    // now (a) reserves a slot in this map under mLock, (b) releases mLock
    // for the slow ctor, (c) re-acquires mLock to insert into mModels +
    // pool map, then (d) publishes result + wakes waiters.
    //
    // Keyed by a kind-qualified path so the same .gguf loaded as both a
    // generation model (load) and an embedding model (loadEmbed) doesn't
    // collide — they're distinct logical loads that produce distinct
    // handles. Concrete keys live at the call sites (one string literal
    // prefix per load method).
    //
    // Deduplication rule: if a second caller arrives for the same key
    // while the first is still in the slow ctor, it waits on mLoadCV
    // rather than racing a duplicate llama_model_load_from_file / Ort
    // session ctor (both expensive and both would double memory).
    struct LoadInProgress {
        bool done = false;
        int64_t handle = 0;        // 0 on failure
        int32_t errCode = 0;       // 0 on success (W_* otherwise)
        std::string errMsg;
        int32_t waiters = 0;       // waiter count; last waiter erases the slot
    };
    std::unordered_map<std::string, std::shared_ptr<LoadInProgress>>
            mLoadingInProgress;
    std::condition_variable mLoadCV;

    // Try to claim an in-progress slot for `key`. Caller must hold `lk`
    // against mLock. Returns:
    //   .slot   non-null  → caller owns the slow ctor; must publish later
    //   .waited non-null  → caller waited on another load; read .waited->
    //                       handle / errCode / errMsg for the result
    struct LoadClaim {
        std::shared_ptr<LoadInProgress> slot;
        std::shared_ptr<LoadInProgress> waited;
    };
    LoadClaim claimLoadSlot(std::unique_lock<std::mutex>& lk,
                            const std::string& key) {
        auto it = mLoadingInProgress.find(key);
        if (it != mLoadingInProgress.end()) {
            auto other = it->second;
            other->waiters++;
            mLoadCV.wait(lk, [&]{ return other->done; });
            other->waiters--;
            // Only the last waiter erases; the owning thread already
            // erased when it published if waiters was 0 at that time.
            if (other->waiters == 0) mLoadingInProgress.erase(key);
            return LoadClaim{nullptr, other};
        }
        auto slot = std::make_shared<LoadInProgress>();
        mLoadingInProgress[key] = slot;
        return LoadClaim{slot, nullptr};
    }

    // Publish the result of a slow ctor to the in-progress slot and
    // wake waiters. Caller must hold mLock via `lk`.
    void publishLoadResult(std::unique_lock<std::mutex>& lk,
                           const std::string& key,
                           const std::shared_ptr<LoadInProgress>& slot,
                           int64_t handle,
                           int32_t errCode,
                           std::string errMsg) {
        (void)lk;
        slot->handle = handle;
        slot->errCode = errCode;
        slot->errMsg = std::move(errMsg);
        slot->done = true;
        mLoadCV.notify_all();
        if (slot->waiters == 0) mLoadingInProgress.erase(key);
    }

    void runInference(int64_t modelHandle,
                      int64_t handle,
                      std::string prompt,
                      int32_t maxTokens,
                      float temperature,
                      std::shared_ptr<IOirWorkerCallback> cb,
                      std::shared_ptr<std::atomic_bool> cancelled) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();

        // v0.6.8: onToken streams inline (incremental); onComplete / onError
        // deferred past lease release + releaseInflight + cleanupRequest so
        // a stalled callback can't pin the ContextLease and deadlock
        // subsequent load()/eviction calls.
        std::function<void()> terminal;
        {
        // v0.6 Phase A: per-model ContextPool. Lease a context slot with
        // priority + timeout so audio-priority submits jump ahead within
        // this llama pool, and no request hangs forever if the pool
        // wedges. (The v0.6 "cross-backend scheduler" story is separate
        // from pool-local priority — see ROADMAP.md v0.7 entry.)

        llama_model* model;
        const llama_vocab* vocab;
        int32_t ctxSize;
        ContextPool* pool = nullptr;
        int priority = ContextPool::PRIO_NORMAL;
        std::chrono::milliseconds timeout{30000};
        int32_t batchSize;
        float tempDefault;
        {
            std::lock_guard<std::mutex> lk(mLock);
            auto it = mModels.find(modelHandle);
            if (it == mModels.end()) {
                terminal = [cb]() { cb->onError(W_MODEL_ERROR, "model unloaded mid-flight"); };
                goto done;
            }
            model = it->second.model;
            vocab = it->second.vocab;
            ctxSize = it->second.context_size;
            auto pit = mLlamaPools.find(modelHandle);
            if (pit != mLlamaPools.end()) pool = pit->second.get();
            priority = mTextCompletePriority;
            timeout = std::chrono::milliseconds(mTextCompleteAcquireTimeoutMs);
            batchSize = mLlamaBatchSize;
            tempDefault = mTextCompleteTemperatureDefault;
        }
        if (!model || !vocab || !pool) {
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "model torn down"); };
            goto done;
        }

        // Priority-aware bounded acquire. Audio submits on a different model
        // don't interact here; audio.*/text.* priority matters only when
        // sharing a pool (not the common case, but correct when it happens).
        {
        ContextLease lease(*pool, priority, timeout);
        llama_context* ctx = lease.ctx();
        if (!ctx) {
            terminal = [cb]() { cb->onError(W_TIMEOUT, "text.complete pool acquire timed out"); };
            goto done;
        }
        // Clear KV from any previous request that used this slot so the new
        // prompt starts fresh. (Prefix caching across requests is a v0.7+
        // perf feature.)
        llama_memory_clear(llama_get_memory(ctx), true);
        // Fold OEM default temperature when caller didn't specify one.
        if (temperature < 0.0f) temperature = tempDefault;

        int32_t n_tokens = llama_tokenize(
                vocab, prompt.c_str(), (int32_t)prompt.size(),
                nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
        if (n_tokens < 0) n_tokens = -n_tokens;

        std::vector<llama_token> tokens(n_tokens);
        n_tokens = llama_tokenize(
                vocab, prompt.c_str(), (int32_t)prompt.size(),
                tokens.data(), (int32_t)tokens.size(), true, true);
        if (n_tokens < 0) {
            terminal = [cb]() { cb->onError(W_INVALID_INPUT, "tokenize failed"); };
            goto done;
        }
        tokens.resize(n_tokens);

        // v0.5 V7: caller-provided maxTokens (submit AIDL) wins; else the
        // OEM-configured text.complete.max_tokens default.
        int32_t defaultMax;
        { std::lock_guard<std::mutex> lk(mLock); defaultMax = mTextCompleteMaxTokens; }
        const int32_t maxGen = maxTokens > 0 ? maxTokens : defaultMax;
        if (n_tokens + maxGen > ctxSize) {
            if (n_tokens >= ctxSize) {
                int32_t keep = ctxSize - maxGen - 1;
                if (keep < 1) keep = 1;
                LOG(WARNING) << "oird: prompt=" << n_tokens
                             << " exceeds ctx=" << ctxSize << "; truncating to " << keep;
                tokens.resize(keep);
                n_tokens = keep;
            }
        }

        // v0.6 Phase A: OEM-tunable batch size via llama.batch_size knob.
        llama_batch batch = llama_batch_init(batchSize, 0, 1);
        int32_t n_eval = 0;
        while (n_eval < n_tokens) {
            if (cancelled->load()) {
                terminal = [cb]() { cb->onError(W_CANCELLED, "cancelled during prefill"); };
                llama_batch_free(batch);
                llama_memory_clear(llama_get_memory(ctx), true);
                goto done;
            }
            int32_t this_batch = std::min(batchSize, n_tokens - n_eval);
            llama_batch_clear_local(batch);
            for (int32_t i = 0; i < this_batch; ++i) {
                bool is_last = (n_eval + i == n_tokens - 1);
                llama_batch_add_local(batch, tokens[n_eval + i],
                        n_eval + i, {0}, is_last);
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG(ERROR) << "oird: prefill decode failed at pos=" << n_eval;
                terminal = [cb]() { cb->onError(W_MODEL_ERROR, "prefill decode failed"); };
                llama_batch_free(batch);
                llama_memory_clear(llama_get_memory(ctx), true);
                goto done;
            }
            n_eval += this_batch;
        }

        llama_sampler* sampler = llama_sampler_chain_init(
                llama_sampler_chain_default_params());
        // v0.6 Phase A: top_p tunable via text.complete.top_p knob.
        float topP;
        { std::lock_guard<std::mutex> lk(mLock); topP = mTextCompleteTopP; }
        if (temperature <= 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));
        }

        int64_t firstTokenMs = -1;
        int outputIndex = 0;
        int32_t emitted = 0;
        int32_t n_cur = n_tokens;

        for (int32_t i = 0; i < maxGen; ++i) {
            if (cancelled->load()) {
                LOG(INFO) << "oird: handle=" << handle << " cancelled step " << i;
                terminal = [cb]() { cb->onError(W_CANCELLED, "cancelled"); };
                llama_sampler_free(sampler);
                llama_batch_free(batch);
                llama_memory_clear(llama_get_memory(ctx), true);
                goto done;
            }

            llama_token next = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, next)) {
                LOG(INFO) << "oird: EOG at step " << i;
                break;
            }
            llama_sampler_accept(sampler, next);

            char piece[256];
            int32_t n = llama_token_to_piece(
                    vocab, next, piece, sizeof(piece),
                    /*lstrip=*/0, /*special=*/false);
            if (n > 0) {
                if (firstTokenMs < 0) {
                    firstTokenMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock::now() - t0).count();
                }
                cb->onToken(std::string(piece, n), outputIndex++);
                ++emitted;
            }

            llama_batch_clear_local(batch);
            llama_batch_add_local(batch, next, n_cur, {0}, true);
            ++n_cur;
            if (llama_decode(ctx, batch) != 0) {
                LOG(ERROR) << "oird: decode failed at generated token " << i;
                terminal = [cb]() { cb->onError(W_MODEL_ERROR, "decode failed mid-stream"); };
                llama_sampler_free(sampler);
                llama_batch_free(batch);
                llama_memory_clear(llama_get_memory(ctx), true);
                goto done;
            }
        }

        int64_t totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - t0).count();
        {
            const int32_t capturedEmitted = emitted;
            const int64_t capturedFirst = firstTokenMs < 0 ? 0 : firstTokenMs;
            const int64_t capturedTotal = totalMs;
            terminal = [cb, capturedEmitted, capturedFirst, capturedTotal]() {
                cb->onComplete(capturedEmitted, capturedFirst, capturedTotal);
            };
        }

        llama_sampler_free(sampler);
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(ctx), true);
        }  // ContextLease released
        }
    done:
        cleanupRequest(handle);
        releaseInflight(modelHandle);
        if (terminal) terminal();
    }

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
    int32_t mBudgetMb = 0;          // 0 = unlimited (default for safety)
    int32_t mWarmTtlSeconds = 60;
    int64_t mTotalBytes = 0;
    int32_t mEvictionCount = 0;
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

} // namespace

int main(int argc, char** argv) {
    // v0.6.9 footgun guard: oird is a binder daemon, not a CLI. Anyone
    // who runs `adb shell oird capabilities` (or any bare argv-passing
    // invocation) starts a SECOND daemon that registers as `oir_worker`
    // and races the init-started instance — servicemanager warns, and
    // the original instance's in-flight calls stall. Only init should
    // exec oird, and init doesn't pass argv. Reject anything else.
    if (argc > 1) {
        fprintf(stderr,
                "oird: daemon has no CLI; use `cmd oir <subcommand>` instead.\n"
                "See `cmd oir help` for supported subcommands.\n");
        return 2;
    }
    android::base::InitLogging(nullptr);
    LOG(INFO) << "oird starting up";

    std::shared_ptr<OirdService> service = ndk::SharedRefBase::make<OirdService>();

    binder_status_t status = AServiceManager_addService(
            service->asBinder().get(), "oir_worker");
    if (status != STATUS_OK) {
        LOG(ERROR) << "oird: AServiceManager_addService failed status=" << status;
        return 1;
    }
    LOG(INFO) << "oird: registered service \"oir_worker\"";

    ABinderProcess_setThreadPoolMaxThreadCount(4);
    ABinderProcess_startThreadPool();
    ABinderProcess_joinThreadPool();
    return 0;
}
