// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// pool/context_pool.h — per-model llama_context pool with priority queue.
//
// v0.7 refactor (step 1 of "drop the bake patches" / decompose oird.cpp):
// extracted verbatim from oird.cpp. Behavior unchanged.
//
// See oird.cpp's "v0.6 Phase A" header comment for the design notes
// (priority bands, ownership, hand-off, shutdown drain).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include "llama.h"
#include "mtmd.h"

namespace oird {

// Wall-clock millisecond timestamp. Defined inline so headers that need
// time-stamping (pool/, sched/, ...) don't all duplicate the snippet.
inline int64_t currentTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

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
    explicit ContextPool(std::vector<PooledContext> contexts);
    ~ContextPool();

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
    Lease acquire(int priority, std::chrono::milliseconds timeout);

    void release(int slotIdx);

    // Wake all waiters with invalid Lease; subsequent acquires return
    // invalid immediately. Blocks until queue drains so destructor can
    // safely free contexts without dangling pointers.
    void shutdown();

    int32_t size() const { return (int32_t)mSlots.size(); }

    // Atomic read — no mutex. Safe to call from dumpsys / monitoring.
    int32_t busyCount() const {
        return mBusyCount.load(std::memory_order_relaxed);
    }

    int32_t waitingCount() const;

    // For dumpsys — per-slot last-used-ms to see which contexts are warm.
    std::vector<int64_t> lastUsedSnapshot() const;

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
    // Queue sorted by (priority asc, enqueueMs asc, id asc). Front = next
    // to grant. Non-owning — Waiter lives on acquire()'s stack.
    std::deque<Waiter*> mQueue;
    std::atomic<int32_t> mBusyCount{0};
    std::atomic<bool> mShutdown{false};
    int64_t mNextWaiterId = 0;
    std::condition_variable mDrainedCv;  // signaled when a waiter removes itself

    int findFreeSlotLocked_() const;
    Lease takeSlotLocked_(int idx);
    void insertSortedLocked_(Waiter* w);
    void removeWaiterByPtrLocked_(Waiter* w);
    void handOffNextLocked_(int slotIdx);
};

// RAII lease wrapper — release on scope exit guarantees no slot leak even
// on exception / early-return paths.
class ContextLease {
public:
    ContextLease() = default;
    ContextLease(ContextPool& pool, int priority, std::chrono::milliseconds timeout);
    ~ContextLease();
    ContextLease(const ContextLease&) = delete;
    ContextLease& operator=(const ContextLease&) = delete;
    ContextLease(ContextLease&& other) noexcept;
    ContextLease& operator=(ContextLease&& other) noexcept;

    bool valid() const { return mCtx != nullptr; }
    llama_context* ctx()  const { return mCtx;  }
    mtmd_context*  mctx() const { return mMCtx; }

private:
    ContextPool* mPool = nullptr;
    llama_context* mCtx = nullptr;
    mtmd_context*  mMCtx = nullptr;
    int mSlot = -1;
};

} // namespace oird
