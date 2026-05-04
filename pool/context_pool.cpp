// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// pool/context_pool.cpp — definitions for ContextPool + ContextLease.

#include "pool/context_pool.h"

#include <algorithm>

#include <android-base/logging.h>

namespace oird {

ContextPool::ContextPool(std::vector<PooledContext> contexts)
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

ContextPool::~ContextPool() {
    shutdown();  // wakes all waiters, waits for in-flight releases
    std::lock_guard<std::mutex> lk(mMtx);
    for (auto& s : mSlots) {
        if (s.payload.mctx) mtmd_free(s.payload.mctx);
        if (s.payload.ctx)  llama_free(s.payload.ctx);
        s.payload = {};
    }
}

ContextPool::Lease ContextPool::acquire(int priority, std::chrono::milliseconds timeout) {
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

void ContextPool::release(int slotIdx) {
    if (slotIdx < 0) return;
    std::unique_lock<std::mutex> lk(mMtx);
    if (slotIdx >= (int)mSlots.size()) return;
    mSlots[slotIdx].inUse = false;
    mSlots[slotIdx].lastUsedMs = currentTimeMs();
    mBusyCount.fetch_sub(1, std::memory_order_relaxed);
    handOffNextLocked_(slotIdx);
}

void ContextPool::shutdown() {
    std::unique_lock<std::mutex> lk(mMtx);
    if (mShutdown.exchange(true)) return;
    // Wake every waiter — each one will observe mShutdown and bail.
    for (auto* w : mQueue) w->cv.notify_one();
    // Wait for queue to drain. Waiters remove themselves as they wake.
    while (!mQueue.empty()) {
        mDrainedCv.wait_for(lk, std::chrono::milliseconds(100));
    }
}

int32_t ContextPool::waitingCount() const {
    std::lock_guard<std::mutex> lk(mMtx);
    return (int32_t)mQueue.size();
}

std::vector<int64_t> ContextPool::lastUsedSnapshot() const {
    std::lock_guard<std::mutex> lk(mMtx);
    std::vector<int64_t> out;
    out.reserve(mSlots.size());
    for (auto& s : mSlots) out.push_back(s.lastUsedMs);
    return out;
}

int ContextPool::findFreeSlotLocked_() const {
    for (size_t i = 0; i < mSlots.size(); ++i) {
        if (!mSlots[i].inUse && mSlots[i].payload.ctx) return (int)i;
    }
    return -1;
}

ContextPool::Lease ContextPool::takeSlotLocked_(int idx) {
    mSlots[idx].inUse = true;
    mSlots[idx].lastUsedMs = currentTimeMs();
    mBusyCount.fetch_add(1, std::memory_order_relaxed);
    return {mSlots[idx].payload.ctx, mSlots[idx].payload.mctx, idx};
}

void ContextPool::insertSortedLocked_(Waiter* w) {
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

void ContextPool::removeWaiterByPtrLocked_(Waiter* w) {
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
void ContextPool::handOffNextLocked_(int slotIdx) {
    if (mQueue.empty()) return;
    Waiter* w = mQueue.front();
    mQueue.pop_front();
    mSlots[slotIdx].inUse = true;
    mBusyCount.fetch_add(1, std::memory_order_relaxed);
    w->grantedSlot = slotIdx;
    w->cv.notify_one();
    mDrainedCv.notify_all();
}

// ---- ContextLease ----

ContextLease::ContextLease(ContextPool& pool, int priority, std::chrono::milliseconds timeout)
        : mPool(&pool) {
    auto l = pool.acquire(priority, timeout);
    mCtx  = l.ctx;
    mMCtx = l.mctx;
    mSlot = l.slotIdx;
}

ContextLease::~ContextLease() {
    if (mPool && mSlot >= 0) mPool->release(mSlot);
}

ContextLease::ContextLease(ContextLease&& other) noexcept
    : mPool(other.mPool), mCtx(other.mCtx), mMCtx(other.mMCtx), mSlot(other.mSlot) {
    other.mPool = nullptr; other.mCtx = nullptr; other.mMCtx = nullptr; other.mSlot = -1;
}

ContextLease& ContextLease::operator=(ContextLease&& other) noexcept {
    if (this != &other) {
        if (mPool && mSlot >= 0) mPool->release(mSlot);
        mPool = other.mPool; mCtx = other.mCtx; mMCtx = other.mMCtx; mSlot = other.mSlot;
        other.mPool = nullptr; other.mCtx = nullptr; other.mMCtx = nullptr; other.mSlot = -1;
    }
    return *this;
}

} // namespace oird
