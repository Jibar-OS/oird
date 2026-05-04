// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// pool/whisper_pool.h — per-model whisper_context pool.
//
// v0.7 refactor: extracted verbatim from oird.cpp.
//
// See oird.cpp's "v0.6.2: same-model concurrency for whisper" header
// comment for the design notes.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

#include "whisper.h"

namespace oird {

class WhisperPool {
public:
    WhisperPool(std::vector<whisper_context*> ctxs, int acquireTimeoutMs);
    ~WhisperPool();

    WhisperPool(const WhisperPool&) = delete;
    WhisperPool& operator=(const WhisperPool&) = delete;

    size_t size() const { return mSlots.size(); }

    // v0.6.3: observability — pairs with getRuntimeStats() so `cmd oir
    // memory` can show whisper pool state the same way it shows llama pool
    // state. Both are best-effort snapshots under the pool's own mutex.
    int32_t busyCount() const;
    int32_t waitingCount() const;

    // Blocks up to acquireTimeoutMs waiting for a free slot.
    // Returns nullptr on timeout; sets slotIdxOut to the leased slot on
    // success so the caller can release() on completion.
    whisper_context* acquire(int& slotIdxOut, int64_t& waitMsOut);

    void release(int slotIdx);

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

} // namespace oird
