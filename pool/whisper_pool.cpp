// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// pool/whisper_pool.cpp — definitions for WhisperPool.

#include "pool/whisper_pool.h"

#include <chrono>

#include <android-base/logging.h>

namespace oird {

WhisperPool::WhisperPool(std::vector<whisper_context*> ctxs, int acquireTimeoutMs)
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

WhisperPool::~WhisperPool() {
    for (auto& s : mSlots) if (s.wctx) whisper_free(s.wctx);
}

int32_t WhisperPool::busyCount() const {
    std::unique_lock<std::mutex> lk(mMu);
    int32_t n = 0;
    for (const auto& s : mSlots) if (s.inUse) ++n;
    return n;
}

int32_t WhisperPool::waitingCount() const {
    std::lock_guard<std::mutex> lk(mMu);
    return mWaiters;
}

whisper_context* WhisperPool::acquire(int& slotIdxOut, int64_t& waitMsOut) {
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

void WhisperPool::release(int slotIdx) {
    std::unique_lock<std::mutex> lk(mMu);
    if (slotIdx < 0 || slotIdx >= static_cast<int>(mSlots.size())) return;
    mSlots[slotIdx].inUse = false;
    mCv.notify_one();
}

} // namespace oird
