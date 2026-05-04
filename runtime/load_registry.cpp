// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// runtime/load_registry.cpp — definitions for LoadRegistry.

#include "runtime/load_registry.h"

namespace oird {

LoadRegistry::Claim LoadRegistry::claim(std::unique_lock<std::mutex>& lk,
                                         const std::string& key) {
    auto it = mInProgress.find(key);
    if (it != mInProgress.end()) {
        auto other = it->second;
        other->waiters++;
        mCv.wait(lk, [&]{ return other->done; });
        other->waiters--;
        // Only the last waiter erases; the owning thread already
        // erased when it published if waiters was 0 at that time.
        if (other->waiters == 0) mInProgress.erase(key);
        return Claim{nullptr, other};
    }
    auto slot = std::make_shared<InProgress>();
    mInProgress[key] = slot;
    return Claim{slot, nullptr};
}

void LoadRegistry::publish(std::unique_lock<std::mutex>& lk,
                            const std::string& key,
                            const std::shared_ptr<InProgress>& slot,
                            int64_t handle,
                            int32_t errCode,
                            std::string errMsg) {
    (void)lk;  // borrow contract: caller holds the daemon's mLock.
    slot->handle = handle;
    slot->errCode = errCode;
    slot->errMsg = std::move(errMsg);
    slot->done = true;
    mCv.notify_all();
    if (slot->waiters == 0) mInProgress.erase(key);
}

} // namespace oird
