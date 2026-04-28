// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// runtime/load_registry.h — in-progress load deduplication.
//
// v0.6.9: in-progress load registry. Every load*() method that was
// previously holding the daemon's mLock across a multi-second model ctor
// (reported on cvd as 'oird has 8 threads: 7 in futex_wait, 1 idle binder')
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
// while the first is still in the slow ctor, it waits on the registry's
// internal cv rather than racing a duplicate llama_model_load_from_file
// / Ort session ctor (both expensive and both would double memory).
//
// Mutex coupling: the registry borrows the caller's mutex (passed as
// std::unique_lock<std::mutex>&). This preserves the v0.6.9 invariant
// that the cv.wait inside claim() drops the daemon's mLock so the slow-
// ctor thread can re-acquire it to publish. A self-contained mutex
// would deadlock waiters that hold mLock while calling claim().
#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace oird {

class LoadRegistry {
public:
    struct InProgress {
        bool done = false;
        int64_t handle = 0;        // 0 on failure
        int32_t errCode = 0;       // 0 on success (W_* otherwise)
        std::string errMsg;
        int32_t waiters = 0;       // waiter count; last waiter erases the slot
    };

    struct Claim {
        // Non-null: caller owns the slow ctor; must call publish() later.
        std::shared_ptr<InProgress> slot;
        // Non-null: caller waited on another in-progress load; read
        // .waited->handle / errCode / errMsg for the result.
        std::shared_ptr<InProgress> waited;
    };

    // Try to claim an in-progress slot for `key`. Caller must hold `lk`
    // against the daemon's mLock. Returns Claim per above.
    Claim claim(std::unique_lock<std::mutex>& lk, const std::string& key);

    // Publish the result of a slow ctor to the in-progress slot and
    // wake waiters. Caller must hold the daemon's mLock via `lk`.
    void publish(std::unique_lock<std::mutex>& lk,
                 const std::string& key,
                 const std::shared_ptr<InProgress>& slot,
                 int64_t handle,
                 int32_t errCode,
                 std::string errMsg);

private:
    std::unordered_map<std::string, std::shared_ptr<InProgress>> mInProgress;
    std::condition_variable mCv;
};

} // namespace oird
