// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// runtime/runtime.cpp — non-inline Runtime method bodies. The hot-path
// methods (releaseInflight, acquireInflightLocked, cleanupRequest) live
// inline in the header; this file holds the resource registry +
// eviction logic where the body is too long for an inline definition.

#include "runtime/runtime.h"

#include <algorithm>

#include <android-base/logging.h>

#include "pool/context_pool.h"   // currentTimeMs()
#include "runtime/model_resource.h"

namespace oird {

Resource* Runtime::registerResourceLocked(std::unique_ptr<Resource> r) {
    Resource* p = r.get();
    mResources.push_back(std::move(r));
    return p;
}

int64_t Runtime::releaseResourceLocked(Resource* r) {
    if (!r) return 0;
    int64_t freed = r->evict();
    auto it = std::find_if(mResources.begin(), mResources.end(),
                           [r](const std::unique_ptr<Resource>& u) {
                               return u.get() == r;
                           });
    if (it != mResources.end()) mResources.erase(it);
    return freed;
}

Resource* Runtime::findModelResourceLocked(int64_t modelHandle) {
    for (auto& r : mResources) {
        if (r->modelHandle() == modelHandle) return r.get();
    }
    return nullptr;
}

int64_t Runtime::evictForBytesLocked(int64_t needed) {
    if (needed <= 0) return 0;
    const int64_t now = currentTimeMs();
    int64_t totalFreed = 0;

    // Snapshot evictable resources sorted by (priority asc, lastAccessMs asc).
    auto candidates = [&]() {
        std::vector<Resource*> out;
        out.reserve(mResources.size());
        for (auto& u : mResources) {
            if (u->isEvictable(now)) out.push_back(u.get());
        }
        std::sort(out.begin(), out.end(),
                  [](Resource* a, Resource* b) {
                      if (a->priority() != b->priority()) {
                          return a->priority() < b->priority();
                      }
                      return a->lastAccessMs() < b->lastAccessMs();
                  });
        return out;
    };

    // Pass 1: pause() — release bytes without killing resources. Models
    // default-implement pause() as no-op so this only helps sessions
    // (when they exist in v0.8+).
    {
        auto pauseList = candidates();
        for (auto* r : pauseList) {
            if (totalFreed >= needed) break;
            int64_t freed = r->pause();
            if (freed > 0) {
                mBudget.subResident(freed);
                totalFreed += freed;
                LOG(INFO) << "oird: paused " << r->description()
                          << " freed=" << (freed >> 20) << "MB";
            }
        }
    }
    if (totalFreed >= needed) return totalFreed;

    // Pass 2: evict() — full teardown. Re-snapshot in case pause()
    // mutated state. evict() removes the resource from the registry
    // (via releaseResourceLocked-equivalent flow below).
    auto evictList = candidates();
    for (auto* r : evictList) {
        if (totalFreed >= needed) break;
        const std::string desc = r->description();  // capture before evict
        int64_t freed = r->evict();
        mBudget.subResident(freed);
        mBudget.recordEviction();
        totalFreed += freed;
        LOG(INFO) << "oird: evicted " << desc
                  << " freed=" << (freed >> 20) << "MB";
        // Drop the unique_ptr from mResources.
        auto it = std::find_if(mResources.begin(), mResources.end(),
                               [r](const std::unique_ptr<Resource>& u) {
                                   return u.get() == r;
                               });
        if (it != mResources.end()) mResources.erase(it);
    }
    return totalFreed;
}

} // namespace oird
