// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// runtime/model_resource.cpp — definitions for ModelResource.

#include "runtime/model_resource.h"

#include "runtime/runtime.h"

namespace oird {

int64_t ModelResource::residentBytes() const {
    auto it = mRt.mModels.find(mHandle);
    return (it == mRt.mModels.end()) ? 0 : it->second.sizeBytes;
}

int64_t ModelResource::lastAccessMs() const {
    auto it = mRt.mModels.find(mHandle);
    return (it == mRt.mModels.end()) ? 0 : it->second.lastAccessMs;
}

bool ModelResource::isEvictable(int64_t nowMs) const {
    auto it = mRt.mModels.find(mHandle);
    if (it == mRt.mModels.end()) return false;
    if (it->second.inFlightCount > 0) return false;
    if (it->second.warmUntilMs > nowMs) return false;
    return true;
}

int64_t ModelResource::evict() {
    auto it = mRt.mModels.find(mHandle);
    if (it == mRt.mModels.end()) return 0;
    int64_t bytes = it->second.sizeBytes;
    // Backend-specific tear-down (free pool entries, llama_free,
    // mtmd_free, ort session delete, ...).
    if (mTearDown) mTearDown(mHandle, it->second);
    // Central tear-down: drop from the model registry.
    mRt.mModels.erase(it);
    return bytes;
}

std::string ModelResource::description() const {
    auto it = mRt.mModels.find(mHandle);
    if (it == mRt.mModels.end()) {
        return "model(handle=" + std::to_string(mHandle) + ", gone)";
    }
    return "model:" + it->second.path
           + " (handle=" + std::to_string(mHandle) + ")";
}

} // namespace oird
