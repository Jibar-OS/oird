// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// runtime/model_resource.h — generic Resource adapter for handle-keyed
// LoadedModel state.
//
// Each backend's load*() registers one of these with Runtime after a
// successful load. The TearDown callback is the backend-specific cleanup
// (free pool entries, llama_free, mtmd_free, ortSession delete...). The
// central tear-down (erase from mModels, Budget accounting) is done
// inside ModelResource::evict() — backends don't need to worry about
// the registry.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "runtime/resource.h"

namespace oird {

class Runtime;        // fwd decl; full def in runtime.h
struct LoadedModel;   // fwd decl; full def in runtime.h

class ModelResource : public Resource {
public:
    using TearDown = std::function<void(int64_t handle, LoadedModel& m)>;

    ModelResource(Runtime& rt, int64_t handle, TearDown tearDown)
            : mRt(rt), mHandle(handle), mTearDown(std::move(tearDown)) {}

    int64_t residentBytes() const override;
    int64_t lastAccessMs() const override;
    bool    isEvictable(int64_t nowMs) const override;
    int     priority() const override { return mPriority; }
    int64_t evict() override;
    std::string description() const override;
    int64_t modelHandle() const override { return mHandle; }

    // load() can mark a model as warm (priority=5) once setWarm(true).
    // Aging logic in OirdService::warm() / cmd oir warm sets this.
    void setPriority(int p) { mPriority = p; }

    int64_t handle() const { return mHandle; }

private:
    Runtime&  mRt;
    int64_t   mHandle;
    TearDown  mTearDown;
    int       mPriority = 0;
};

} // namespace oird
