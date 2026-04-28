// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// runtime/resource.h — abstraction for memory-pressure-managed state.
//
// v0.7-post step F: the eviction primitive is "make N bytes available."
// Today that means tearing down the LRU model. v0.8 introduces observe
// sessions (audio.observe / vision.observe / world.observe) that share
// memory pressure with models but have richer lifecycle (pause vs.
// evict, priority bias for user-visible activity, multi-model pinning).
//
// Both shapes implement Resource. Runtime::evictForBytesLocked walks the
// registry in (priority asc, lastAccessMs asc) order, doing a pass-1
// pause() followed by pass-2 evict() if still over budget.
//
// For models, see runtime/model_resource.h — a generic adapter that
// wraps a LoadedModel with a backend-supplied tear-down lambda.
#pragma once

#include <cstdint>
#include <string>

namespace oird {

class Resource {
public:
    virtual ~Resource() = default;

    // Bytes accounted for in Budget. May change over time (sessions
    // with growing buffers, models loading additional KV cache).
    virtual int64_t residentBytes() const = 0;

    // For LRU policy. Older lastAccessMs = more evictable.
    virtual int64_t lastAccessMs() const = 0;

    // True if eviction is allowed right now. Models: skip if
    // inFlightCount > 0 or warmUntilMs > nowMs. Sessions: skip if
    // mid-cascade.
    virtual bool isEvictable(int64_t nowMs) const = 0;

    // Eviction picks LOWEST priority first; ties broken by oldest
    // lastAccessMs. Convention:
    //    0  cold one-shot model      (default for ModelResource)
    //    5  warmed one-shot model
    //   10  background session
    //   20  user-visible active session (e.g., audio.observe with
    //       speech detected in the last few seconds)
    virtual int priority() const { return 0; }

    // Pass-1 of eviction: release bytes WITHOUT killing the resource.
    // Default is no-op (models can't usefully pause). Sessions can
    // drop buffered state but stay registered. Caller holds
    // Runtime::mLock. Returns bytes freed.
    virtual int64_t pause() { return 0; }

    // Pass-2 of eviction: full teardown. Resource is dead and will be
    // unregistered after this returns. Caller holds Runtime::mLock.
    // Returns bytes freed.
    virtual int64_t evict() = 0;

    // For dumpsys / log lines.
    virtual std::string description() const = 0;

    // For Runtime::findModelResourceLocked. ModelResource overrides to
    // return its handle; non-model Resources (sessions, etc.) leave the
    // default (-1 = "not handle-keyed"). Avoids dynamic_cast (oird
    // builds with -fno-rtti).
    virtual int64_t modelHandle() const { return -1; }
};

} // namespace oird
