// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/llama_backend.h — LlamaBackend class.
//
// v0.7-post step 2b1 (minimal): per-handle llama context pool map (was
// OirdService::mLlamaPools) moves into a backend class. Method bodies
// stay on OirdService for now because their eviction loops touch
// non-llama state (mWhisperPools, mOcrRec); a follow-up step will
// extract a Runtime::evictForBytes coordinator and let LlamaBackend
// own its methods too.
//
// Knob ownership stays on OirdService until a future step.
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "pool/context_pool.h"
#include "runtime/runtime.h"

namespace oird {

class LlamaBackend {
public:
    // Constructor takes a Runtime& for symmetry with future backend
    // classes that will use it; currently the field-only LlamaBackend
    // doesn't actually need the reference, so we accept and discard it.
    explicit LlamaBackend(Runtime& /*rt*/) {}

    // Per-handle llama context pool. Keyed by modelHandle. Created at
    // load time; destroyed on unload / LRU eviction. Public until the
    // VLM extraction (step 5) — VlmBackend currently registers VLM
    // ContextPools here too.
    std::unordered_map<int64_t, std::unique_ptr<ContextPool>> mPools;

    // Cross-backend hook: free this handle's llama-specific pool entry.
    // Caller must hold the daemon's mLock. Erase on absent map keys is
    // a no-op.
    void eraseModel(int64_t handle) { mPools.erase(handle); }
};

} // namespace oird
