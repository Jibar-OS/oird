// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/whisper_backend.h — WhisperBackend class.
//
// v0.7-post step 3a (mirror of step 2b1 for whisper): the per-handle
// whisper context pool map (was OirdService::mWhisperPools) moves into
// a backend class. Method bodies (loadWhisper, submitTranscribe) stay
// on OirdService for now — same staging as LlamaBackend; the eviction
// coupling that blocked the move is resolved by step F's
// Runtime::evictForBytesLocked.
//
// Knob ownership stays on OirdService until method-body migration.
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "pool/whisper_pool.h"
#include "runtime/runtime.h"

namespace oird {

class WhisperBackend {
public:
    // Constructor takes a Runtime& for symmetry with future backend
    // classes that will use it; the field-only WhisperBackend doesn't
    // currently need the reference, so accept and discard it. Same
    // pattern as LlamaBackend.
    explicit WhisperBackend(Runtime& /*rt*/) {}

    // Per-handle whisper context pool. Keyed by modelHandle. Created
    // at loadWhisper time; destroyed on unload / LRU eviction. Public
    // until method bodies migrate (step 3b) — OirdService methods
    // currently access via mWhisper.mPools.
    std::unordered_map<int64_t, std::unique_ptr<WhisperPool>> mPools;

    // Cross-backend hook for memory-pressure eviction. Caller holds
    // the daemon's mLock. Erase on absent map keys is a no-op.
    void eraseModel(int64_t handle) { mPools.erase(handle); }
};

} // namespace oird
