// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/whisper_backend.h — WhisperBackend class.
//
// v0.7-post step 3b: full migration. The 2 AIDL methods (loadWhisper,
// submitTranscribe) + 4 audio.transcribe knobs (contexts_per_model,
// acquire_timeout_ms, priority, whisper_language) + per-backend
// ModelResource teardown all live here. Mirrors LlamaBackend (step 2b2).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <aidl/com/android/server/oir/IOirWorkerCallback.h>

#include "pool/context_pool.h"   // for ContextPool::PRIO_*
#include "pool/whisper_pool.h"
#include "runtime/runtime.h"

namespace oird {

class WhisperBackend {
public:
    explicit WhisperBackend(Runtime& rt) : mRt(rt) {}

    // ---- AIDL-shaped capability methods ----

    ::ndk::ScopedAStatus loadWhisper(const std::string& modelPath, int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitTranscribe(
            int64_t modelHandle,
            const std::string& audioPath,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerCallback>& cb,
            int64_t* _aidl_return);

    // ---- Cross-backend hooks ----

    // Free this handle's whisper-specific state. Caller holds mRt.mLock.
    void eraseModel(int64_t handle);

    // ---- Knob accessors / setters ----
    int32_t audioTranscribePriority() const { return mAudioTranscribePriority; }

    bool setKnobFloat(const std::string& key, float value);
    bool setKnobString(const std::string& key, const std::string& value);

    // Per-handle whisper context pool. Public until method-body migration
    // is fully audited (no other backends touch it now that whisper bodies
    // live here). Keep public for cross-backend ModelResource teardown
    // visibility (the kitchen-sink lambda still references mWhisper.mPools
    // for non-whisper handles, harmless no-op).
    std::unordered_map<int64_t, std::unique_ptr<WhisperPool>> mPools;

private:
    Runtime& mRt;

    // audio.transcribe knobs.
    int32_t mAudioTranscribeContextsPerModel = 2;
    int32_t mAudioTranscribeAcquireTimeoutMs = 60000;
    int32_t mAudioTranscribePriority = ContextPool::PRIO_AUDIO_REALTIME;
    std::string mAudioTranscribeLanguage = "en";

    // Register a whisper-specific ModelResource (per-backend tearDown
    // replaces kitchen-sink for whisper loads). Caller holds mRt.mLock.
    void registerWhisperModelResourceLocked(int64_t handle);
};

} // namespace oird
