// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/llama_backend.h — LlamaBackend class.
//
// v0.7-post step 2b2: full backend extraction. The 6 llama-driven AIDL
// implementations (load, loadEmbed, submit, submitEmbed, submitTranslate,
// + the runInference worker) live here, along with all 12 llama knobs
// (text.complete.* + text.embed.* + sampling defaults + llama.batch_size).
// OirdService keeps the AIDL stubs as 1-line wrappers calling into
// mLlama; setCapabilityFloat dispatches llama keys via mLlama.setKnobFloat;
// priorityForCapability uses mLlama.text*Priority() accessors.
//
// Per-handle context pool map (mPools) + per-handle ModelResource
// teardown is owned here too. Llama-specific teardown frees the pool
// entry + llama_free + llama_model_free for non-VLM handles.
//
// VLM coupling: VLMs (vision.describe) also use ContextPool today, and
// pool entries live in this map. Until VlmBackend extraction (step 5b),
// the OirdService VLM paths read/write mLlama.mPools + use
// mLlama.mLlamaBatchSize directly via the public accessors.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <aidl/com/android/server/oir/IOirWorkerCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerVectorCallback.h>

#include "pool/context_pool.h"
#include "runtime/runtime.h"

namespace oird {

class LlamaBackend {
public:
    explicit LlamaBackend(Runtime& rt) : mRt(rt) {}

    // ---- AIDL-shaped capability methods (called from OirdService wrappers) ----

    ::ndk::ScopedAStatus load(const std::string& modelPath, int64_t* _aidl_return);
    ::ndk::ScopedAStatus loadEmbed(const std::string& modelPath, int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitEmbed(
            int64_t modelHandle,
            const std::string& text,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerVectorCallback>& cb,
            int64_t* _aidl_return);
    ::ndk::ScopedAStatus submit(
            int64_t modelHandle,
            const std::string& prompt,
            int32_t maxTokens,
            float temperature,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerCallback>& callback,
            int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitTranslate(
            int64_t modelHandle,
            const std::string& prompt,
            int32_t maxTokens,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerCallback>& cb,
            int64_t* _aidl_return);

    // ---- Cross-backend hooks ----

    // Free this handle's llama-specific state. Caller holds mRt.mLock.
    // Called by the per-llama-handle ModelResource teardown lambda.
    void eraseModel(int64_t handle);

    // ---- Knob accessors (used by OirdService::priorityForCapability) ----
    int32_t textCompletePriority() const { return mTextCompletePriority; }
    int32_t textEmbedPriority() const { return mTextEmbedPriority; }

    // Returns true if `key` was a llama knob (handled). Caller must hold
    // mRt.mLock. Used by OirdService::setCapabilityFloat to dispatch knobs
    // by capability prefix.
    bool setKnobFloat(const std::string& key, float value);

    // ---- VLM coupling (until step 5b extracts VlmBackend) ----
    // VLMs share the llama context pool map. Public field is the
    // explicit, temporary coupling.
    std::unordered_map<int64_t, std::unique_ptr<ContextPool>> mPools;

    // VLM also uses llama batches. Public for OirdService::loadVlm /
    // submitDescribeImage to read.
    int32_t mLlamaBatchSize = 512;

    // VLM-callable accessors for shared knobs (until VLM extracts).
    int32_t textCompleteContextsPerModel() const { return mTextCompleteContextsPerModel; }
    int32_t textCompleteAcquireTimeoutMs() const { return mTextCompleteAcquireTimeoutMs; }

private:
    Runtime& mRt;

    // Llama-specific knobs (private; access via accessors / setKnobFloat).
    int32_t mTextCompleteNCtx              = 2048;
    int32_t mTextCompleteMaxTokens         = 256;
    int32_t mTextEmbedNCtx                 = 512;
    int32_t mTextCompleteContextsPerModel  = 4;
    int32_t mTextEmbedContextsPerModel     = 2;
    int32_t mTextCompleteAcquireTimeoutMs  = 30000;
    int32_t mTextEmbedAcquireTimeoutMs     = 10000;
    int32_t mTextCompletePriority          = ContextPool::PRIO_NORMAL;
    int32_t mTextEmbedPriority             = ContextPool::PRIO_NORMAL;
    float   mTextCompleteTemperatureDefault = 0.7f;
    float   mTextCompleteTopP              = 0.9f;

    // Worker — runs inside Scheduler thread via submit's enqueue.
    void runInference(int64_t modelHandle,
                      int64_t handle,
                      std::string prompt,
                      int32_t maxTokens,
                      float temperature,
                      std::shared_ptr<aidl::com::android::server::oir::IOirWorkerCallback> cb,
                      std::shared_ptr<std::atomic_bool> cancelled,
                      std::shared_ptr<InFlightGuard> guard);

    // Register a llama-specific ModelResource (per-backend tearDown
    // replaces kitchen-sink for llama loads). Caller holds mRt.mLock.
    void registerLlamaModelResourceLocked(int64_t handle);
};

} // namespace oird
