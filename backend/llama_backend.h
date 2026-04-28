// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/llama_backend.h — LlamaBackend class. Owns text.complete,
// text.embed, text.translate plus the per-handle ContextPool map.
//
// VLM coupling: VlmBackend stores its mtmd-wrapped llama contexts in
// mPools alongside text models, and uses mLlamaBatchSize for prefill.
// Both are public so VlmBackend can reach them directly.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <llama.h>

#include <aidl/com/android/server/oir/IOirWorkerCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerVectorCallback.h>

#include "pool/context_pool.h"
#include "runtime/runtime.h"

namespace oird {

// v0.6 Phase A: estimate KV cache bytes per llama_context so the
// Budget can account pool overhead. Formula:
//   n_ctx × n_layer × 2 (K+V) × n_kv_head × head_dim × bytes_per_elem
// Elements are fp16 by default in llama.cpp (2 bytes).
inline int64_t estimateKvBytesPerContext(llama_model* m, int32_t n_ctx) {
    if (!m || n_ctx <= 0) return 0;
    int64_t n_layer = llama_model_n_layer(m);
    int64_t n_embd  = llama_model_n_embd(m);
    int64_t n_head  = llama_model_n_head(m);
    int64_t n_head_kv = llama_model_n_head_kv(m);
    if (n_head_kv <= 0) n_head_kv = n_head;
    if (n_head <= 0 || n_layer <= 0 || n_embd <= 0) return 0;
    int64_t head_dim = n_embd / n_head;
    return (int64_t)n_ctx * n_layer * 2 * n_head_kv * head_dim * 2;
}

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
