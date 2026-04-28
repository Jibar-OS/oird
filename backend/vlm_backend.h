// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/vlm_backend.h — VlmBackend class.
//
// v0.7-post step 5b: full VLM (vision.describe) extraction. The 2 AIDL
// methods (loadVlm, submitDescribeImage) + 6 vision.describe knobs +
// per-backend ModelResource teardown all live here.
//
// VLM is unique in that it uses llama context pools (mtmd_context wraps
// a llama_context internally for the text-side of the cascade). The
// VLM pool entries live in mLlama.mPools alongside text models.
// VlmBackend therefore holds a LlamaBackend& reference in addition to
// the standard Runtime&.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <aidl/com/android/server/oir/IOirWorkerCallback.h>

#include "backend/llama_backend.h"
#include "pool/context_pool.h"  // for ContextPool::PRIO_*
#include "runtime/runtime.h"

namespace oird {

class VlmBackend {
public:
    // Takes Runtime& for cross-cutting state and LlamaBackend& for
    // shared context pool storage (VLM pool entries live in
    // mLlama.mPools because they're built on llama_context).
    VlmBackend(Runtime& rt, LlamaBackend& llama) : mRt(rt), mLlama(llama) {}

    // ---- AIDL-shaped capability methods ----

    ::ndk::ScopedAStatus loadVlm(const std::string& clipPath,
                                  const std::string& llmPath,
                                  int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitDescribeImage(
            int64_t modelHandle,
            const std::string& imagePath,
            const std::string& prompt,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerCallback>& cb,
            int64_t* _aidl_return);

    // ---- Cross-backend hooks ----

    // Free this handle's VLM-specific state (mtmd_context). Caller holds
    // mRt.mLock. The shared pool entry in mLlama.mPools is freed by
    // LlamaBackend::eraseModel.
    void eraseModel(int64_t handle);

    // ---- Knob accessors / setters ----
    int32_t visionDescribePriority() const { return mVisionDescribePriority; }

    bool setKnobFloat(const std::string& key, float value);

private:
    Runtime&      mRt;
    LlamaBackend& mLlama;

    // vision.describe knobs.
    int32_t mVisionDescribeNCtx              = 4096;
    int32_t mVisionDescribeNBatch            = 2048;
    int32_t mVisionDescribeMaxTokens         = 256;
    int32_t mVisionDescribeContextsPerModel  = 1;  // VLMs are 4GB+
    int32_t mVisionDescribeAcquireTimeoutMs = 60000;
    int32_t mVisionDescribePriority          = ContextPool::PRIO_NORMAL;

    void registerVlmModelResourceLocked(int64_t handle);
};

} // namespace oird
