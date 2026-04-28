// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/vlm_backend.h — VlmBackend placeholder class.
//
// v0.7-post step 5a: VLM (vision.describe) doesn't have unique
// per-handle state today — VLM ContextPools live in mLlama.mPools
// alongside text models, and per-handle mtmd_context lives in the
// LoadedModel struct. This skeleton establishes the parallel
// pattern with LlamaBackend / WhisperBackend / OrtBackend so step 5b
// has a slot to migrate vision.describe's knobs (mVisionDescribeNCtx,
// mVisionDescribeContextsPerModel, etc.) and method bodies into.
//
// VlmBackend::eraseModel is currently a no-op because the kitchen-sink
// tear-down lambda in OirdService::registerModelResourceLocked still
// frees mtmdCtx for VLM handles. When method bodies migrate, this
// class takes over that responsibility.
#pragma once

#include <cstdint>

#include "runtime/runtime.h"

namespace oird {

class VlmBackend {
public:
    explicit VlmBackend(Runtime& /*rt*/) {}

    // No-op until step 5b migration. VLM handles share mLlama.mPools
    // (the llama context pool map) and have mtmdCtx in LoadedModel,
    // both freed elsewhere.
    void eraseModel(int64_t /*handle*/) {}
};

} // namespace oird
