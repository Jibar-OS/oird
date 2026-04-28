// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/ort_backend.h — OrtBackend class.
//
// v0.7-post step 4a: per-handle OCR-recognizer cache (was
// OirdService::mOcrRec) moves into a backend class. Method bodies
// (loadOnnx, submit*Synthesize/Classify/Rerank/Ocr/Detect/...) and
// other ORT-shared state (mOrtEnv, per-capability knobs) stay on
// OirdService until step 4b method-body migration.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "runtime/runtime.h"

namespace oird {

class OrtBackend {
public:
    explicit OrtBackend(Runtime& /*rt*/) {}

    // v0.6 Phase B: per-det-handle cache for the vision.ocr recognizer
    // ORT session + vocabulary. Lazy-loaded on first submitOcr after
    // sidecar existence check; stays resident for the lifetime of the
    // det model. Cleared when the det model evicts.
    struct OcrRec {
        Ort::Session* session = nullptr;
        std::vector<std::string> vocab;   // idx 0 = CTC blank
    };
    std::unordered_map<int64_t, OcrRec> mOcrRec;

    // Cross-backend hook for memory-pressure eviction. Caller holds
    // the daemon's mLock. Erase on absent map keys is a no-op.
    void eraseModel(int64_t handle) {
        auto it = mOcrRec.find(handle);
        if (it != mOcrRec.end()) {
            delete it->second.session;
            mOcrRec.erase(it);
        }
    }
};

} // namespace oird
