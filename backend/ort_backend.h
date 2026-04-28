// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/ort_backend.h — OrtBackend class.
//
// v0.7-post step 4b: full migration. The 10 ORT-driven AIDL methods
// (loadOnnx + 7 submit methods + loadVisionEmbed + loadVad) plus all
// ORT-specific knobs and the lazy ORT env singleton live here.
//
// Capabilities served:
//   - vision.detect, vision.embed, vision.ocr  (ONNX visual models)
//   - audio.synthesize, audio.vad              (ONNX audio models)
//   - text.classify, text.rerank               (ONNX text models)
//
// VLM coupling: mImageMaxPixels is currently used by OirdService VLM
// paths too (vision.describe decodes images). Public accessor until
// VlmBackend extraction (step 5b).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <aidl/com/android/server/oir/IOirWorkerCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerVectorCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerAudioCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerBboxCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerRealtimeBooleanCallback.h>

#include "image_decode.h"  // for kDefaultMaxImagePixels
#include "pool/context_pool.h"  // for ContextPool::PRIO_*
#include "runtime/runtime.h"

namespace oird {

class OrtBackend {
public:
    explicit OrtBackend(Runtime& rt) : mRt(rt) {}

    // ---- AIDL-shaped capability methods ----

    ::ndk::ScopedAStatus loadOnnx(const std::string& modelPath,
                                   bool isDetection,
                                   int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitSynthesize(
            int64_t modelHandle,
            const std::string& text,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerAudioCallback>& cb,
            int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitClassify(
            int64_t modelHandle,
            const std::string& text,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerVectorCallback>& cb,
            int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitRerank(
            int64_t modelHandle,
            const std::string& query,
            const std::vector<std::string>& candidates,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerVectorCallback>& cb,
            int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitOcr(
            int64_t modelHandle,
            const std::string& imagePath,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerBboxCallback>& cb,
            int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitDetect(
            int64_t modelHandle,
            const std::string& imagePath,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerBboxCallback>& cb,
            int64_t* _aidl_return);
    ::ndk::ScopedAStatus loadVisionEmbed(const std::string& modelPath, int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitVisionEmbed(
            int64_t modelHandle,
            const std::string& imagePath,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerVectorCallback>& cb,
            int64_t* _aidl_return);
    ::ndk::ScopedAStatus loadVad(const std::string& modelPath, int64_t* _aidl_return);
    ::ndk::ScopedAStatus submitVad(
            int64_t modelHandle,
            const std::string& pcmPath,
            const std::shared_ptr<aidl::com::android::server::oir::IOirWorkerRealtimeBooleanCallback>& cb,
            int64_t* _aidl_return);

    // ---- Cross-backend hooks ----

    // Free this handle's ORT-specific state. Caller holds mRt.mLock.
    void eraseModel(int64_t handle);

    // ---- Knob accessors / setters ----
    int32_t audioVadPriority() const         { return mAudioVadPriority; }
    int32_t audioSynthesizePriority() const  { return mAudioSynthesizePriority; }

    // VLM coupling: vision.describe decodes images via the same
    // image.max_pixels cap. Public accessor until VlmBackend extracts.
    size_t imageMaxPixels() const { return mImageMaxPixels; }

    bool setKnobFloat(const std::string& key, float value);
    bool setKnobString(const std::string& key, const std::string& value);

    // ---- ORT shared infrastructure ----

    // Lazily create the per-process Ort::Env on first ORT load. All
    // sessions share it (ORT recommendation).
    void ensureOrtEnvLocked();
    Ort::SessionOptions makeOrtSessionOptions(bool isDetection) const;

    // ---- Public per-handle state (kitchen-sink + dumpsys access) ----

    // Per-det-handle cache for the vision.ocr recognizer ORT session +
    // vocabulary. Lazy-loaded on first submitOcr after sidecar existence
    // check; stays resident for the lifetime of the det model.
    struct OcrRec {
        Ort::Session* session = nullptr;
        std::vector<std::string> vocab;   // idx 0 = CTC blank
    };
    std::unordered_map<int64_t, OcrRec> mOcrRec;

private:
    Runtime& mRt;

    // ORT-specific knobs (alphabetical by feature).
    float   mDetectScoreThresh = 0.25f;
    float   mDetectIouThresh   = 0.45f;
    int32_t mVisionDetectInputSize = 640;
    std::string mVisionDetectFamily    = "rtdetr";
    std::string mVisionDetectNormalize = "0_to_1";

    int32_t mVisionEmbedInputSize  = 224;
    float   mVisionEmbedNormMean   = 0.5f;
    float   mVisionEmbedNormStd    = 0.5f;

    size_t mImageMaxPixels = kDefaultMaxImagePixels;

    int32_t mAudioSynthesizeSampleRate = 22050;
    float   mAudioSynthesizeLengthScale = 1.0f;
    float   mAudioSynthesizeNoiseScale  = 0.667f;
    int32_t mAudioSynthesizePriority    = ContextPool::PRIO_AUDIO_REALTIME;

    float   mVadVoiceThreshold = 0.5f;
    int32_t mVadSampleRateHz   = 16000;
    int32_t mVadWindowSamples  = 512;
    int32_t mVadContextSamples = 64;
    int32_t mAudioVadPriority  = ContextPool::PRIO_AUDIO_REALTIME;

    std::unique_ptr<Ort::Env> mOrtEnv;

    void registerOrtModelResourceLocked(int64_t handle);
};

} // namespace oird
