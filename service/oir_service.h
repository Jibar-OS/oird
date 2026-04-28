// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// service/oir_service.h — OirdService AIDL implementation.
//
// Owns Runtime + 4 backends (LlamaBackend, WhisperBackend, OrtBackend,
// VlmBackend) and routes every AIDL method to the appropriate backend.
// All capability bodies + per-backend knobs live in backend/*.cpp files.
#pragma once

#include <android-base/logging.h>
#include <android/binder_manager.h>
#include <android/binder_process.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <aidl/com/android/server/oir/BnOirWorker.h>
#include <aidl/com/android/server/oir/IOirWorkerCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerVectorCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerAudioCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerBboxCallback.h>
#include <aidl/com/android/server/oir/IOirWorkerRealtimeBooleanCallback.h>
#include <aidl/com/android/server/oir/MemoryStats.h>

#include "backend/llama_backend.h"
#include "backend/ort_backend.h"
#include "backend/vlm_backend.h"
#include "backend/whisper_backend.h"
#include "runtime/runtime.h"

namespace oird {

using aidl::com::android::server::oir::BnOirWorker;
using aidl::com::android::server::oir::IOirWorkerCallback;
using aidl::com::android::server::oir::IOirWorkerVectorCallback;
using aidl::com::android::server::oir::IOirWorkerAudioCallback;
using aidl::com::android::server::oir::IOirWorkerBboxCallback;
using aidl::com::android::server::oir::IOirWorkerRealtimeBooleanCallback;
using aidl::com::android::server::oir::MemoryStats;

class OirdService : public BnOirWorker {
public:
    OirdService();

    ~OirdService() override;

    ::ndk::ScopedAStatus load(const std::string& modelPath, int64_t* _aidl_return) override;

    // v0.4 H4-A: load a model in embedding mode.
    // Separate code path from load() because llama_context_params differ materially
    // (embeddings=true, pooling_type=MEAN, smaller n_ctx is fine for sentence-level).
    ::ndk::ScopedAStatus loadEmbed(const std::string& modelPath, int64_t* _aidl_return) override;

    // v0.4 H4-A: run text through an embedding model; callback receives one pooled vector.
    // Synchronous in current v0.4 (no async thread) since embedding is ~1ms for MiniLM.
    ::ndk::ScopedAStatus submitEmbed(int64_t modelHandle,
                                     const std::string& text,
                                     const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                     int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus loadWhisper(const std::string& modelPath, int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus submitTranscribe(int64_t modelHandle,
                                          const std::string& audioPath,
                                          const std::shared_ptr<IOirWorkerCallback>& cb,
                                          int64_t* _aidl_return) override;

    // ========================================================================
    // ONNX Runtime integration — state as of v0.6.
    //
    // loadOnnx() is fully implemented — it creates an ORT session from an
    // on-disk .onnx model and tracks it in mRt.mModels under the same budget/LRU
    // machinery as llama and whisper. Validation fires at load time for
    // detect, synthesize, embed, and vad shapes (see validateOrtContract +
    // per-submit loadOnnx sites). Consumers can load any ONNX model through
    // this path; mismatched shapes surface MODEL_INCOMPATIBLE before the
    // first inference.
    //
    // All four submit* ONNX paths are shipping real:
    //   - submitDetect — image decode + letterbox + NMS over YOLOv8 or
    //     RT-DETR outputs; shape-validated at loadOnnx.
    //   - submitSynthesize — real Piper TTS with a permissive G2P sidecar
    //     (phonemes.json bundled per-voice). Scales + sample rate are
    //     OEM-tunable via setCapabilityFloat; shape-validated at loadOnnx
    //     (3 inputs: input/input_lengths/scales; 1 output).
    //   - submitVisionEmbed — SigLIP/CLIP-family pooled embeddings;
    //     shape-validated at loadOnnx.
    //   - submitVad — Silero VAD streaming on/off transitions;
    //     shape-validated at loadOnnx.
    //
    // Handle lifecycle + AIDL surface have been stable since v0.4.
    // ========================================================================

    ::ndk::ScopedAStatus loadOnnx(const std::string& modelPath,
                                  bool isDetection,
                                  int64_t* _aidl_return) override;

    // v0.4 H2 → v0.6 typed-error stub: Piper ORT session loads successfully
    // but the G2P (text→phoneme) frontend isn't integrated yet. Every
    // submit returns CAPABILITY_UNAVAILABLE_NO_MODEL immediately so apps
    // get a handleable error (OirCapabilityUnavailableException in SDK
    // once v0.7 ships it) instead of silent bad audio. v0.6 Phase C / G2P
    // slice lands the real impl; this whole block goes away then.
    //
    // Worker-side code W_CAPABILITY_UNSUPPORTED=12; OIRService maps to
    // Java OIRError.CAPABILITY_UNAVAILABLE_NO_MODEL=9 (same class of
    // "configured but not runnable on this device" errors).
    // v0.6 Phase C: audio.synthesize — real Piper inference path.
    //
    // Contract (Piper VITS ONNX):
    //   input tensor  "input"         int64   [1, N_phonemes]
    //   input tensor  "input_lengths" int64   [1]
    //   input tensor  "scales"        float32 [3]   (noise, length, noise_w)
    //   output tensor "output"        float32 [1, 1, n_samples]  @ 22050 Hz mono
    //
    // The model is shape-validated at loadOnnx time (see validateOrtContract).
    // G2P (grapheme→phoneme) is provided by an OEM-baked sidecar next to the
    // ONNX: `<model-path>.phonemes.json`. Format is a UTF-8 JSON object with:
    //   { "version": 1,
    //     "phoneme_ids": { "<phoneme-str>": <int>, ... },   // eSpeak IPA → model id
    //     "grapheme_map": { "<grapheme-or-word>": [<ph-id>, ...], ... } }
    // The runtime reads this once at first submitSynthesize call and caches
    // per-handle. This keeps AOSP source platform-neutral — locale-specific
    // phonetics live in the model package the OEM ships, same as whisper
    // models, vision.detect class sidecars, and VLM mmproj files.
    //
    // If the sidecar is missing, runtime returns CAPABILITY_UNAVAILABLE_NO_MODEL
    // rather than SIGSEGV or silent bad audio. This is the v0.6 "declared but
    // not runnable on this device" pattern.
    ::ndk::ScopedAStatus submitSynthesize(int64_t modelHandle,
                                          const std::string& text,
                                          const std::shared_ptr<IOirWorkerAudioCallback>& cb,
                                          int64_t* _aidl_return) override;

    // v0.6 Phase B: text.classify. Runs an ONNX classifier (encoder + logits head)
    // over the input text and returns per-label softmax scores as a Vector.
    //
    // Contract: single input tensor "input_ids" int64 [1, N_tokens],
    // optional "attention_mask" int64 [1, N_tokens]. Output "logits"
    // float32 [1, N_labels]. Tokenization requires a `<model>.tokenizer.json`
    // HuggingFace-format sidecar; runtime returns NO_MODEL if absent.
    //
    // This keeps AOSP source tokenizer-free: OEMs bake the model + its
    // tokenizer.json as a pair, same convention as vision.detect classes.
    ::ndk::ScopedAStatus submitClassify(int64_t modelHandle,
                                        const std::string& text,
                                        const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                        int64_t* _aidl_return) override;

    // v0.6 Phase B: text.rerank. Cross-encoder scoring — runs (query,
    // candidate[i]) through a reranker and emits one score per candidate.
    //
    // Contract (MS-MARCO-MiniLM style cross-encoder ONNX):
    //   input  "input_ids"      int64 [1, N_tokens]    — [CLS] q [SEP] cand [SEP]
    //   input  "attention_mask" int64 [1, N_tokens]
    //   input  "token_type_ids" int64 [1, N_tokens]    — 0 for q, 1 for cand
    //   output "logits"         float32 [1, 1]         — scalar relevance
    // Same tokenizer sidecar as submitClassify.
    ::ndk::ScopedAStatus submitRerank(int64_t modelHandle,
                                       const std::string& query,
                                       const std::vector<std::string>& candidates,
                                       const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                       int64_t* _aidl_return) override;

    // v0.6 Phase B: text.translate. Thin wrapper around submit() that forces
    // translation-leaning sampler settings (low temp, no top-k). The prompt
    // passed here has already been rewritten by OIRService into an
    // instruction-style translation request, so the llama model generates
    // the translation as plain text. Zero new runtime — reuses text.complete
    // ContextPool.
    ::ndk::ScopedAStatus submitTranslate(int64_t modelHandle,
                                          const std::string& prompt,
                                          int32_t maxTokens,
                                          const std::shared_ptr<IOirWorkerCallback>& cb,
                                          int64_t* _aidl_return) override;

    // v0.6 Phase B: vision.ocr. Detection + recognition chain.
    //
    // Contract:
    //   det  ONNX (the loaded model):  image → N regions as BoundingBoxes
    //   rec  ONNX (sidecar `<model>.rec.onnx`):  cropped region → token IDs
    //   vocab sidecar (`<model>.rec.vocab.txt`):  id → UTF-8 character
    //
    // If either sidecar is missing runtime returns CAPABILITY_UNAVAILABLE_NO_MODEL.
    // The det-only case (bboxes without text) is not useful as OCR so it
    // does not fall through to a partial result.
    ::ndk::ScopedAStatus submitOcr(int64_t modelHandle,
                                    const std::string& imagePath,
                                    const std::shared_ptr<IOirWorkerBboxCallback>& cb,
                                    int64_t* _aidl_return) override;

    // v0.4 H3: vision.detect real impl.
    // Decode → letterbox 640×640 → YOLOv8 inference → parse (1, 84, 8400)
    // → NMS → map coords back to source image → bboxes callback.
    // Default class labels = COCO-80 inline. Sidecar <model>.classes.json
    // support deferred to v0.5 (trivial std::ifstream + simple tokenizer).
    ::ndk::ScopedAStatus submitDetect(int64_t modelHandle,
                                      const std::string& imagePath,
                                      const std::shared_ptr<IOirWorkerBboxCallback>& cb,
                                      int64_t* _aidl_return) override;

    // v0.4 H4-B: vision.embed — AIDL + ORT session load + preprocess + pool
    // shipped real. SigLIP-base validated on cvd (768-dim output vector).
    ::ndk::ScopedAStatus loadVisionEmbed(const std::string& modelPath, int64_t* _aidl_return) override;

    // v0.4 H4-B: vision.embed real impl.
    // Decode → resize 224×224 (bilinear) → SigLIP normalize (mean=0.5 std=0.5)
    // → ORT Run → mean-pool if patch-tokens → L2 normalize → emit vector.
    ::ndk::ScopedAStatus submitVisionEmbed(int64_t modelHandle,
                                           const std::string& imagePath,
                                           const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                           int64_t* _aidl_return) override;

    // ========================================================================
    // v0.4 H6 — vision.describe via clip.cpp + LLaVA-family VLM.
    //
    // Two model files: CLIP mmproj (.gguf) + LLM (.gguf). Loaded together
    // as a single LoadedModel handle — both share budget, both evict as one.
    // ========================================================================

    ::ndk::ScopedAStatus loadVlm(const std::string& clipPath,
                                 const std::string& llmPath,
                                 int64_t* _aidl_return) override;

    // Describe an image. Uses LLaVA-style template:
    //   USER: <image-tokens>\n<prompt>\nASSISTANT:
    // Token streaming via cb->onToken(String, int outputIndex).
    ::ndk::ScopedAStatus submitDescribeImage(int64_t modelHandle,
                                             const std::string& imagePath,
                                             const std::string& prompt,
                                             const std::shared_ptr<IOirWorkerCallback>& cb,
                                             int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus unload(int64_t modelHandle) override;

    ::ndk::ScopedAStatus submit(int64_t modelHandle,
                                const std::string& prompt,
                                int32_t maxTokens,
                                float temperature,
                                const std::shared_ptr<IOirWorkerCallback>& callback,
                                int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus cancel(int64_t requestHandle) override;

    // ========================================================================
    // v0.5 V5: audio.vad via Silero VAD (ONNX, MIT, ~2 MB).
    //
    // Silero v4/v5 ONNX signature:
    //   inputs:  input=[1,N_samples] float  (N_samples=512 @ 16kHz = 32ms)
    //            state=[2,1,128] float      (LSTM hidden+cell state)
    //            sr=[1] int64               (16000)
    //   outputs: output=[1,1] float         (voice probability 0–1)
    //            stateN=[2,1,128] float     (updated state)
    //
    // Caller contract: submitVad consumes a PCM16 LE 16kHz mono file and emits
    // one onState(bool, timestampMs) per 32ms window. Threshold 0.5.
    // Live-mic streaming is an app-side composition — client feeds microphone
    // PCM through the same shape. Shape frozen in v0.5.
    // ========================================================================
    ::ndk::ScopedAStatus loadVad(const std::string& modelPath,
                                 int64_t* _aidl_return) override;

    ::ndk::ScopedAStatus submitVad(int64_t modelHandle,
                                   const std::string& pcmPath,
                                   const std::shared_ptr<IOirWorkerRealtimeBooleanCallback>& cb,
                                   int64_t* _aidl_return) override;

    // v0.4 S2-B: configure budget + warm TTL. Called once by OIRService at attachWorker.
    ::ndk::ScopedAStatus setConfig(int32_t memoryBudgetMb, int32_t warmTtlSeconds) override;

    // v0.5 V7: per-capability tuning via (key, float-value) pushed by OIRService
    // at attachWorker. Unknown keys are logged and ignored so OEMs can add new
    // knobs in their config files ahead of runtime support without errors.
    ::ndk::ScopedAStatus setCapabilityFloat(const std::string& key, float value) override;

    // v0.5 V7 full: string-valued counterpart. Same routing pattern; unknown
    // keys logged and ignored.
    ::ndk::ScopedAStatus setCapabilityString(const std::string& key,
                                             const std::string& value) override;

    // v0.4 S3: mark a model as warm — unevictable for warm_ttl_seconds past this call.
    ::ndk::ScopedAStatus warm(int64_t modelHandle) override;

    // v0.6.3: per-handle runtime snapshot. Pairs with getMemoryStats
    // (model-level RSS totals) — this one exposes the live pool state
    // ContextPool + WhisperPool expose as getters. Returned as a
    // newline-separated TSV so `cmd oir memory` can print verbatim
    // without needing a new parcelable layout.
    ::ndk::ScopedAStatus dumpRuntimeStats(std::string* _aidl_return) override;

    // v0.6.3: worker-side file stat fallback used by OIRService when
    // system_server's SELinux scope can't `getattr` an `oir_model_*_file`
    // label directly. Rechecked through oird's domain which does have
    // the read rule platform-wide. Path resolution is native stat();
    // symlinks resolve; permission-denied maps to false.
    ::ndk::ScopedAStatus fileIsReadable(const std::string& path, bool* _aidl_return) override;

    // v0.4 S2: snapshot of worker memory state for cmd oir dumpsys memory.
    ::ndk::ScopedAStatus getMemoryStats(MemoryStats* _aidl_return) override;

private:
    // Cross-cutting daemon state (mutex, model handle table, budget,
    // scheduler, load-dedup registry, in-flight tracking, warm TTL).
    // Each backend holds a Runtime& and reaches shared state through it.
    Runtime mRt;

    // Resolve the scheduler priority for a capability name. Reads the
    // per-capability priority knob from the owning backend; unknown caps
    // default to PRIO_NORMAL.
    int32_t priorityForCapability(const std::string& cap);

    // The 4 backends. Each owns its capability bodies, knobs, and
    // per-handle state; OirdService routes AIDL calls to one of them.
    // VlmBackend takes mLlama& because VLM contexts live in the llama
    // pool (mtmd_context wraps a llama_context).
    LlamaBackend   mLlama  {mRt};
    WhisperBackend mWhisper{mRt};
    OrtBackend     mOrt    {mRt};
    VlmBackend     mVlm    {mRt, mLlama};
};

} // namespace oird
