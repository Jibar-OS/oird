// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/ort.cpp — out-of-class definitions for OirdService methods
// that drive the ONNX Runtime backend (vision.detect, vision.embed,
// vision.ocr, audio.synthesize, audio.vad, text.classify, text.rerank).
//
// v0.7 step 7d: extracted from service/oir_service.h. No semantic change.

#include "service/oir_service.h"

namespace oird {

::ndk::ScopedAStatus OirdService::loadOnnx(const std::string& modelPath,
                              bool isDetection,
                              int64_t* _aidl_return) {
    // v0.6.9: mRt.mLock shrunk around slow ctor. This method had a latent
    // self-deadlock before: line ~2041 re-locked mRt.mLock inside the
    // detection branch while already holding the outer lock_guard.
    // std::mutex is non-recursive → futex_wait hang. The refactor
    // snapshots mVisionDetectInputSize under the initial lock and
    // releases before Ort::Session ctor.
    const std::string key = std::string(isDetection ? "onnx-det:" : "onnx-synth:") + modelPath;
    std::unique_lock<std::mutex> lk(mRt.mLock);
    for (auto& [h, m] : mRt.mModels) {
        if (m.path == modelPath && m.isOnnx && m.onnxIsDetection == isDetection) {
            LOG(INFO) << "oird: onnx model already loaded path=" << modelPath << " handle=" << h;
            *_aidl_return = h;
            return ::ndk::ScopedAStatus::ok();
        }
    }

    auto claim = mRt.mLoadRegistry.claim(lk, key);
    if (claim.waited) {
        if (claim.waited->errCode != 0) {
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    claim.waited->errCode, claim.waited->errMsg.c_str());
        }
        *_aidl_return = claim.waited->handle;
        return ::ndk::ScopedAStatus::ok();
    }
    auto slot = claim.slot;

    const int64_t newSize = fileSizeBytes(modelPath);
    if (mRt.mBudget.budgetMb() > 0 && !mRt.mBudget.fitsAfter(newSize)) {
        int64_t needed = (mRt.mBudget.totalBytes() + newSize) - mRt.mBudget.budgetBytes();
        mRt.evictForBytesLocked(needed);
        if (!mRt.mBudget.fitsAfter(newSize)) {
            const std::string msg = "budget exceeded; nothing evictable";
            mRt.mLoadRegistry.publish(lk, key, slot, 0, W_INSUFFICIENT_MEMORY, msg);
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_INSUFFICIENT_MEMORY, msg.c_str());
        }
    }

    // Snapshot tunables under lock.
    const int32_t kIn = mVisionDetectInputSize;
    mRt.mBudget.addResident(newSize);

    lk.unlock();

    // --- slow ctor: Ort env + Session, mRt.mLock NOT held ---
    ensureOrtEnv();
    Ort::SessionOptions so = makeOrtSessionOptions(isDetection);
    Ort::Session* session = nullptr;
    try {
        session = new Ort::Session(*mOrtEnv, modelPath.c_str(), so);
    } catch (const Ort::Exception& e) {
        LOG(ERROR) << "oird: Ort::Session failed for " << modelPath << ": " << e.what();
        const std::string msg = std::string("onnx load failed: ") + e.what();
        lk.lock();
        mRt.mBudget.subResident(newSize);
        mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, msg);
        return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                W_MODEL_ERROR, msg.c_str());
    }
    // Validate the ONNX shape contract — see above for per-kind rules.
    if (isDetection) {
        std::vector<std::vector<int64_t>> inShapes = {
            {-1, 3, kIn, kIn},  // batch wildcard, spatial fixed
        };
        std::string err = validateOrtContract(session, 1, inShapes, "vision.detect");
        if (!err.empty()) {
            LOG(ERROR) << "oird: " << err;
            delete session;
            lk.lock();
            mRt.mBudget.subResident(newSize);
            mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_INCOMPATIBLE, err);
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_INCOMPATIBLE, err.c_str());
        }
    } else {
        // Infer "this is Piper" from `!isDetection` + input-name sniff.
        // Other single-Session ONNX (classify/rerank) skip Piper's
        // 3-input contract so they don't get wrongly rejected.
        bool looksLikePiper =
            session->GetInputCount() == 3
            && session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get() == std::string("input")
            && session->GetInputNameAllocated(1, Ort::AllocatorWithDefaultOptions()).get() == std::string("input_lengths")
            && session->GetInputNameAllocated(2, Ort::AllocatorWithDefaultOptions()).get() == std::string("scales");
        if (looksLikePiper) {
            std::vector<std::vector<int64_t>> inShapes = {
                {1, -1},  // input: batch 1, phLen dynamic
                {1},      // input_lengths
                {3},      // scales
            };
            std::string err = validateOrtContract(session, 3, inShapes, "audio.synthesize");
            if (!err.empty()) {
                LOG(ERROR) << "oird: " << err;
                delete session;
                lk.lock();
                mRt.mBudget.subResident(newSize);
                mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_INCOMPATIBLE, err);
                return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                        W_MODEL_INCOMPATIBLE, err.c_str());
            }
        }
    }

    // Per-handle detect labels are a best-effort sidecar read (file I/O
    // only, not held locked).
    std::vector<std::string> detectLabels;
    if (isDetection) {
        detectLabels = readDetectClassLabels(modelPath);
    }

    lk.lock();

    const int64_t handle = mRt.mNextModelHandle++;
    const int64_t now = currentTimeMs();
    LoadedModel lm;
    lm.ortSession = session;
    lm.handle = handle;
    lm.path = modelPath;
    lm.sizeBytes = newSize;
    lm.loadTimestampMs = now;
    lm.lastAccessMs = now;
    lm.isOnnx = true;
    lm.onnxIsDetection = isDetection;
    if (isDetection) {
        lm.detectClassLabels = std::move(detectLabels);
    }
    mRt.mModels[handle] = std::move(lm);
    registerModelResourceLocked(handle);

    mRt.mLoadRegistry.publish(lk, key, slot, handle, 0, "");

    *_aidl_return = handle;
    LOG(INFO) << "oird: onnx model loaded handle=" << handle << " path=" << modelPath
              << " kind=" << (isDetection ? "detect" : "synth")
              << " size=" << (newSize >> 20) << "MB"
              << " resident=" << (mRt.mBudget.totalBytes() >> 20) << "/" << mRt.mBudget.budgetMb() << "MB";
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::submitSynthesize(int64_t modelHandle,
                                      const std::string& text,
                                      const std::shared_ptr<IOirWorkerAudioCallback>& cb,
                                      int64_t* _aidl_return) {
    Ort::Session* session = nullptr;
    std::string sidecarPath;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isOnnx || it->second.onnxIsDetection) {
            cb->onError(W_INVALID_INPUT, "handle not an onnx synthesis model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        session     = it->second.ortSession;
        sidecarPath = it->second.path + ".phonemes.json";
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue on the cross-backend scheduler at audio-realtime
    // priority (matches audio.transcribe / audio.vad). Synthesize's
    // wall time is usually short, but routing through the scheduler
    // means a text.complete backlog doesn't delay TTS.
    mRt.mScheduler->enqueue(priorityForCapability("audio.synthesize"),
        [this, modelHandle, text, cb, session, sidecarPath, guard]() {
            // v0.6.8: terminal cb (onComplete/onError) fires AFTER
            // releaseInflight. Streaming onChunk stays inline because
            // PCM is produced incrementally; if one onChunk binder call
            // stalls, the in-flight ref is the only resource held —
            // that's the minimum achievable for a streaming shape.
            std::function<void()> terminal;
            size_t nSamples = 0;
            size_t phCount = 0;
            int64_t totalMs = 0;
            {
                if (text.empty()) {
                    terminal = [cb]() { cb->onError(W_INVALID_INPUT, "text is empty"); };
                    goto done;
                }

                PhonemeMap phonemes;
                if (!loadPhonemeSidecar(sidecarPath, phonemes)) {
                    std::string msg = "no G2P sidecar at " + sidecarPath
                            + " — OEM must bake phonemes.json next to the Piper model";
                    terminal = [cb, msg]() {
                        cb->onError(W_CAPABILITY_UNAVAILABLE_NO_MODEL, msg.c_str());
                    };
                    goto done;
                }

                std::vector<int64_t> phIds = graphemesToPhonemeIds(text, phonemes);
                if (phIds.empty()) {
                    terminal = [cb]() {
                        cb->onError(W_INVALID_INPUT, "G2P produced empty phoneme sequence");
                    };
                    goto done;
                }
                phCount = phIds.size();

                Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(
                        OrtArenaAllocator, OrtMemTypeDefault);
                const int64_t phLen = (int64_t)phIds.size();
                std::array<int64_t, 2> inputShape{1, phLen};
                std::array<int64_t, 1> lenShape{1};
                std::array<int64_t, 1> scalesShape{3};
                std::array<int64_t, 1> lenData{phLen};
                // Piper scales[] is [noise_scale, length_scale, noise_w].
                // First two are OEM-tunable via audio.synthesize.noise_scale
                // and audio.synthesize.length_scale (applied through
                // setCapabilityFloat). noise_w has no knob yet — keeping
                // the standard Piper default of 0.8.
                std::array<float, 3> scales{
                    mAudioSynthesizeNoiseScale,
                    mAudioSynthesizeLengthScale,
                    0.8f,
                };

                auto inputT = Ort::Value::CreateTensor<int64_t>(
                        meminfo, phIds.data(), phIds.size(),
                        inputShape.data(), inputShape.size());
                auto lenT = Ort::Value::CreateTensor<int64_t>(
                        meminfo, lenData.data(), lenData.size(),
                        lenShape.data(), lenShape.size());
                auto scalesT = Ort::Value::CreateTensor<float>(
                        meminfo, scales.data(), scales.size(),
                        scalesShape.data(), scalesShape.size());

                const char* inputNames[]  = {"input", "input_lengths", "scales"};
                const char* outputNames[] = {"output"};
                Ort::Value inputs[] = {std::move(inputT), std::move(lenT), std::move(scalesT)};

                std::vector<Ort::Value> outputs;
                try {
                    outputs = session->Run(Ort::RunOptions{nullptr},
                                            inputNames, inputs, 3,
                                            outputNames, 1);
                } catch (const std::exception& e) {
                    std::string msg = std::string("Piper ORT Run failed: ") + e.what();
                    terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                    goto done;
                }
                if (outputs.empty()) {
                    terminal = [cb]() { cb->onError(W_MODEL_ERROR, "Piper produced no output"); };
                    goto done;
                }

                const float* pcmF32 = outputs[0].GetTensorData<float>();
                auto outInfo = outputs[0].GetTensorTypeAndShapeInfo();
                nSamples = outInfo.GetElementCount();
                // OEM-tunable via audio.synthesize.sample_rate_hz. Voices
                // ship at fixed rates (Piper en-US/lessac is 22050) — this
                // knob is for OEMs bundling non-default voices.
                const int32_t kSampleRateHz = mAudioSynthesizeSampleRate;
                constexpr int32_t kChannels = 1;
                constexpr int32_t kEncodingPcmFloat = 4;

                const size_t kChunkSamples = (size_t)kSampleRateHz / 10;
                totalMs = (int64_t)((double)nSamples / (double)kSampleRateHz * 1000.0);
                size_t emitted = 0;
                while (emitted < nSamples) {
                    const size_t remaining = nSamples - emitted;
                    const size_t thisChunk = std::min(kChunkSamples, remaining);
                    const bool last = (emitted + thisChunk == nSamples);
                    std::vector<uint8_t> pcmBytes(thisChunk * sizeof(float));
                    std::memcpy(pcmBytes.data(), pcmF32 + emitted, pcmBytes.size());
                    cb->onChunk(pcmBytes, kSampleRateHz, kChannels, kEncodingPcmFloat, last);
                    emitted += thisChunk;
                }
                terminal = [cb, totalMs]() { cb->onComplete((int32_t)totalMs); };
            }
        done:
            guard->release();  // explicit early release; matches v0.6.8 ordering.
            if (terminal) terminal();

            LOG(INFO) << "oird: submitSynthesize handle=" << modelHandle
                      << " text.len=" << text.size()
                      << " phonemes=" << phCount
                      << " samples=" << nSamples
                      << " ms=" << totalMs;
        });
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::submitClassify(int64_t modelHandle,
                                    const std::string& text,
                                    const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                    int64_t* _aidl_return) {
    Ort::Session* session = nullptr;
    std::string sidecarPath;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isOnnx || it->second.onnxIsDetection) {
            cb->onError(W_INVALID_INPUT, "handle not an onnx text-classifier model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        session     = it->second.ortSession;
        sidecarPath = it->second.path + ".tokenizer.json";
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue on the cross-backend scheduler at text-normal
    // priority. ORT Run() is thread-safe so no pool needed — the
    // scheduler worker runs it directly. mRt.mLock never held across Run().
    mRt.mScheduler->enqueue(priorityForCapability("text.classify"),
        [modelHandle, text, cb, session, sidecarPath, guard]() {
            // v0.6.8: terminal cb fires after releaseInflight.
            std::function<void()> terminal;
            size_t nTokens = 0;
            size_t nLabels = 0;
            {
                if (text.empty()) {
                    terminal = [cb]() { cb->onError(W_INVALID_INPUT, "text is empty"); };
                    goto done;
                }

                HfTokenizer tok;
                if (!loadHfTokenizerSidecar(sidecarPath, tok)) {
                    std::string msg = "no tokenizer sidecar at " + sidecarPath
                            + " — OEM must bake tokenizer.json next to the classifier";
                    terminal = [cb, msg]() {
                        cb->onError(W_CAPABILITY_UNAVAILABLE_NO_MODEL, msg.c_str());
                    };
                    goto done;
                }

                std::vector<int64_t> inputIds     = tok.encode(text);
                std::vector<int64_t> attentionMask(inputIds.size(), 1);
                const int64_t nTok = (int64_t)inputIds.size();
                nTokens = inputIds.size();

                Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(
                        OrtArenaAllocator, OrtMemTypeDefault);
                std::array<int64_t, 2> shape{1, nTok};
                auto idsT = Ort::Value::CreateTensor<int64_t>(
                        meminfo, inputIds.data(), inputIds.size(),
                        shape.data(), shape.size());
                auto maskT = Ort::Value::CreateTensor<int64_t>(
                        meminfo, attentionMask.data(), attentionMask.size(),
                        shape.data(), shape.size());

                const char* inputNames[]  = {"input_ids", "attention_mask"};
                const char* outputNames[] = {"logits"};
                Ort::Value inputs[] = {std::move(idsT), std::move(maskT)};

                std::vector<Ort::Value> outputs;
                try {
                    outputs = session->Run(Ort::RunOptions{nullptr},
                                            inputNames, inputs, 2,
                                            outputNames, 1);
                } catch (const std::exception& e) {
                    std::string msg = std::string("classifier ORT Run failed: ") + e.what();
                    terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                    goto done;
                }

                const float* logits = outputs[0].GetTensorData<float>();
                nLabels = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
                std::vector<float> scores(nLabels);
                float maxLogit = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < nLabels; ++i) maxLogit = std::max(maxLogit, logits[i]);
                float sumExp = 0.f;
                for (size_t i = 0; i < nLabels; ++i) { scores[i] = std::exp(logits[i] - maxLogit); sumExp += scores[i]; }
                for (size_t i = 0; i < nLabels; ++i) scores[i] /= sumExp;

                terminal = [cb, scores = std::move(scores)]() { cb->onVector(scores); };
            }
        done:
            guard->release();  // explicit early release; matches v0.6.8 ordering.
            if (terminal) terminal();

            LOG(INFO) << "oird: submitClassify handle=" << modelHandle
                      << " text.len=" << text.size()
                      << " tokens=" << nTokens
                      << " labels=" << nLabels;
        });
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::submitRerank(int64_t modelHandle,
                                   const std::string& query,
                                   const std::vector<std::string>& candidates,
                                   const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                   int64_t* _aidl_return) {
    Ort::Session* session = nullptr;
    std::string sidecarPath;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isOnnx || it->second.onnxIsDetection) {
            cb->onError(W_INVALID_INPUT, "handle not an onnx reranker model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        session     = it->second.ortSession;
        sidecarPath = it->second.path + ".tokenizer.json";
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue. Rerank loops Run() once per candidate so it
    // can be chunky; scheduler dispatch keeps the binder thread free.
    mRt.mScheduler->enqueue(priorityForCapability("text.rerank"),
        [modelHandle, query, candidates, cb, session, sidecarPath, guard]() {
            // v0.6.8: terminal cb deferred past releaseInflight.
            std::function<void()> terminal;
            {
                if (query.empty() || candidates.empty()) {
                    terminal = [cb]() { cb->onError(W_INVALID_INPUT, "query/candidates empty"); };
                    goto done;
                }

                HfTokenizer tok;
                if (!loadHfTokenizerSidecar(sidecarPath, tok)) {
                    std::string msg = "no tokenizer sidecar at " + sidecarPath
                            + " — OEM must bake tokenizer.json next to the reranker";
                    terminal = [cb, msg]() {
                        cb->onError(W_CAPABILITY_UNAVAILABLE_NO_MODEL, msg.c_str());
                    };
                    goto done;
                }

                std::vector<float> scores;
                scores.reserve(candidates.size());
                bool runFailed = false;
                std::string runErr;

                for (const auto& cand : candidates) {
                    std::vector<int64_t> inputIds     = tok.encodePair(query, cand);
                    std::vector<int64_t> typeIds      = tok.typeIdsForPair(query, cand);
                    std::vector<int64_t> attentionMask(inputIds.size(), 1);
                    const int64_t nTok = (int64_t)inputIds.size();

                    Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(
                            OrtArenaAllocator, OrtMemTypeDefault);
                    std::array<int64_t, 2> shape{1, nTok};
                    auto idsT  = Ort::Value::CreateTensor<int64_t>(
                            meminfo, inputIds.data(), inputIds.size(),
                            shape.data(), shape.size());
                    auto maskT = Ort::Value::CreateTensor<int64_t>(
                            meminfo, attentionMask.data(), attentionMask.size(),
                            shape.data(), shape.size());
                    auto typeT = Ort::Value::CreateTensor<int64_t>(
                            meminfo, typeIds.data(), typeIds.size(),
                            shape.data(), shape.size());

                    const char* inputNames[]  = {"input_ids", "attention_mask", "token_type_ids"};
                    const char* outputNames[] = {"logits"};
                    Ort::Value inputs[] = {std::move(idsT), std::move(maskT), std::move(typeT)};

                    std::vector<Ort::Value> outputs;
                    try {
                        outputs = session->Run(Ort::RunOptions{nullptr},
                                                inputNames, inputs, 3,
                                                outputNames, 1);
                    } catch (const std::exception& e) {
                        runErr = std::string("reranker ORT Run failed: ") + e.what();
                        runFailed = true;
                        break;
                    }
                    const float* logits = outputs[0].GetTensorData<float>();
                    scores.push_back(logits[0]);
                }
                if (runFailed) {
                    terminal = [cb, runErr]() { cb->onError(W_MODEL_ERROR, runErr.c_str()); };
                    goto done;
                }

                terminal = [cb, scores = std::move(scores)]() { cb->onVector(scores); };
            }
        done:
            guard->release();  // explicit early release; matches v0.6.8 ordering.
            if (terminal) terminal();

            LOG(INFO) << "oird: submitRerank handle=" << modelHandle
                      << " query.len=" << query.size()
                      << " candidates=" << candidates.size();
        });
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::submitOcr(int64_t modelHandle,
                                const std::string& imagePath,
                                const std::shared_ptr<IOirWorkerBboxCallback>& cb,
                                int64_t* _aidl_return) {
    Ort::Session* detSession = nullptr;
    std::string basePath;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isOnnx || !it->second.onnxIsDetection) {
            cb->onError(W_INVALID_INPUT, "handle not an onnx OCR-detection model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        detSession = it->second.ortSession;
        basePath   = it->second.path;
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue on scheduler.
    mRt.mScheduler->enqueue(priorityForCapability("vision.ocr"),
        [this, modelHandle, imagePath, cb, detSession, basePath, guard]() {
    // v0.6.8: terminal cb deferred past releaseInflight.
    std::function<void()> terminal;
    size_t candCount = 0, keptCount = 0;
    int imgW = 0, imgH = 0;
    {
    // Require both sidecars up-front; partial OCR isn't meaningful.
    const std::string recPath   = basePath + ".rec.onnx";
    const std::string vocabPath = basePath + ".rec.vocab.txt";
    if (!fileExists(recPath) || !fileExists(vocabPath)) {
        std::string msg = "OCR requires det+rec+vocab triplet; missing "
                          + (fileExists(recPath) ? vocabPath : recPath);
        terminal = [cb, msg]() {
            cb->onError(W_CAPABILITY_UNAVAILABLE_NO_MODEL, msg.c_str());
        };
        goto done;
    }

    // Lazy-load rec session + vocab (first submitOcr for this handle).
    // Cached in mOcrRec; released when the det model evicts.
    Ort::Session* recSession = nullptr;
    std::vector<std::string> vocab;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto oit = mOcrRec.find(modelHandle);
        if (oit == mOcrRec.end()) {
            // Load rec ONNX session. Reuse the static ORT env from mOrtEnv.
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(2);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            Ort::Session* rs = nullptr;
            try {
                rs = new Ort::Session(*mOrtEnv, recPath.c_str(), opts);
            } catch (const std::exception& e) {
                std::string msg = std::string("rec ONNX load failed: ") + e.what();
                terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                goto done;
            }
            // Load vocab file (one UTF-8 token per line; index 0 = CTC blank).
            std::ifstream vf(vocabPath);
            std::vector<std::string> v;
            std::string line;
            while (std::getline(vf, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                v.push_back(std::move(line));
            }
            if (v.size() < 2) {
                delete rs;
                terminal = [cb]() {
                    cb->onError(W_MODEL_ERROR, "rec vocab < 2 entries (need blank + ≥1 char)");
                };
                goto done;
            }
            mOcrRec[modelHandle] = OcrRec{rs, std::move(v)};
            oit = mOcrRec.find(modelHandle);
            LOG(INFO) << "oird: loaded OCR rec handle=" << modelHandle
                      << " rec=" << recPath << " vocab_sz=" << oit->second.vocab.size();
        }
        recSession = oit->second.session;
        vocab      = oit->second.vocab;  // copy under lock, immutable after load
    }

    // Decode source image.
    RgbImage img;
    bool ok = false;
    std::string ext4 = imagePath.size() >= 4 ? imagePath.substr(imagePath.size() - 4) : "";
    std::string ext5 = imagePath.size() >= 5 ? imagePath.substr(imagePath.size() - 5) : "";
    for (auto& c : ext4) c = (char)tolower((unsigned char)c);
    for (auto& c : ext5) c = (char)tolower((unsigned char)c);
    if (ext4 == ".jpg" || ext5 == ".jpeg") ok = decodeJpeg(imagePath, img, mImageMaxPixels);
    else if (ext4 == ".png") ok = decodePng(imagePath, img, mImageMaxPixels);
    if (!ok) {
        std::string msg = "image decode failed (need .jpg/.jpeg/.png): " + imagePath;
        terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
        goto done;
    }
    imgW = img.w;
    imgH = img.h;

    // --- Stage 1: Run detection via the loaded det model ---
    // Expected output shape: either [1, 5, N] (YOLO with 1 class) or
    // [1, N, 6] (DETR). The detection family knob (vision.detect.family)
    // controls parsing — OCR uses the same knob, defaulted per
    // oir_config.xml. Low-confidence / overlapping regions NMS-pruned
    // using the existing detect thresholds.
    int32_t detectInputSize;
    float   scoreThresh;
    float   iouThresh;
    std::string detectFamily;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        detectInputSize = mVisionDetectInputSize;
        scoreThresh     = mDetectScoreThresh;
        iouThresh       = mDetectIouThresh;
        detectFamily    = mVisionDetectFamily;
    }

    const int kIn = detectInputSize;
    const float detScale = std::min((float)kIn / img.w, (float)kIn / img.h);
    const int newW = (int)std::round(img.w * detScale);
    const int newH = (int)std::round(img.h * detScale);
    const int padX = (kIn - newW) / 2;
    const int padY = (kIn - newH) / 2;
    std::vector<float> detInput(3 * kIn * kIn, 114.0f / 255.0f);
    for (int y = 0; y < newH; ++y) {
        float fy = (y + 0.5f) / detScale - 0.5f;
        int y0 = std::max(0, (int)std::floor(fy));
        int y1 = std::min(img.h - 1, y0 + 1);
        float wy = fy - y0;
        for (int x = 0; x < newW; ++x) {
            float fx = (x + 0.5f) / detScale - 0.5f;
            int x0 = std::max(0, (int)std::floor(fx));
            int x1 = std::min(img.w - 1, x0 + 1);
            float wx = fx - x0;
            for (int c = 0; c < 3; ++c) {
                float p00 = img.px[(y0 * img.w + x0) * 3 + c];
                float p01 = img.px[(y0 * img.w + x1) * 3 + c];
                float p10 = img.px[(y1 * img.w + x0) * 3 + c];
                float p11 = img.px[(y1 * img.w + x1) * 3 + c];
                float a = p00 * (1 - wx) + p01 * wx;
                float b = p10 * (1 - wx) + p11 * wx;
                float v = (a * (1 - wy) + b * wy) / 255.0f;
                detInput[c * kIn * kIn + (y + padY) * kIn + (x + padX)] = v;
            }
        }
    }

    std::array<int64_t, 4> detShape{1, 3, kIn, kIn};
    Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
    auto detT = Ort::Value::CreateTensor<float>(
            meminfo, detInput.data(), detInput.size(),
            detShape.data(), detShape.size());

    // Output-node names vary per exporter; probe common alternatives.
    Ort::AllocatorWithDefaultOptions alloc;
    auto detInputName  = detSession->GetInputNameAllocated(0, alloc);
    auto detOutputName = detSession->GetOutputNameAllocated(0, alloc);
    const char* detInNames[]  = {detInputName.get()};
    const char* detOutNames[] = {detOutputName.get()};
    Ort::Value detInputs[] = {std::move(detT)};

    std::vector<Ort::Value> detOutputs;
    try {
        detOutputs = detSession->Run(Ort::RunOptions{nullptr},
                                      detInNames, detInputs, 1,
                                      detOutNames, 1);
    } catch (const std::exception& e) {
        std::string msg = std::string("OCR det ORT Run failed: ") + e.what();
        terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
        goto done;
    }

    // Parse detections as axis-aligned text boxes. Output layout:
    //   YOLO-1-class:  [1, 5, N]  (cx, cy, w, h, score)  per-column N-wise
    //   DETR-style:    [1, N, 6]  (x1, y1, x2, y2, score, class)
    const bool useDetr = (detectFamily == "rtdetr" || detectFamily == "detr");
    const float* detData = detOutputs[0].GetTensorData<float>();
    auto detInfo = detOutputs[0].GetTensorTypeAndShapeInfo();
    auto detShapeOut = detInfo.GetShape();
    struct TextBox { float x1, y1, x2, y2, score; };
    std::vector<TextBox> cands;
    if (useDetr && detShapeOut.size() == 3) {
        const int64_t N = detShapeOut[1];
        const int64_t F = detShapeOut[2];
        for (int64_t i = 0; i < N; ++i) {
            float score = detData[i * F + 4];
            if (score < scoreThresh) continue;
            cands.push_back({detData[i*F+0], detData[i*F+1],
                             detData[i*F+2], detData[i*F+3], score});
        }
    } else if (detShapeOut.size() == 3) {
        const int64_t C = detShapeOut[1];   // 5 for 1-class YOLO text detector
        const int64_t N = detShapeOut[2];
        for (int64_t i = 0; i < N; ++i) {
            float score = (C >= 5) ? detData[4 * N + i] : 0.f;
            if (score < scoreThresh) continue;
            float cx = detData[0 * N + i];
            float cy = detData[1 * N + i];
            float w  = detData[2 * N + i];
            float h  = detData[3 * N + i];
            cands.push_back({cx - w/2, cy - h/2, cx + w/2, cy + h/2, score});
        }
    } else {
        terminal = [cb]() { cb->onError(W_MODEL_ERROR, "unsupported OCR det output shape"); };
        goto done;
    }
    candCount = cands.size();

    // Inverse letterbox: map back to original image coords, clip.
    for (auto& c : cands) {
        c.x1 = std::max(0.f, (c.x1 - padX) / detScale);
        c.y1 = std::max(0.f, (c.y1 - padY) / detScale);
        c.x2 = std::min((float)img.w, (c.x2 - padX) / detScale);
        c.y2 = std::min((float)img.h, (c.y2 - padY) / detScale);
    }

    // Simple IoU-NMS — reuses the same thresholds as vision.detect.
    std::sort(cands.begin(), cands.end(),
              [](const TextBox& a, const TextBox& b) { return a.score > b.score; });
    std::vector<bool> keep(cands.size(), true);
    for (size_t i = 0; i < cands.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < cands.size(); ++j) {
            if (!keep[j]) continue;
            float xx1 = std::max(cands[i].x1, cands[j].x1);
            float yy1 = std::max(cands[i].y1, cands[j].y1);
            float xx2 = std::min(cands[i].x2, cands[j].x2);
            float yy2 = std::min(cands[i].y2, cands[j].y2);
            float inter = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);
            float a = std::max(0.f, cands[i].x2 - cands[i].x1)
                    * std::max(0.f, cands[i].y2 - cands[i].y1);
            float b = std::max(0.f, cands[j].x2 - cands[j].x1)
                    * std::max(0.f, cands[j].y2 - cands[j].y1);
            if (inter / (a + b - inter + 1e-6f) > iouThresh) keep[j] = false;
        }
    }

    // --- Stage 2: For each kept box, crop → rec → CTC decode ---
    // PaddleOCR-compatible rec input: 3-channel float32 [1, 3, 48, W]
    // where W scales with aspect ratio (quantized to a multiple of 8).
    // Pixel normalization: (x/255 - 0.5) / 0.5 per channel.
    constexpr int kRecH = 48;
    constexpr int kRecWMin = 48;
    constexpr int kRecWMax = 640;
    std::vector<int> xs, ys, widths, heights, labelsPerBox;
    std::vector<std::string> labelsFlat;
    std::vector<float> scoresFlat;

    for (size_t i = 0; i < cands.size(); ++i) {
        if (!keep[i]) continue;
        const auto& c = cands[i];
        int cx1 = (int)std::max(0.f, std::floor(c.x1));
        int cy1 = (int)std::max(0.f, std::floor(c.y1));
        int cx2 = (int)std::min((float)img.w, std::ceil(c.x2));
        int cy2 = (int)std::min((float)img.h, std::ceil(c.y2));
        int cw = cx2 - cx1;
        int ch = cy2 - cy1;
        if (cw < 4 || ch < 4) continue;

        // Resize crop to (kRecH, recW) preserving aspect.
        int recW = std::max(kRecWMin,
                std::min(kRecWMax, (int)std::round((float)cw * kRecH / ch)));
        recW = ((recW + 7) / 8) * 8;  // multiple of 8 for kernel alignment

        std::vector<float> recInput(3 * kRecH * recW, 0.f);
        const float sx = (float)cw / recW;
        const float sy = (float)ch / kRecH;
        for (int y = 0; y < kRecH; ++y) {
            float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = std::max(0, (int)std::floor(fy));
            int y1 = std::min(ch - 1, y0 + 1);
            float wy = fy - y0;
            for (int x = 0; x < recW; ++x) {
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = std::max(0, (int)std::floor(fx));
                int x1 = std::min(cw - 1, x0 + 1);
                float wx = fx - x0;
                for (int chn = 0; chn < 3; ++chn) {
                    int srcYX0 = (cy1 + y0) * img.w + (cx1 + x0);
                    int srcYX1 = (cy1 + y0) * img.w + (cx1 + x1);
                    int srcY1X0 = (cy1 + y1) * img.w + (cx1 + x0);
                    int srcY1X1 = (cy1 + y1) * img.w + (cx1 + x1);
                    float p00 = img.px[srcYX0  * 3 + chn];
                    float p01 = img.px[srcYX1  * 3 + chn];
                    float p10 = img.px[srcY1X0 * 3 + chn];
                    float p11 = img.px[srcY1X1 * 3 + chn];
                    float a = p00 * (1 - wx) + p01 * wx;
                    float b = p10 * (1 - wx) + p11 * wx;
                    float v = (a * (1 - wy) + b * wy) / 255.0f;
                    // PaddleOCR normalization: (x - 0.5) / 0.5
                    recInput[chn * kRecH * recW + y * recW + x] = (v - 0.5f) / 0.5f;
                }
            }
        }

        // Rec ORT run.
        std::array<int64_t, 4> recShape{1, 3, kRecH, recW};
        auto recT = Ort::Value::CreateTensor<float>(
                meminfo, recInput.data(), recInput.size(),
                recShape.data(), recShape.size());
        auto recInName  = recSession->GetInputNameAllocated(0, alloc);
        auto recOutName = recSession->GetOutputNameAllocated(0, alloc);
        const char* recInNames[]  = {recInName.get()};
        const char* recOutNames[] = {recOutName.get()};
        Ort::Value recInputs[] = {std::move(recT)};

        std::vector<Ort::Value> recOutputs;
        try {
            recOutputs = recSession->Run(Ort::RunOptions{nullptr},
                                          recInNames, recInputs, 1,
                                          recOutNames, 1);
        } catch (const std::exception&) {
            continue;  // skip this region; don't fail the whole request
        }

        // Rec output: [1, T, C]  (logits or softmax-probs, per-timestep classes)
        const float* logits = recOutputs[0].GetTensorData<float>();
        auto recInfo = recOutputs[0].GetTensorTypeAndShapeInfo();
        auto recOutShape = recInfo.GetShape();
        if (recOutShape.size() != 3) continue;
        const int64_t T = recOutShape[1];
        const int64_t C = recOutShape[2];

        // CTC greedy decode: argmax per timestep, collapse repeats, drop blank(0).
        std::string text;
        int64_t prevId = -1;
        float scoreSum = 0.f;
        int   scoreN   = 0;
        for (int64_t t = 0; t < T; ++t) {
            int64_t bestId = 0;
            float bestVal = logits[t * C + 0];
            for (int64_t k = 1; k < C; ++k) {
                float v = logits[t * C + k];
                if (v > bestVal) { bestVal = v; bestId = k; }
            }
            if (bestId != 0 && bestId != prevId
                    && (size_t)bestId < vocab.size()) {
                text += vocab[bestId];
                scoreSum += bestVal;
                scoreN++;
            }
            prevId = bestId;
        }
        if (text.empty()) continue;  // no characters decoded

        // Emit this region.
        xs.push_back((int)std::round(c.x1));
        ys.push_back((int)std::round(c.y1));
        widths.push_back((int)std::round(c.x2 - c.x1));
        heights.push_back((int)std::round(c.y2 - c.y1));
        labelsPerBox.push_back(1);
        labelsFlat.push_back(std::move(text));
        scoresFlat.push_back(scoreN > 0 ? scoreSum / scoreN : c.score);
    }

    keptCount = labelsFlat.size();
    terminal = [cb,
                xs = std::move(xs),
                ys = std::move(ys),
                widths = std::move(widths),
                heights = std::move(heights),
                labelsPerBox = std::move(labelsPerBox),
                labelsFlat = std::move(labelsFlat),
                scoresFlat = std::move(scoresFlat)]() {
        cb->onBoundingBoxes(xs, ys, widths, heights,
                            labelsPerBox, labelsFlat, scoresFlat);
    };
    }
done:
    guard->release();  // explicit early release; matches v0.6.8 ordering.
    if (terminal) terminal();

    LOG(INFO) << "oird: submitOcr handle=" << modelHandle
              << " img=" << imgW << "x" << imgH
              << " candidates=" << candCount
              << " kept=" << keptCount;
        });  // v0.6.4: close mRt.mScheduler->enqueue lambda
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::submitDetect(int64_t modelHandle,
                                  const std::string& imagePath,
                                  const std::shared_ptr<IOirWorkerBboxCallback>& cb,
                                  int64_t* _aidl_return) {
    Ort::Session* session = nullptr;
    std::vector<std::string> classLabels; // v0.5 V8: per-model sidecar, empty → COCO-80 fallback
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isOnnx || !it->second.onnxIsDetection) {
            cb->onError(W_INVALID_INPUT, "handle not an onnx detection model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        session = it->second.ortSession;
        classLabels = it->second.detectClassLabels; // copy under lock (labels are immutable post-load)
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue on scheduler. classLabels was already copied
    // under lock above; capture by move so the lambda owns it.
    mRt.mScheduler->enqueue(priorityForCapability("vision.detect"),
        [this, modelHandle, imagePath, cb, session, guard,
         classLabels = std::move(classLabels)]() mutable {
    // v0.6.8: terminal cb deferred past releaseInflight.
    std::function<void()> terminal;
    int imgW = 0, imgH = 0;
    size_t candCount = 0, keptCount = 0;
    int64_t t0 = 0, t1 = 0;
    {
    // v0.5 V7: snapshot detect-tuning knobs under lock.
    int32_t detectInputSize;
    std::string detectFamily;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        detectInputSize = mVisionDetectInputSize;
        detectFamily    = mVisionDetectFamily;
    }

    // Decode
    RgbImage img;
    bool ok = false;
    std::string ext4 = imagePath.size() >= 4 ? imagePath.substr(imagePath.size() - 4) : "";
    std::string ext5 = imagePath.size() >= 5 ? imagePath.substr(imagePath.size() - 5) : "";
    for (auto& c : ext4) c = (char)tolower((unsigned char)c);
    for (auto& c : ext5) c = (char)tolower((unsigned char)c);
    if (ext4 == ".jpg" || ext5 == ".jpeg") ok = decodeJpeg(imagePath, img, mImageMaxPixels);
    else if (ext4 == ".png") ok = decodePng(imagePath, img, mImageMaxPixels);
    if (!ok) {
        std::string msg = "image decode failed (need .jpg/.jpeg/.png): " + imagePath;
        terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
        goto done;
    }
    imgW = img.w; imgH = img.h;

    // Letterbox to kIn×kIn (OEM-tunable, default 640 for YOLOv8n),
    // preserving aspect ratio, padding gray (114/255).
    // Record scale + pad so bboxes can be mapped back to source coords.
    const int kIn = detectInputSize;
    const float scale = std::min((float)kIn / img.w, (float)kIn / img.h);
    const int newW = (int)std::round(img.w * scale);
    const int newH = (int)std::round(img.h * scale);
    const int padX = (kIn - newW) / 2;
    const int padY = (kIn - newH) / 2;
    std::vector<float> input(3 * kIn * kIn, 114.0f / 255.0f);  // gray fill

    for (int y = 0; y < newH; ++y) {
        float fy = (y + 0.5f) / scale - 0.5f;
        int y0 = std::max(0, (int)std::floor(fy));
        int y1 = std::min(img.h - 1, y0 + 1);
        float wy = fy - y0;
        for (int x = 0; x < newW; ++x) {
            float fx = (x + 0.5f) / scale - 0.5f;
            int x0 = std::max(0, (int)std::floor(fx));
            int x1 = std::min(img.w - 1, x0 + 1);
            float wx = fx - x0;
            for (int c = 0; c < 3; ++c) {
                float p00 = img.px[(y0 * img.w + x0) * 3 + c];
                float p01 = img.px[(y0 * img.w + x1) * 3 + c];
                float p10 = img.px[(y1 * img.w + x0) * 3 + c];
                float p11 = img.px[(y1 * img.w + x1) * 3 + c];
                float a = p00 * (1 - wx) + p01 * wx;
                float b = p10 * (1 - wx) + p11 * wx;
                float v = (a * (1 - wy) + b * wy) / 255.0f;
                int outY = y + padY;
                int outX = x + padX;
                input[c * kIn * kIn + outY * kIn + outX] = v;
            }
        }
    }

    // ORT Run
    try {
        Ort::AllocatorWithDefaultOptions alloc;
        if (session->GetInputCount() < 1 || session->GetOutputCount() < 1) {
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "ORT session has 0 inputs or outputs"); };
            goto done;
        }
        // v0.5: pull all output names so the family-dispatched parsers can
        // pick the tensors they need by name (RT-DETR returns 3, YOLOv8 returns 1).
        const size_t nIn  = session->GetInputCount();
        const size_t nOut = session->GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> inNamePtrs;
        std::vector<Ort::AllocatedStringPtr> outNamePtrs;
        std::vector<const char*> inNames;
        std::vector<const char*> outNames;
        for (size_t i = 0; i < nIn; ++i) {
            inNamePtrs.push_back(session->GetInputNameAllocated(i, alloc));
            inNames.push_back(inNamePtrs.back().get());
        }
        for (size_t i = 0; i < nOut; ++i) {
            outNamePtrs.push_back(session->GetOutputNameAllocated(i, alloc));
            outNames.push_back(outNamePtrs.back().get());
        }

        std::array<int64_t, 4> inShape = {1, 3, kIn, kIn};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memInfo, input.data(), input.size(), inShape.data(), inShape.size());

        t0 = currentTimeMs();
        std::vector<Ort::Value> outputs = session->Run(
                Ort::RunOptions{nullptr},
                inNames.data(), &inputTensor, 1,
                outNames.data(), outNames.size());
        t1 = currentTimeMs();

        if (outputs.empty()) {
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "ORT Run returned no outputs"); };
            goto done;
        }

        // v0.5: dispatch on family. Default-fill in CapabilityRegistry +
        // OEM knob (vision.detect.family). "yolov8" / "yolov5" use the
        // anchor-based YOLO parser; "rtdetr" / "detr" use the query-based
        // DETR parser. Anything else falls back to YOLOv8 with a warning.
        const bool useDetr = (detectFamily == "rtdetr" || detectFamily == "detr");
        Ort::Value& outTensor = outputs[0];
        auto info = outTensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        float* outData = outTensor.GetTensorMutableData<float>();

        // v0.5 V7: NMS thresholds are OEM-tunable via <capability_tuning>
        // in oir_config.xml. Defaults (0.25/0.45) match v0.4 behavior.
        const float kScoreThresh = mDetectScoreThresh;
        const float kIouThresh   = mDetectIouThresh;

        struct Candidate {
            float x1, y1, x2, y2;
            int classIdx;
            float score;
        };
        std::vector<Candidate> cands;
        cands.reserve(256);

        auto unletterbox = [&](float& x1, float& y1, float& x2, float& y2) {
            x1 = std::max(0.0f, std::min((x1 - padX) / scale, (float)img.w));
            y1 = std::max(0.0f, std::min((y1 - padY) / scale, (float)img.h));
            x2 = std::max(0.0f, std::min((x2 - padX) / scale, (float)img.w));
            y2 = std::max(0.0f, std::min((y2 - padY) / scale, (float)img.h));
        };

        if (!useDetr) {
            // ---- YOLOv8 parser (anchor-based, single output tensor) ----
            // Output shape: (1, 84, 8400) where 84 = 4 bbox + 80 classes.
            // Some exporters produce (1, 8400, 84) transposed — detect both.
            int nPred = 0;
            int nAttr = 0;
            bool transposed = false;
            if (shape.size() == 3 && shape[0] == 1) {
                if (shape[1] == 84 && shape[2] > 100) {
                    nAttr = (int)shape[1]; nPred = (int)shape[2];
                } else if (shape[2] == 84 && shape[1] > 100) {
                    nPred = (int)shape[1]; nAttr = (int)shape[2]; transposed = true;
                }
            }
            if (nAttr != 84 || nPred < 1) {
                std::string msg = "unexpected YOLO output shape; need (1,84,N) or (1,N,84)";
                terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                goto done;
            }
            const int nClasses = nAttr - 4;

            auto at = [&](int anchor, int attr) -> float {
                return transposed ? outData[anchor * nAttr + attr]
                                   : outData[attr * nPred + anchor];
            };

            for (int a = 0; a < nPred; ++a) {
                int bestCls = -1;
                float bestScore = 0.0f;
                for (int c = 0; c < nClasses; ++c) {
                    float s = at(a, 4 + c);
                    if (s > bestScore) { bestScore = s; bestCls = c; }
                }
                if (bestScore < kScoreThresh) continue;

                // YOLOv8 bbox is (cx, cy, w, h) in kIn × kIn space.
                float cx = at(a, 0), cy = at(a, 1);
                float w  = at(a, 2), h  = at(a, 3);
                float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
                float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;
                unletterbox(x1, y1, x2, y2);
                cands.push_back({x1, y1, x2, y2, bestCls, bestScore});
            }
        } else {
            // ---- RT-DETR / DETR parser (query-based) ----
            // Accepts two export conventions:
            //   (a) PaddlePaddle/NVIDIA triple: "boxes" [B,N,4], "scores" [B,N],
            //       "labels" [B,N int64].
            //   (b) HuggingFace transformers pair: "pred_boxes" [B,N,4],
            //       "logits" [B,N,C] — scores/labels derived via argmax per query.
            // Boxes are typically cxcywh; some exports use xyxy. Heuristic:
            // if any value > 2.0 → already in pixel space (kIn-scale); else
            // assume normalized 0-1 and multiply by kIn.
            const float* boxesData  = nullptr;
            const float* scoresData = nullptr;
            const int64_t* labelsData = nullptr;
            const float* logitsData = nullptr;
            int logitsNumClasses = 0;
            int nQueries = 0;
            int boxStride = 4;
            bool xyxy = false;
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto si = outputs[i].GetTensorTypeAndShapeInfo();
                auto sh = si.GetShape();
                std::string nm = (i < outNames.size()) ? std::string(outNames[i]) : "";
                auto elemType = si.GetElementType();
                if (nm == "boxes" || nm == "pred_boxes"
                    || (sh.size() == 3 && sh.back() == 4 && boxesData == nullptr)) {
                    boxesData = outputs[i].GetTensorData<float>();
                    if (sh.size() == 3) nQueries = (int)sh[1];
                    if (nm.find("xyxy") != std::string::npos) xyxy = true;
                } else if (nm == "logits"
                           || (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                               && sh.size() == 3 && logitsData == nullptr)) {
                    // HF transformers: [B, N, C] class logits per query.
                    logitsData = outputs[i].GetTensorData<float>();
                    logitsNumClasses = (int)sh[2];
                    if (nQueries == 0) nQueries = (int)sh[1];
                } else if (nm == "scores" || (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                              && sh.size() == 2 && scoresData == nullptr
                                              && (int)sh[1] == nQueries)) {
                    scoresData = outputs[i].GetTensorData<float>();
                } else if (nm == "labels" || elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    labelsData = outputs[i].GetTensorData<int64_t>();
                }
            }
            // Derive scores/labels from logits if the model uses the HF
            // (logits + pred_boxes) export convention.
            std::vector<float>   derivedScores;
            std::vector<int64_t> derivedLabels;
            if (logitsData && nQueries > 0 && logitsNumClasses > 0
                && (!scoresData || !labelsData)) {
                derivedScores.resize(nQueries);
                derivedLabels.resize(nQueries);
                for (int q = 0; q < nQueries; ++q) {
                    const float* row = logitsData + (size_t)q * logitsNumClasses;
                    int bestC = 0;
                    float bestL = row[0];
                    for (int c = 1; c < logitsNumClasses; ++c) {
                        if (row[c] > bestL) { bestL = row[c]; bestC = c; }
                    }
                    // Sigmoid for RT-DETR (which uses focal-loss-style per-class
                    // sigmoid scores, not softmax). Good enough as a scalar
                    // confidence signal downstream of NMS.
                    derivedScores[q] = 1.0f / (1.0f + std::exp(-bestL));
                    derivedLabels[q] = bestC;
                }
                scoresData = derivedScores.data();
                labelsData = derivedLabels.data();
            }
            if (!boxesData || !scoresData || !labelsData || nQueries < 1) {
                terminal = [cb]() {
                    cb->onError(W_MODEL_ERROR,
                                "RT-DETR output mismatch: need boxes+scores+labels "
                                "(triple) or pred_boxes+logits (HF)");
                };
                goto done;
            }
            // Box-space heuristic: scan first 16 values for any > 2.0.
            bool inPixelSpace = false;
            for (int i = 0; i < std::min(16, nQueries * 4); ++i) {
                if (std::fabs(boxesData[i]) > 2.0f) { inPixelSpace = true; break; }
            }
            const float boxScale = inPixelSpace ? 1.0f : (float)kIn;
            for (int q = 0; q < nQueries; ++q) {
                float s = scoresData[q];
                if (s < kScoreThresh) continue;
                int cls = (int)labelsData[q];
                float a = boxesData[q * 4 + 0] * boxScale;
                float b = boxesData[q * 4 + 1] * boxScale;
                float c = boxesData[q * 4 + 2] * boxScale;
                float d = boxesData[q * 4 + 3] * boxScale;
                float x1, y1, x2, y2;
                if (xyxy) {
                    x1 = a; y1 = b; x2 = c; y2 = d;
                } else {
                    // cxcywh
                    x1 = a - c * 0.5f; y1 = b - d * 0.5f;
                    x2 = a + c * 0.5f; y2 = b + d * 0.5f;
                }
                unletterbox(x1, y1, x2, y2);
                cands.push_back({x1, y1, x2, y2, cls, s});
            }
        }

        // Sort by score descending.
        std::sort(cands.begin(), cands.end(),
                  [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        // NMS per class.
        std::vector<bool> keep(cands.size(), true);
        auto iou = [](const Candidate& a, const Candidate& b) -> float {
            float ix1 = std::max(a.x1, b.x1);
            float iy1 = std::max(a.y1, b.y1);
            float ix2 = std::min(a.x2, b.x2);
            float iy2 = std::min(a.y2, b.y2);
            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;
            float aArea = std::max(0.0f, (a.x2 - a.x1)) * std::max(0.0f, (a.y2 - a.y1));
            float bArea = std::max(0.0f, (b.x2 - b.x1)) * std::max(0.0f, (b.y2 - b.y1));
            float uni = aArea + bArea - inter;
            return uni > 0.0f ? inter / uni : 0.0f;
        };
        for (size_t i = 0; i < cands.size(); ++i) {
            if (!keep[i]) continue;
            for (size_t j = i + 1; j < cands.size(); ++j) {
                if (!keep[j]) continue;
                if (cands[j].classIdx != cands[i].classIdx) continue;
                if (iou(cands[i], cands[j]) > kIouThresh) keep[j] = false;
            }
        }

        // Embedded COCO-80 fallback (standard YOLOv8 training set). OEMs
        // override by shipping <model>.classes.json alongside the .onnx —
        // parsed at load time and resolved in the labels block below (v0.5 V8).
        static const char* kCoco80[] = {
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
            "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
            "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        };

        // Build flattened arrays for the callback.
        std::vector<int> xs, ys, widths, heights, labelsPerBox;
        std::vector<std::string> labelsFlat;
        std::vector<float> scoresFlat;
        int nKept = 0;
        for (size_t i = 0; i < cands.size(); ++i) {
            if (!keep[i]) continue;
            const auto& c = cands[i];
            int cls = c.classIdx;
            // v0.5 V8: prefer OEM sidecar labels; fall back to embedded COCO-80.
            std::string name;
            if (cls >= 0 && cls < (int)classLabels.size()) {
                name = classLabels[cls];
            } else if (cls >= 0 && cls < (int)(sizeof(kCoco80)/sizeof(kCoco80[0]))) {
                name = kCoco80[cls];
            } else {
                name = "unknown";
            }
            xs.push_back((int)std::round(c.x1));
            ys.push_back((int)std::round(c.y1));
            widths.push_back((int)std::round(c.x2 - c.x1));
            heights.push_back((int)std::round(c.y2 - c.y1));
            labelsPerBox.push_back(1);
            labelsFlat.push_back(std::move(name));
            scoresFlat.push_back(c.score);
            ++nKept;
        }

        candCount = cands.size();
        keptCount = nKept;
        terminal = [cb,
                    xs = std::move(xs),
                    ys = std::move(ys),
                    widths = std::move(widths),
                    heights = std::move(heights),
                    labelsPerBox = std::move(labelsPerBox),
                    labelsFlat = std::move(labelsFlat),
                    scoresFlat = std::move(scoresFlat)]() {
            cb->onBoundingBoxes(xs, ys, widths, heights,
                                labelsPerBox, labelsFlat, scoresFlat);
        };
    } catch (const Ort::Exception& e) {
        std::string msg = std::string("ORT inference failed: ") + e.what();
        terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
    }
    }
done:
    guard->release();  // explicit early release; matches v0.6.8 ordering.
    if (terminal) terminal();

    LOG(INFO) << "oird: detect handle=" << modelHandle
              << " img=" << imgW << "x" << imgH
              << " candidates=" << candCount
              << " kept=" << keptCount
              << " wall_ms=" << (t1 - t0);
        });  // v0.6.4: close mRt.mScheduler->enqueue lambda
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::loadVisionEmbed(const std::string& modelPath, int64_t* _aidl_return) {
    // v0.6.9: mRt.mLock shrunk. Original code had a nested lock_guard on
    // mRt.mLock inside the validate block (line ~3364) while already
    // holding the outer lock_guard → non-recursive self-deadlock.
    // The snapshot now happens before the slow ctor and after lock
    // release.
    const std::string key = "onnx-ve:" + modelPath;
    std::unique_lock<std::mutex> lk(mRt.mLock);
    for (auto& [h, m] : mRt.mModels) {
        if (m.path == modelPath && m.isVisionEmbed) {
            *_aidl_return = h;
            return ::ndk::ScopedAStatus::ok();
        }
    }

    auto claim = mRt.mLoadRegistry.claim(lk, key);
    if (claim.waited) {
        if (claim.waited->errCode != 0) {
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    claim.waited->errCode, claim.waited->errMsg.c_str());
        }
        *_aidl_return = claim.waited->handle;
        return ::ndk::ScopedAStatus::ok();
    }
    auto slot = claim.slot;

    const int64_t newSize = fileSizeBytes(modelPath);
    const int32_t kTarget = mVisionEmbedInputSize;
    mRt.mBudget.addResident(newSize);

    lk.unlock();

    ensureOrtEnv();
    Ort::SessionOptions so = makeOrtSessionOptions(false);
    Ort::Session* session = nullptr;
    try {
        session = new Ort::Session(*mOrtEnv, modelPath.c_str(), so);
    } catch (const Ort::Exception& e) {
        LOG(ERROR) << "oird: Ort::Session (vision embed) failed: " << e.what();
        const std::string msg = std::string("vision embed load failed: ") + e.what();
        lk.lock();
        mRt.mBudget.subResident(newSize);
        mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, msg);
        return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                W_MODEL_ERROR, msg.c_str());
    }
    // v0.6 Phase A: SigLIP-style shape contract — [1, 3, kTarget, kTarget]
    // with batch wildcard. Output dim varies by model (768 base / 1024
    // large), not constrained here.
    {
        std::vector<std::vector<int64_t>> inShapes = {
            {-1, 3, kTarget, kTarget},
        };
        std::string err = validateOrtContract(session, 1, inShapes, "vision.embed");
        if (!err.empty()) {
            LOG(ERROR) << "oird: " << err;
            delete session;
            lk.lock();
            mRt.mBudget.subResident(newSize);
            mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_INCOMPATIBLE, err);
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_INCOMPATIBLE, err.c_str());
        }
    }

    lk.lock();

    const int64_t handle = mRt.mNextModelHandle++;
    const int64_t now = currentTimeMs();
    LoadedModel lm;
    lm.ortSession = session;
    lm.handle = handle;
    lm.path = modelPath;
    lm.sizeBytes = newSize;
    lm.loadTimestampMs = now;
    lm.lastAccessMs = now;
    lm.isOnnx = true;
    lm.isVisionEmbed = true;
    mRt.mModels[handle] = std::move(lm);
    registerModelResourceLocked(handle);

    mRt.mLoadRegistry.publish(lk, key, slot, handle, 0, "");

    *_aidl_return = handle;
    LOG(INFO) << "oird: vision embed model loaded handle=" << handle << " path=" << modelPath;
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::submitVisionEmbed(int64_t modelHandle,
                                       const std::string& imagePath,
                                       const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                       int64_t* _aidl_return) {
    Ort::Session* session = nullptr;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isVisionEmbed) {
            cb->onError(W_INVALID_INPUT, "handle not a vision embed model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        session = it->second.ortSession;
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue on scheduler.
    mRt.mScheduler->enqueue(priorityForCapability("vision.embed"),
        [this, modelHandle, imagePath, cb, session, guard]() {
    // v0.6.8: terminal cb deferred past releaseInflight.
    std::function<void()> terminal;
    int imgW = 0, imgH = 0;
    size_t vecDim = 0;
    int64_t t0 = 0, t1 = 0;
    {
    // Decode — sniff extension (jpg/jpeg/png).
    RgbImage img;
    bool ok = false;
    std::string ext4 = imagePath.size() >= 4 ? imagePath.substr(imagePath.size() - 4) : "";
    std::string ext5 = imagePath.size() >= 5 ? imagePath.substr(imagePath.size() - 5) : "";
    for (auto& c : ext4) c = (char)tolower((unsigned char)c);
    for (auto& c : ext5) c = (char)tolower((unsigned char)c);
    if (ext4 == ".jpg" || ext5 == ".jpeg") ok = decodeJpeg(imagePath, img, mImageMaxPixels);
    else if (ext4 == ".png") ok = decodePng(imagePath, img, mImageMaxPixels);
    if (!ok) {
        std::string msg = "image decode failed (need .jpg/.jpeg/.png): " + imagePath;
        terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
        goto done;
    }
    imgW = img.w; imgH = img.h;

    // v0.5 V7: OEM-tunable input size + normalization (defaults: SigLIP-base
    // 224 w/ mean=0.5, std=0.5 → [-1, 1] range).
    int32_t kTarget;
    float normMean;
    float normStd;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        kTarget  = mVisionEmbedInputSize;
        normMean = mVisionEmbedNormMean;
        normStd  = mVisionEmbedNormStd;
    }
    // Preprocess: bilinear resize kTarget×kTarget, CHW float, encoder normalization.
    std::vector<float> input(3 * kTarget * kTarget);
    const float sx = (float)img.w / kTarget;
    const float sy = (float)img.h / kTarget;
    for (int y = 0; y < kTarget; ++y) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = std::max(0, (int)std::floor(fy));
        int y1 = std::min(img.h - 1, y0 + 1);
        float wy = fy - y0;
        for (int x = 0; x < kTarget; ++x) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = std::max(0, (int)std::floor(fx));
            int x1 = std::min(img.w - 1, x0 + 1);
            float wx = fx - x0;
            for (int c = 0; c < 3; ++c) {
                float p00 = img.px[(y0 * img.w + x0) * 3 + c];
                float p01 = img.px[(y0 * img.w + x1) * 3 + c];
                float p10 = img.px[(y1 * img.w + x0) * 3 + c];
                float p11 = img.px[(y1 * img.w + x1) * 3 + c];
                float a = p00 * (1 - wx) + p01 * wx;
                float b = p10 * (1 - wx) + p11 * wx;
                float v = (a * (1 - wy) + b * wy) / 255.0f;
                v = (v - normMean) / normStd;
                input[c * kTarget * kTarget + y * kTarget + x] = v;
            }
        }
    }

    // ORT Run.
    try {
        Ort::AllocatorWithDefaultOptions alloc;
        if (session->GetInputCount() < 1 || session->GetOutputCount() < 1) {
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "ORT session has 0 inputs or outputs"); };
            goto done;
        }
        Ort::AllocatedStringPtr inNamePtr  = session->GetInputNameAllocated(0, alloc);
        Ort::AllocatedStringPtr outNamePtr = session->GetOutputNameAllocated(0, alloc);
        const char* inName  = inNamePtr.get();
        const char* outName = outNamePtr.get();

        std::array<int64_t, 4> inShape = {1, 3, kTarget, kTarget};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memInfo, input.data(), input.size(), inShape.data(), inShape.size());

        t0 = currentTimeMs();
        std::vector<Ort::Value> outputs = session->Run(
                Ort::RunOptions{nullptr},
                &inName, &inputTensor, 1,
                &outName, 1);
        t1 = currentTimeMs();

        if (outputs.empty()) {
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "ORT Run returned no outputs"); };
            goto done;
        }
        Ort::Value& outTensor = outputs[0];
        auto info = outTensor.GetTensorTypeAndShapeInfo();
        size_t n_elems = info.GetElementCount();
        float* outData = outTensor.GetTensorMutableData<float>();
        std::vector<int64_t> shape = info.GetShape();

        std::vector<float> vec;
        if (shape.size() == 2 && shape[0] == 1) {
            // (1, dim) — already pooled.
            vec.assign(outData, outData + shape[1]);
        } else if (shape.size() == 3 && shape[0] == 1) {
            // (1, n_patches, dim) — mean-pool over patches.
            int n_patches = (int)shape[1];
            int dim = (int)shape[2];
            vec.assign(dim, 0.0f);
            for (int p = 0; p < n_patches; ++p)
                for (int d = 0; d < dim; ++d)
                    vec[d] += outData[p * dim + d];
            for (float& v : vec) v /= n_patches;
        } else {
            // Unknown shape — return raw and let caller deal.
            vec.assign(outData, outData + n_elems);
        }

        // L2-normalize for cosine-similarity convention.
        double sum2 = 0.0;
        for (float v : vec) sum2 += (double)v * v;
        float norm = (float)std::sqrt(sum2);
        if (norm > 1e-8f) for (float& v : vec) v /= norm;

        vecDim = vec.size();
        terminal = [cb, vec = std::move(vec)]() { cb->onVector(vec); };
    } catch (const Ort::Exception& e) {
        std::string msg = std::string("ORT inference failed: ") + e.what();
        terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
    }
    }
done:
    guard->release();  // explicit early release; matches v0.6.8 ordering.
    if (terminal) terminal();
    LOG(INFO) << "oird: vision embed handle=" << modelHandle
              << " img=" << imgW << "x" << imgH
              << " dim=" << vecDim
              << " wall_ms=" << (t1 - t0);
        });  // v0.6.4: close mRt.mScheduler->enqueue lambda
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::loadVad(const std::string& modelPath,
                             int64_t* _aidl_return) {
    // v0.6.9: mRt.mLock shrunk. Original code had a nested lock_guard on
    // mRt.mLock (line ~4036) while already holding the outer lock_guard →
    // non-recursive self-deadlock the moment loadVad was called.
    // Snapshot VAD tunables under the initial lock before releasing.
    const std::string key = "vad:" + modelPath;
    std::unique_lock<std::mutex> lk(mRt.mLock);
    for (auto& [h, m] : mRt.mModels) {
        if (m.path == modelPath && m.isVad) {
            *_aidl_return = h;
            return ::ndk::ScopedAStatus::ok();
        }
    }

    auto claim = mRt.mLoadRegistry.claim(lk, key);
    if (claim.waited) {
        if (claim.waited->errCode != 0) {
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    claim.waited->errCode, claim.waited->errMsg.c_str());
        }
        *_aidl_return = claim.waited->handle;
        return ::ndk::ScopedAStatus::ok();
    }
    auto slot = claim.slot;

    const int64_t newSize = fileSizeBytes(modelPath);
    const int32_t wSamples = mVadWindowSamples;
    const int32_t cSamples = mVadContextSamples;
    mRt.mBudget.addResident(newSize);

    lk.unlock();

    ensureOrtEnv();
    Ort::SessionOptions so = makeOrtSessionOptions(false);
    Ort::Session* session = nullptr;
    try {
        session = new Ort::Session(*mOrtEnv, modelPath.c_str(), so);
    } catch (const Ort::Exception& e) {
        LOG(ERROR) << "oird: Ort::Session (vad) failed: " << e.what();
        const std::string msg = std::string("vad load failed: ") + e.what();
        lk.lock();
        mRt.mBudget.subResident(newSize);
        mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, msg);
        return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                W_MODEL_ERROR, msg.c_str());
    }
    // v0.6 Phase A: Silero shape contract — audio input, LSTM state, sr.
    // Wildcards on runtime-dependent dims so OEMs running 8 kHz or
    // adjusted window still load clean.
    {
        std::vector<std::vector<int64_t>> inShapes = {
            {1, (int64_t)(cSamples + wSamples)},
            {2, 1, 128},
            {1},
        };
        std::string err = validateOrtContract(session, 3, inShapes, "audio.vad");
        if (!err.empty()) {
            LOG(ERROR) << "oird: " << err;
            delete session;
            lk.lock();
            mRt.mBudget.subResident(newSize);
            mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_INCOMPATIBLE, err);
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_INCOMPATIBLE, err.c_str());
        }
    }

    lk.lock();

    const int64_t handle = mRt.mNextModelHandle++;
    const int64_t now = currentTimeMs();
    LoadedModel lm;
    lm.ortSession = session;
    lm.handle = handle;
    lm.path = modelPath;
    lm.sizeBytes = newSize;
    lm.loadTimestampMs = now;
    lm.lastAccessMs = now;
    lm.isOnnx = true;
    lm.isVad = true;
    mRt.mModels[handle] = std::move(lm);
    registerModelResourceLocked(handle);

    mRt.mLoadRegistry.publish(lk, key, slot, handle, 0, "");

    *_aidl_return = handle;
    LOG(INFO) << "oird: vad model loaded handle=" << handle << " path=" << modelPath
              << " size=" << (newSize >> 10) << "KB";
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::submitVad(int64_t modelHandle,
                               const std::string& pcmPath,
                               const std::shared_ptr<IOirWorkerRealtimeBooleanCallback>& cb,
                               int64_t* _aidl_return) {
    Ort::Session* session = nullptr;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isVad) {
            cb->onError(W_INVALID_INPUT, "handle not a vad model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        session = it->second.ortSession;
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue on the cross-backend scheduler at audio-realtime
    // priority so a VAD request on a shared hardware queue jumps ahead
    // of queued text/vision submits. ORT Run() is thread-safe so no
    // pool needed — the scheduler worker drives the per-window loop
    // directly against the cached session pointer.
    mRt.mScheduler->enqueue(priorityForCapability("audio.vad"),
        [this, modelHandle, pcmPath, cb, session, guard]() {
            // v0.6.8: onState streams inline; onComplete / onError
            // deferred past releaseInflight.
            std::function<void()> terminal;
            size_t sampleCount = 0;
            int64_t windowsProcessed = 0;
            {
                std::ifstream f(pcmPath, std::ios::binary);
                if (!f.is_open()) {
                    std::string msg = "pcm open failed: " + pcmPath;
                    terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
                    goto done;
                }
                f.seekg(0, std::ios::end);
                const std::streamsize byteCount = f.tellg();
                f.seekg(0, std::ios::beg);
                sampleCount = byteCount / 2;
                std::vector<int16_t> samples(sampleCount);
                f.read(reinterpret_cast<char*>(samples.data()), sampleCount * 2);
                f.close();

                int32_t sampleRateHz;
                int32_t windowSamples;
                int32_t contextSamples;
                float voiceThreshold;
                {
                    std::lock_guard<std::mutex> lk(mRt.mLock);
                    sampleRateHz   = mVadSampleRateHz;
                    windowSamples  = mVadWindowSamples;
                    contextSamples = mVadContextSamples;
                    voiceThreshold = mVadVoiceThreshold;
                }
                const int kWindow  = windowSamples;
                const int kContext = contextSamples;
                const int kInputLen = kContext + kWindow;
                const int64_t kSr = sampleRateHz;
                std::vector<float> state(2 * 1 * 128, 0.0f);
                std::vector<float> context(kContext, 0.0f);
                std::vector<float> inputBuf(kInputLen, 0.0f);
                std::vector<int64_t> srBuf{kSr};

                const std::array<int64_t, 2> inputShape  = {1, kInputLen};
                const std::array<int64_t, 3> stateShape  = {2, 1, 128};
                const int64_t* srShapePtr = nullptr;

                const char* inputNames[]  = {"input", "state", "sr"};
                const char* outputNames[] = {"output", "stateN"};

                const auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

                try {
                    for (size_t off = 0; off + kWindow <= samples.size(); off += kWindow) {
                        std::copy(context.begin(), context.end(), inputBuf.begin());
                        for (int i = 0; i < kWindow; ++i) {
                            inputBuf[kContext + i] = samples[off + i] / 32768.0f;
                        }

                        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                                memInfo, inputBuf.data(), inputBuf.size(),
                                inputShape.data(), inputShape.size());
                        Ort::Value stateTensor = Ort::Value::CreateTensor<float>(
                                memInfo, state.data(), state.size(),
                                stateShape.data(), stateShape.size());
                        Ort::Value srTensor = Ort::Value::CreateTensor<int64_t>(
                                memInfo, srBuf.data(), srBuf.size(),
                                srShapePtr, 0);

                        std::array<Ort::Value, 3> ins = {
                            std::move(inputTensor),
                            std::move(stateTensor),
                            std::move(srTensor),
                        };
                        auto outs = session->Run(Ort::RunOptions{nullptr},
                                                 inputNames, ins.data(), ins.size(),
                                                 outputNames, 2);
                        const float prob = *outs[0].GetTensorData<float>();
                        const float* newState = outs[1].GetTensorData<float>();
                        std::copy(newState, newState + state.size(), state.begin());
                        std::copy(inputBuf.end() - kContext, inputBuf.end(), context.begin());

                        const bool isVoice = prob > voiceThreshold;
                        const int64_t timestampMs = (off * 1000) / kSr;
                        if (windowsProcessed < 10 || windowsProcessed % 25 == 0) {
                            LOG(INFO) << "oird: vad window=" << windowsProcessed
                                      << " t=" << timestampMs << "ms prob=" << prob;
                        }
                        cb->onState(isVoice, timestampMs);
                        ++windowsProcessed;
                    }
                    terminal = [cb]() { cb->onComplete(); };
                } catch (const Ort::Exception& e) {
                    std::string msg = std::string("vad inference failed: ") + e.what();
                    terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
                }
            }
        done:
            guard->release();  // explicit early release; matches v0.6.8 ordering.
            if (terminal) terminal();
            LOG(INFO) << "oird: vad handle=" << modelHandle
                      << " samples=" << sampleCount
                      << " windows=" << windowsProcessed;
        });
    return ::ndk::ScopedAStatus::ok();
}

} // namespace oird
