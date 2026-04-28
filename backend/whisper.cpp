// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/whisper.cpp — out-of-class definitions for OirdService methods
// that drive the whisper.cpp backend (audio.transcribe).
//
// v0.7 step 7: extracted from service/oir_service.h. No semantic change;
// pure relocation. The class declaration remains in service/oir_service.h.

#include "service/oir_service.h"

namespace oird {

::ndk::ScopedAStatus OirdService::loadWhisper(const std::string& modelPath, int64_t* _aidl_return) {
    // v0.6.9: mLock shrunk around slow ctor.
    const std::string key = "whisper:" + modelPath;
    std::unique_lock<std::mutex> lk(mLock);
    for (auto& [h, m] : mModels) {
        if (m.path == modelPath && m.isWhisper) {
            LOG(INFO) << "oird: whisper already loaded path=" << modelPath << " handle=" << h;
            *_aidl_return = h;
            return ::ndk::ScopedAStatus::ok();
        }
    }

    auto claim = mLoadRegistry.claim(lk, key);
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
    if (mBudget.budgetMb() > 0 && (mBudget.totalBytes() + newSize) > mBudget.budgetBytes()) {
        const int64_t budgetBytes = mBudget.budgetBytes();
        const int64_t now = currentTimeMs();
        std::vector<std::pair<int64_t, int64_t>> candidates;
        for (const auto& [h, m] : mModels) {
            if (m.inFlightCount > 0) continue;
            if (m.warmUntilMs > now) continue;
            candidates.emplace_back(m.lastAccessMs, h);
        }
        std::sort(candidates.begin(), candidates.end());
        int64_t freed = 0;
        for (const auto& [_ts, h] : candidates) {
            if (mBudget.totalBytes() + newSize - freed <= budgetBytes) break;
            auto it = mModels.find(h);
            if (it == mModels.end()) continue;
            mLlamaPools.erase(h);
            {
                auto oit = mOcrRec.find(h);
                if (oit != mOcrRec.end()) {
                    delete oit->second.session;
                    mOcrRec.erase(oit);
                }
            }
            if (it->second.ctx) llama_free(it->second.ctx);
            if (it->second.model) llama_model_free(it->second.model);
            mWhisperPools.erase(h);
            it->second.wctx = nullptr;
            freed += it->second.sizeBytes;
            LOG(INFO) << "oird: evicted handle=" << h << " path=" << it->second.path;
            mModels.erase(it);
            mBudget.recordEviction();
        }
        mBudget.subResident(freed);
        if (mBudget.totalBytes() + newSize > budgetBytes) {
            const std::string msg = "budget exceeded; nothing evictable";
            mLoadRegistry.publish(lk, key, slot, 0, W_INSUFFICIENT_MEMORY, msg);
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_INSUFFICIENT_MEMORY, msg.c_str());
        }
    }

    const int32_t poolSize = std::max(1, mAudioTranscribeContextsPerModel);
    const int32_t acquireTimeoutMs = mAudioTranscribeAcquireTimeoutMs;
    mBudget.addResident(newSize);

    lk.unlock();

    // v0.6.2: build a whisper_context pool. Each context mmaps the same
    // weights file — kernel COW-shares pages so actual extra cost is
    // per-ctx state (tens of MB each for whisper-tiny).
    whisper_context_params wp = whisper_context_default_params();
    wp.use_gpu = false;

    std::vector<whisper_context*> ctxs;
    ctxs.reserve(poolSize);
    for (int i = 0; i < poolSize; ++i) {
        whisper_context* c = whisper_init_from_file_with_params(modelPath.c_str(), wp);
        if (!c) {
            LOG(ERROR) << "oird: whisper_init_from_file_with_params failed for "
                       << modelPath << " (pool ctx " << i << "/" << poolSize << ")";
            for (auto* prev : ctxs) whisper_free(prev);
            lk.lock();
            mBudget.subResident(newSize);
            mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, "whisper load failed");
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, "whisper load failed");
        }
        ctxs.push_back(c);
    }

    lk.lock();

    const int64_t handle = mNextModelHandle++;
    const int64_t now = currentTimeMs();
    LoadedModel lm;
    // Legacy lm.wctx kept pointing at the first ctx for any v0.5-era
    // non-pooled path; submitTranscribe leases from the pool, not this.
    lm.wctx = ctxs.front();
    lm.handle = handle;
    lm.path = modelPath;
    lm.sizeBytes = newSize;
    lm.loadTimestampMs = now;
    lm.lastAccessMs = now;
    lm.isWhisper = true;
    // newSize already reserved pre-slow-ctor.
    mModels[handle] = std::move(lm);

    mWhisperPools[handle] = std::make_unique<WhisperPool>(
            std::move(ctxs), acquireTimeoutMs);

    mLoadRegistry.publish(lk, key, slot, handle, 0, "");

    *_aidl_return = handle;
    LOG(INFO) << "oird: whisper model loaded handle=" << handle << " path=" << modelPath
              << " size=" << (newSize >> 20) << "MB"
              << " pool=" << poolSize
              << " resident=" << (mBudget.totalBytes() >> 20) << "/" << mBudget.budgetMb() << "MB";
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::submitTranscribe(int64_t modelHandle,
                                      const std::string& audioPath,
                                      const std::shared_ptr<IOirWorkerCallback>& cb,
                                      int64_t* _aidl_return) {
    WhisperPool* pool = nullptr;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mLock);
        auto it = mModels.find(modelHandle);
        if (it == mModels.end() || !it->second.isWhisper) {
            cb->onError(W_INVALID_INPUT, "handle not a whisper model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        auto pit = mWhisperPools.find(modelHandle);
        if (pit == mWhisperPools.end() || !pit->second) {
            cb->onError(W_MODEL_ERROR, "whisper pool missing for handle");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        pool = pit->second.get();
        it->second.lastAccessMs = currentTimeMs();
        guard = acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.3: enqueue the inference body onto the cross-backend
    // scheduler instead of running it inline — unblocks the binder
    // thread and lets audio-priority submits cut ahead of lower-
    // priority ones queued for other backends.
    mScheduler->enqueue(priorityForCapability("audio.transcribe"),
        [this, modelHandle, audioPath, cb, pool, guard]() {
            // v0.6.8: release WhisperLease + inflight BEFORE firing the
            // terminal callback. cb->onToken mid-flight must stay inside
            // the lease scope (whisper's segment callback drives it);
            // only onComplete/onError are deferred.
            std::function<void()> terminal;
            int64_t waitMs = 0;
            size_t pcmSize = 0;
            int n_segments = 0;
            int64_t wall = 0;
            {
                std::vector<float> pcm;
                if (!readWav16(audioPath, pcm)) {
                    terminal = [cb, audioPath]() {
                        cb->onError(W_INVALID_INPUT, "WAV must be 16-bit mono 16 kHz: " + audioPath);
                    };
                    goto done;
                }
                pcmSize = pcm.size();

                int slotIdx = -1;
                whisper_context* wctx = pool->acquire(slotIdx, waitMs);
                if (!wctx) {
                    const int timeoutMs = mAudioTranscribeAcquireTimeoutMs;
                    terminal = [cb, timeoutMs]() {
                        cb->onError(W_TIMEOUT,
                            "whisper pool acquire timed out after "
                            + std::to_string(timeoutMs) + "ms");
                    };
                    goto done;
                }
                WhisperLease lease(pool, wctx, slotIdx, waitMs);
                if (waitMs > 100) {
                    LOG(INFO) << "oird: transcribe lease wait=" << waitMs << "ms";
                }

                std::string whisperLang;
                {
                    std::lock_guard<std::mutex> lk(mLock);
                    whisperLang = mAudioTranscribeLanguage;
                }
                whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                wp.print_progress = false;
                wp.print_special = false;
                wp.print_realtime = false;
                wp.translate = false;
                wp.language = whisperLang.c_str();
                wp.single_segment = false;

                struct CbCtx { std::shared_ptr<IOirWorkerCallback> cb; };
                CbCtx ctx = { cb };
                wp.new_segment_callback = [](whisper_context* c, whisper_state* /*unused*/, int n_new, void* user) {
                    auto* ctx = (CbCtx*)user;
                    int n_segments = whisper_full_n_segments(c);
                    for (int i = n_segments - n_new; i < n_segments; i++) {
                        const char* text = whisper_full_get_segment_text(c, i);
                        if (text && *text) ctx->cb->onToken(std::string(text), i);
                    }
                };
                wp.new_segment_callback_user_data = &ctx;

                int64_t t0 = currentTimeMs();
                int rc = whisper_full(wctx, wp, pcm.data(), (int)pcm.size());
                int64_t t1 = currentTimeMs();
                wall = t1 - t0;

                if (rc != 0) {
                    terminal = [cb, rc]() {
                        cb->onError(W_MODEL_ERROR, "whisper_full failed rc=" + std::to_string(rc));
                    };
                    goto done;
                }

                n_segments = whisper_full_n_segments(wctx);
                terminal = [cb, n_segments, wall]() {
                    cb->onComplete(n_segments, 0, wall);
                };
            }
        done:
            // WhisperLease dtor fired at the scope close above.
            guard->release();  // explicit early release; matches v0.6.8 ordering.
            if (terminal) terminal();

            LOG(INFO) << "oird: transcribe handle=" << modelHandle
                      << " audio_s=" << (pcmSize / 16000)
                      << " segments=" << n_segments
                      << " wait_ms=" << waitMs
                      << " wall_ms=" << wall;
        });
    return ::ndk::ScopedAStatus::ok();
}

} // namespace oird
