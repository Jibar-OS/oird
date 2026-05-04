// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// backend/whisper_backend.cpp — WhisperBackend method bodies.
// Drives audio.transcribe (whisper.cpp).

#include "backend/whisper_backend.h"

#include <fstream>

#include <android-base/logging.h>
#include <whisper.h>

#include "common/error_codes.h"
#include "pool/whisper_pool.h"
#include "runtime/model_resource.h"
#include "runtime/runtime.h"

namespace oird {

using aidl::com::android::server::oir::IOirWorkerCallback;

namespace {

// 16-bit mono 16 kHz WAV → std::vector<float>. Minimal parser:
// handles "fmt " + "data" and skips LIST/INFO chunks. Returns false
// on any non-conforming format.
bool readWav16(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char riff[12];
    f.read(riff, 12);
    if (f.gcount() != 12) return false;
    if (std::string(riff, 4) != "RIFF" || std::string(riff + 8, 4) != "WAVE") return false;
    int16_t channels = 0; int32_t sampleRate = 0; int16_t bits = 0;
    std::vector<char> data;
    while (f) {
        char id[4]; int32_t sz = 0;
        f.read(id, 4); if (f.gcount() != 4) break;
        f.read((char*)&sz, 4); if (f.gcount() != 4) break;
        std::string chunkId(id, 4);
        if (chunkId == "fmt ") {
            std::vector<char> fmt(sz);
            f.read(fmt.data(), sz);
            if (sz < 16) return false;
            channels   = *(int16_t*)(fmt.data() + 2);
            sampleRate = *(int32_t*)(fmt.data() + 4);
            bits       = *(int16_t*)(fmt.data() + 14);
        } else if (chunkId == "data") {
            data.resize(sz);
            f.read(data.data(), sz);
            break;
        } else {
            f.seekg(sz, std::ios::cur);
        }
    }
    if (channels != 1 || sampleRate != 16000 || bits != 16 || data.empty()) return false;
    const int16_t* pcm = (const int16_t*)data.data();
    size_t n = data.size() / 2;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = (float)pcm[i] / 32768.0f;
    return true;
}

} // anonymous namespace

::ndk::ScopedAStatus WhisperBackend::loadWhisper(const std::string& modelPath, int64_t* _aidl_return) {
    // v0.6.9: mRt.mLock shrunk around slow ctor.
    const std::string key = "whisper:" + modelPath;
    std::unique_lock<std::mutex> lk(mRt.mLock);
    for (auto& [h, m] : mRt.mModels) {
        if (m.path == modelPath && m.isWhisper) {
            LOG(INFO) << "oird: whisper already loaded path=" << modelPath << " handle=" << h;
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

    const int32_t poolSize = std::max(1, mAudioTranscribeContextsPerModel);
    const int32_t acquireTimeoutMs = mAudioTranscribeAcquireTimeoutMs;
    mRt.mBudget.addResident(newSize);

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
            mRt.mBudget.subResident(newSize);
            mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, "whisper load failed");
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, "whisper load failed");
        }
        ctxs.push_back(c);
    }

    lk.lock();

    const int64_t handle = mRt.mNextModelHandle++;
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
    mRt.mModels[handle] = std::move(lm);
    registerWhisperModelResourceLocked(handle);

    mPools[handle] = std::make_unique<WhisperPool>(
            std::move(ctxs), acquireTimeoutMs);

    mRt.mLoadRegistry.publish(lk, key, slot, handle, 0, "");

    *_aidl_return = handle;
    LOG(INFO) << "oird: whisper model loaded handle=" << handle << " path=" << modelPath
              << " size=" << (newSize >> 20) << "MB"
              << " pool=" << poolSize
              << " resident=" << (mRt.mBudget.totalBytes() >> 20) << "/" << mRt.mBudget.budgetMb() << "MB";
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus WhisperBackend::submitTranscribe(int64_t modelHandle,
                                      const std::string& audioPath,
                                      const std::shared_ptr<IOirWorkerCallback>& cb,
                                      int64_t* _aidl_return) {
    WhisperPool* pool = nullptr;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isWhisper) {
            cb->onError(W_INVALID_INPUT, "handle not a whisper model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        auto pit = mPools.find(modelHandle);
        if (pit == mPools.end() || !pit->second) {
            cb->onError(W_MODEL_ERROR, "whisper pool missing for handle");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        pool = pit->second.get();
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.3: enqueue the inference body onto the cross-backend
    // scheduler instead of running it inline — unblocks the binder
    // thread and lets audio-priority submits cut ahead of lower-
    // priority ones queued for other backends.
    mRt.mScheduler->enqueue(mAudioTranscribePriority,
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
                    std::lock_guard<std::mutex> lk(mRt.mLock);
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


// ---- Cross-backend hooks + knob dispatch + resource registration ----

void WhisperBackend::eraseModel(int64_t handle) {
    // Caller holds mRt.mLock. erase on absent map keys is a no-op.
    mPools.erase(handle);
    auto it = mRt.mModels.find(handle);
    if (it != mRt.mModels.end()) {
        // wctx is owned by WhisperPool — already freed by erase above.
        // Just null the legacy pointer.
        it->second.wctx = nullptr;
    }
}

bool WhisperBackend::setKnobFloat(const std::string& key, float value) {
    if (key == "audio.transcribe.contexts_per_model") {
        int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 8) n = 8;
        mAudioTranscribeContextsPerModel = n; return true;
    }
    if (key == "audio.transcribe.acquire_timeout_ms") {
        int32_t n = (int32_t)value; if (n < 100) n = 100;
        mAudioTranscribeAcquireTimeoutMs = n; return true;
    }
    if (key == "audio.transcribe.priority") {
        mAudioTranscribePriority = (int32_t)value; return true;
    }
    return false;
}

bool WhisperBackend::setKnobString(const std::string& key, const std::string& value) {
    if (key == "audio.transcribe.whisper_language") {
        mAudioTranscribeLanguage = value; return true;
    }
    return false;
}

void WhisperBackend::registerWhisperModelResourceLocked(int64_t handle) {
    mRt.registerResourceLocked(std::make_unique<ModelResource>(
        mRt, handle,
        [this](int64_t h, LoadedModel& /*m*/) { eraseModel(h); }
    ));
}

} // namespace oird
