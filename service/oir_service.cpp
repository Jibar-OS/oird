// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// service/oir_service.cpp — out-of-class definitions for the OirdService
// "glue" methods: ctor / dtor / lifecycle (warm / unload / cancel),
// configuration (setConfig / setCapabilityFloat / setCapabilityString),
// dumpsys helpers (dumpRuntimeStats / fileIsReadable / getMemoryStats),
// shared private helpers.
//
// Backend-specific bodies live in backend/{llama,whisper,vlm,ort}.cpp.
//
// v0.7 step 7e: extracted from service/oir_service.h. No semantic change.

#include "service/oir_service.h"

namespace oird {

OirdService::OirdService() {
    llama_backend_init();
    // v0.6.3: build the cross-backend scheduler. hardware_concurrency
    // clamped so single-core dev environments still get 4 workers
    // (avoids head-of-line blocking) and big servers don't spawn 64
    // worker threads unnecessarily.
    unsigned cores = std::thread::hardware_concurrency();
    int workers = static_cast<int>(std::clamp<unsigned>(cores, 4u, 16u));
    mRt.mScheduler = std::make_unique<Scheduler>(workers);
    LOG(INFO) << "oird: llama.cpp backend initialized; scheduler workers=" << workers;
}

OirdService::~OirdService() {
    // Stop the scheduler BEFORE tearing down model state — otherwise
    // an in-flight worker task could deref a freed llama_model while
    // the destructor races it. Scheduler dtor joins all workers.
    mRt.mScheduler.reset();
    std::lock_guard<std::mutex> lk(mRt.mLock);
    // Mirror the LRU-eviction cleanup (see loadOnnx / load / loadEmbed
    // / loadWhisper / loadVlm eviction paths). Process-exit would
    // eventually reclaim these, but tests, restart sequences, and any
    // future non-exit teardown path all want symmetric destruction.
    mLlama.mPools.clear();                         // ContextPool dtor frees every pooled ctx
    mWhisperPools.clear();                       // WhisperPool dtor runs whisper_free per slot
    for (auto& [_h, r] : mOcrRec) delete r.session;
    mOcrRec.clear();
    for (auto& [h, m] : mRt.mModels) {
        if (m.ctx) llama_free(m.ctx);
        if (m.model) llama_model_free(m.model);
        // m.wctx is the legacy single-whisper pointer. v0.6.2 moved
        // whisper state into mWhisperPools above; the pointer here is
        // a dangling-after-pool-clear reference in newer paths and
        // never set in older ones. Do NOT whisper_free it — would
        // double-free against the pool.
        delete m.ortSession;
        if (m.mtmdCtx) mtmd_free(m.mtmdCtx);
    }
    mRt.mModels.clear();
    llama_backend_free();
}

bool OirdService::readWav16(const std::string& path, std::vector<float>& out) {
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

::ndk::ScopedAStatus OirdService::unload(int64_t modelHandle) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    auto it = mRt.mModels.find(modelHandle);
    if (it == mRt.mModels.end()) return ::ndk::ScopedAStatus::ok();
    const std::string path = it->second.path;
    Resource* r = mRt.findModelResourceLocked(modelHandle);
    if (r) {
        int64_t freed = mRt.releaseResourceLocked(r);
        mRt.mBudget.subResident(freed);
    } else {
        // No registered resource (shouldn't happen post-step-F).
        // Fall back to old kitchen-sink tear-down.
        LOG(WARNING) << "oird: unload(" << modelHandle
                     << ") had no registered ModelResource — falling back";
        mWhisperPools.erase(modelHandle);
        mLlama.mPools.erase(modelHandle);
        auto oit = mOcrRec.find(modelHandle);
        if (oit != mOcrRec.end()) { delete oit->second.session; mOcrRec.erase(oit); }
        if (it->second.ctx) llama_free(it->second.ctx);
        if (it->second.model) llama_model_free(it->second.model);
        delete it->second.ortSession;
        if (it->second.mtmdCtx) mtmd_free(it->second.mtmdCtx);
        mRt.mBudget.subResident(it->second.sizeBytes);
        mRt.mModels.erase(it);
    }
    LOG(INFO) << "oird: unloaded handle=" << modelHandle << " path=" << path
              << " resident=" << (mRt.mBudget.totalBytes() >> 20) << "MB";
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::cancel(int64_t requestHandle) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    auto it = mRt.mActiveRequests.find(requestHandle);
    if (it != mRt.mActiveRequests.end()) {
        it->second->store(true);
        LOG(INFO) << "oird: cancel requested for handle=" << requestHandle;
    }
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::setConfig(int32_t memoryBudgetMb, int32_t warmTtlSeconds) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    mRt.mBudget.setBudgetMb(memoryBudgetMb);
    mRt.mWarmTtlSeconds = warmTtlSeconds;
    LOG(INFO) << "oird: config budget=" << memoryBudgetMb << "MB warm_ttl=" << warmTtlSeconds << "s";
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::setCapabilityFloat(const std::string& key, float value) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    if (key == "vision.detect.score_threshold") {
        mDetectScoreThresh = value;
    } else if (key == "vision.detect.iou_threshold") {
        mDetectIouThresh = value;
    } else if (key == "audio.vad.voice_threshold") {
        mVadVoiceThreshold = value;
    } else if (key == "audio.vad.sample_rate_hz") {
        // Float-typed AIDL; cast back to int.
        mVadSampleRateHz = (int32_t)value;
    } else if (key == "audio.vad.window_samples") {
        mVadWindowSamples = (int32_t)value;
    } else if (key == "audio.vad.context_samples") {
        mVadContextSamples = (int32_t)value;
    } else if (key == "text.complete.n_ctx") {
        mTextCompleteNCtx = (int32_t)value;
    } else if (key == "text.complete.max_tokens") {
        mTextCompleteMaxTokens = (int32_t)value;
    } else if (key == "text.embed.n_ctx") {
        mTextEmbedNCtx = (int32_t)value;
    } else if (key == "vision.describe.n_ctx") {
        mVisionDescribeNCtx = (int32_t)value;
    } else if (key == "vision.describe.n_batch") {
        mVisionDescribeNBatch = (int32_t)value;
    } else if (key == "vision.describe.max_tokens") {
        mVisionDescribeMaxTokens = (int32_t)value;
    } else if (key == "vision.embed.input_size") {
        mVisionEmbedInputSize = (int32_t)value;
    } else if (key == "vision.embed.normalize_mean") {
        mVisionEmbedNormMean = value;
    } else if (key == "vision.embed.normalize_std") {
        mVisionEmbedNormStd = value;
    } else if (key == "vision.detect.input_size") {
        mVisionDetectInputSize = (int32_t)value;
    } else if (key == "image.max_pixels") {
        // v0.7 hardening — cap on decoded JPEG/PNG pixel count to
        // protect oird from pathological untrusted images. 0 disables
        // the cap. Default kDefaultMaxImagePixels = 16M (~48 MB RGB).
        mImageMaxPixels = (size_t)value;
    } else if (key == "audio.synthesize.sample_rate_hz") {
        mAudioSynthesizeSampleRate = (int32_t)value;
    } else if (key == "audio.synthesize.length_scale") {
        mAudioSynthesizeLengthScale = value;
    } else if (key == "audio.synthesize.noise_scale") {
        mAudioSynthesizeNoiseScale = value;
    } else if (key == "text.complete.contexts_per_model") {
        // v0.6 Phase A: per-capability pool sizes. Clamped to [1,16]
        // to avoid runaway memory on config typos.
        int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
        mTextCompleteContextsPerModel = n;
    } else if (key == "text.embed.contexts_per_model") {
        int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
        mTextEmbedContextsPerModel = n;
    } else if (key == "vision.describe.contexts_per_model") {
        int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
        mVisionDescribeContextsPerModel = n;
    } else if (key == "text.complete.acquire_timeout_ms") {
        int32_t n = (int32_t)value; if (n < 100) n = 100;
        mTextCompleteAcquireTimeoutMs = n;
    } else if (key == "text.embed.acquire_timeout_ms") {
        int32_t n = (int32_t)value; if (n < 100) n = 100;
        mTextEmbedAcquireTimeoutMs = n;
    } else if (key == "vision.describe.acquire_timeout_ms") {
        int32_t n = (int32_t)value; if (n < 100) n = 100;
        mVisionDescribeAcquireTimeoutMs = n;
    } else if (key == "audio.transcribe.contexts_per_model") {
        int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 8) n = 8;
        mAudioTranscribeContextsPerModel = n;
    } else if (key == "audio.transcribe.acquire_timeout_ms") {
        int32_t n = (int32_t)value; if (n < 100) n = 100;
        mAudioTranscribeAcquireTimeoutMs = n;
    } else if (key == "text.complete.priority") {
        mTextCompletePriority = (int32_t)value;
    } else if (key == "text.embed.priority") {
        mTextEmbedPriority = (int32_t)value;
    } else if (key == "vision.describe.priority") {
        mVisionDescribePriority = (int32_t)value;
    } else if (key == "audio.transcribe.priority") {
        mAudioTranscribePriority = (int32_t)value;
    } else if (key == "audio.vad.priority") {
        mAudioVadPriority = (int32_t)value;
    } else if (key == "audio.synthesize.priority") {
        mAudioSynthesizePriority = (int32_t)value;
    } else if (key == "text.complete.temperature") {
        if (value >= 0.0f && value <= 2.0f) mTextCompleteTemperatureDefault = value;
    } else if (key == "text.complete.top_p") {
        if (value > 0.0f && value <= 1.0f) mTextCompleteTopP = value;
    } else if (key == "llama.batch_size") {
        int32_t n = (int32_t)value; if (n < 32) n = 32; if (n > 4096) n = 4096;
        mLlamaBatchSize = n;
    } else {
        LOG(WARNING) << "oird: unknown capability tuning key " << key
                     << " = " << value << " (ignored)";
        return ::ndk::ScopedAStatus::ok();
    }
    LOG(INFO) << "oird: tuning " << key << " = " << value;
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::warm(int64_t modelHandle) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    auto it = mRt.mModels.find(modelHandle);
    if (it != mRt.mModels.end()) {
        it->second.warmUntilMs = currentTimeMs() + (int64_t)mRt.mWarmTtlSeconds * 1000;
        LOG(INFO) << "oird: warm handle=" << modelHandle
                  << " until=" << it->second.warmUntilMs;
    }
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::dumpRuntimeStats(std::string* _aidl_return) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    std::string out;
    const int64_t MB = 1024 * 1024;
    for (const auto& [h, m] : mRt.mModels) {
        int32_t poolSize = 0, busy = 0, waiting = 0;
        const char* backend = "?";
        if (m.isWhisper) {
            backend = "whisper";
            auto pit = mWhisperPools.find(h);
            if (pit != mWhisperPools.end() && pit->second) {
                poolSize = static_cast<int32_t>(pit->second->size());
                busy     = pit->second->busyCount();
                waiting  = pit->second->waitingCount();
            }
        } else if (m.isVlm) {
            backend = "mtmd";
            auto pit = mLlama.mPools.find(h);
            if (pit != mLlama.mPools.end() && pit->second) {
                poolSize = pit->second->size();
                busy     = pit->second->busyCount();
                waiting  = pit->second->waitingCount();
            }
        } else if (m.isOnnx || m.isVad) {
            backend = "ort";  // ORT has no pool abstraction by design
        } else if (m.isVisionEmbed) {
            backend = "ort";
        } else {
            // llama — text.complete / text.embed / text.translate.
            backend = m.isEmbedding ? "llama_embed" : "llama";
            auto pit = mLlama.mPools.find(h);
            if (pit != mLlama.mPools.end() && pit->second) {
                poolSize = pit->second->size();
                busy     = pit->second->busyCount();
                waiting  = pit->second->waitingCount();
            }
        }
        out += std::to_string(h);
        out += '\t'; out += backend;
        out += '\t'; out += std::to_string(poolSize);
        out += '\t'; out += std::to_string(busy);
        out += '\t'; out += std::to_string(waiting);
        out += '\t'; out += std::to_string(m.sizeBytes / MB);
        out += '\t'; out += m.path;
        out += '\n';
    }
    *_aidl_return = std::move(out);
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::fileIsReadable(const std::string& path, bool* _aidl_return) {
    if (path.empty()) { *_aidl_return = false; return ::ndk::ScopedAStatus::ok(); }
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
        *_aidl_return = false;
        return ::ndk::ScopedAStatus::ok();
    }
    if (!S_ISREG(st.st_mode)) {
        *_aidl_return = false;
        return ::ndk::ScopedAStatus::ok();
    }
    // Regular file + stat succeeded. Check read-openable — the
    // caller's real question is "can inference load this?", which
    // requires open-for-read. Closing immediately; we're not
    // actually loading here.
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { *_aidl_return = false; return ::ndk::ScopedAStatus::ok(); }
    ::close(fd);
    *_aidl_return = true;
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::getMemoryStats(MemoryStats* _aidl_return) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    const int64_t MB = 1024 * 1024;
    int64_t totalBytes = 0;
    std::vector<std::string> paths;
    std::vector<int32_t> sizes;
    std::vector<int64_t> loadTimes;
    std::vector<int64_t> accessTimes;
    for (const auto& [h, m] : mRt.mModels) {
        paths.push_back(m.path);
        sizes.push_back(static_cast<int32_t>(m.sizeBytes / MB));
        loadTimes.push_back(m.loadTimestampMs);
        accessTimes.push_back(m.lastAccessMs);
        totalBytes += m.sizeBytes;
    }
    _aidl_return->modelCount = static_cast<int32_t>(mRt.mModels.size());
    _aidl_return->residentMb = static_cast<int32_t>(totalBytes / MB);
    _aidl_return->modelPaths = std::move(paths);
    _aidl_return->modelSizesMb = std::move(sizes);
    _aidl_return->loadTimestampMs = std::move(loadTimes);
    _aidl_return->lastAccessMs = std::move(accessTimes);
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus OirdService::setCapabilityString(const std::string& key,
                                         const std::string& value) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    if (key == "audio.transcribe.whisper_language") {
        mAudioTranscribeLanguage = value;
    } else if (key == "vision.detect.family") {
        mVisionDetectFamily = value;
    } else if (key == "vision.detect.normalize") {
        mVisionDetectNormalize = value;
    } else {
        LOG(WARNING) << "oird: unknown capability tuning string key " << key
                     << " = " << value << " (ignored)";
        return ::ndk::ScopedAStatus::ok();
    }
    LOG(INFO) << "oird: tuning " << key << " = \"" << value << "\"";
    return ::ndk::ScopedAStatus::ok();
}

void OirdService::cleanupRequest(int64_t handle) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    mRt.mActiveRequests.erase(handle);
}

void OirdService::registerModelResourceLocked(int64_t handle) {
    mRt.registerResourceLocked(std::make_unique<ModelResource>(
        mRt, handle,
        [this](int64_t h, LoadedModel& m) {
            // Kitchen-sink tear-down: frees state across all backends.
            // Until backends are fully extracted (steps 3-5), each
            // model has the same tear-down regardless of which backend
            // loaded it. After extraction, this specializes per-backend.
            mLlama.mPools.erase(h);
            mWhisperPools.erase(h);
            auto oit = mOcrRec.find(h);
            if (oit != mOcrRec.end()) {
                delete oit->second.session;
                mOcrRec.erase(oit);
            }
            if (m.ctx) llama_free(m.ctx);
            if (m.model) llama_model_free(m.model);
            m.wctx = nullptr;
            delete m.ortSession;
            if (m.mtmdCtx) mtmd_free(m.mtmdCtx);
        }
    ));
}

// v0.7-post step 2a: releaseInflight() and mRt.acquireInflightLocked() moved
// to Runtime (defined inline in runtime/runtime.h).

int32_t OirdService::priorityForCapability(const std::string& cap) {
    std::lock_guard<std::mutex> lk(mRt.mLock);
    if (cap == "audio.transcribe")  return mAudioTranscribePriority;
    if (cap == "audio.vad")         return mAudioVadPriority;
    if (cap == "audio.synthesize")  return mAudioSynthesizePriority;
    if (cap == "text.complete"
            || cap == "text.translate") return mTextCompletePriority;
    if (cap == "text.embed"
            || cap == "text.classify"
            || cap == "text.rerank")    return mTextEmbedPriority;
    if (cap == "vision.describe")   return mVisionDescribePriority;
    if (cap == "vision.embed"
            || cap == "vision.detect"
            || cap == "vision.ocr")     return ContextPool::PRIO_NORMAL;
    return ContextPool::PRIO_NORMAL;
}

Ort::SessionOptions OirdService::makeOrtSessionOptions(bool isDetection) const {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(std::max(2, (int)sysconf(_SC_NPROCESSORS_ONLN) / 2));
    so.SetGraphOptimizationLevel(
        isDetection ? GraphOptimizationLevel::ORT_ENABLE_ALL
                    : GraphOptimizationLevel::ORT_ENABLE_BASIC);
    return so;
}

void OirdService::ensureOrtEnv() {
    if (!mOrtEnv) {
        mOrtEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "oird");
    }
}

} // namespace oird
