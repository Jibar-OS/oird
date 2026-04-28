// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// backend/llama_backend.cpp — LlamaBackend method bodies.
// Drives text.complete, text.embed, text.translate. vision.describe
// lives in vlm_backend.cpp (mtmd wraps llama_context).

#include "backend/llama_backend.h"

#include <chrono>

#include <android-base/logging.h>

#include "common/error_codes.h"
#include "pool/context_pool.h"
#include "runtime/model_resource.h"

namespace oird {

using aidl::com::android::server::oir::IOirWorkerCallback;
using aidl::com::android::server::oir::IOirWorkerVectorCallback;

namespace {

// Inline batch helpers lifted from AAOSP's llm_jni.cpp.
inline void llama_batch_clear_local(struct llama_batch& batch) {
    batch.n_tokens = 0;
}

inline void llama_batch_add_local(
        struct llama_batch& batch,
        llama_token id,
        llama_pos pos,
        const std::vector<llama_seq_id>& seq_ids,
        bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = (int32_t)seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

} // anonymous namespace

::ndk::ScopedAStatus LlamaBackend::load(const std::string& modelPath, int64_t* _aidl_return) {
    // v0.6.9: mRt.mLock is dropped around the slow ctor. See
    // runtime/load_registry.h for the dedup-on-key rationale.
    const std::string key = "llama-gen:" + modelPath;
    std::unique_lock<std::mutex> lk(mRt.mLock);

    // v0.4 S2: idempotent same-path detect — return existing handle if
    // same model already loaded as a generation model.
    for (auto& [h, m] : mRt.mModels) {
        if (m.path == modelPath
                && !m.isEmbedding && !m.isWhisper && !m.isOnnx && !m.isVlm) {
            LOG(INFO) << "oird: model already loaded path=" << modelPath << " handle=" << h;
            *_aidl_return = h;
            return ::ndk::ScopedAStatus::ok();
        }
    }

    // v0.6.9: concurrent-load dedup. If another thread is already loading
    // the same key, wait here instead of racing a duplicate slow ctor.
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

    // v0.4 S2-B: budget check + LRU eviction. Eviction skips in-flight + warmed.
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

    // Snapshot tunables under lock so slow-init uses a consistent set.
    const int32_t kCtxSize = mTextCompleteNCtx;
    const int32_t poolSize = std::max(1, mTextCompleteContextsPerModel);

    // Reserve the file-size share of resident memory up front so a
    // concurrent load of a *different* path sees our pending bytes.
    // KV-cache bytes are added once known (after slow ctor).
    mRt.mBudget.addResident(newSize);

    lk.unlock();

    LOG(INFO) << "oird: loading " << modelPath << " ctx=" << kCtxSize;

    // --- slow ctor, mRt.mLock NOT held ---
    constexpr int32_t kGpuLayers = 0;
    const int32_t totalCores = std::max(2, (int32_t)sysconf(_SC_NPROCESSORS_ONLN));
    int32_t threads = std::max(1, totalCores / poolSize);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = kGpuLayers;
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    llama_model* model = llama_model_load_from_file(modelPath.c_str(), mparams);
    if (!model) {
        LOG(ERROR) << "oird: llama_model_load_from_file failed for " << modelPath;
        lk.lock();
        mRt.mBudget.subResident(newSize);
        mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, "model load failed");
        return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                W_MODEL_ERROR, "model load failed");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = kCtxSize;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;

    std::vector<PooledContext> pooledCtxs;
    pooledCtxs.reserve(poolSize);
    for (int32_t i = 0; i < poolSize; ++i) {
        llama_context* c = llama_init_from_model(model, cparams);
        if (!c) {
            LOG(ERROR) << "oird: llama_init_from_model failed at pool slot " << i;
            for (auto& p : pooledCtxs) llama_free(p.ctx);
            llama_model_free(model);
            lk.lock();
            mRt.mBudget.subResident(newSize);
            mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, "context init failed");
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, "context init failed");
        }
        pooledCtxs.push_back({c, nullptr});
    }

    const int64_t kvPerCtx = estimateKvBytesPerContext(model, kCtxSize);
    const int64_t poolKvBytes = kvPerCtx * poolSize;

    // --- re-lock to insert ---
    lk.lock();

    const int64_t handle = mRt.mNextModelHandle++;
    const int64_t now = currentTimeMs();
    LoadedModel lm;
    lm.model = model;
    lm.vocab = llama_model_get_vocab(model);
    lm.ctx = nullptr;
    lm.context_size = kCtxSize;
    lm.n_threads = threads;
    lm.handle = handle;
    lm.path = modelPath;
    lm.sizeBytes = newSize + poolKvBytes;
    lm.loadTimestampMs = now;
    lm.lastAccessMs = now;
    lm.hasLlamaPool = true;
    // newSize was already added to mRt.mBudget at reservation above; only
    // the KV bytes are new here.
    mRt.mBudget.addResident(poolKvBytes);
    mRt.mModels[handle] = std::move(lm);
    registerLlamaModelResourceLocked(handle);
    mPools[handle] = std::make_unique<ContextPool>(std::move(pooledCtxs));

    mRt.mLoadRegistry.publish(lk, key, slot, handle, 0, "");

    *_aidl_return = handle;
    LOG(INFO) << "oird: model loaded handle=" << handle << " path=" << modelPath
              << " size=" << (newSize >> 20) << "MB"
              << " pool=" << poolSize << " ctx × " << (kvPerCtx >> 20) << "MB KV"
              << " = " << (mRt.mModels[handle].sizeBytes >> 20) << "MB total"
              << " resident=" << (mRt.mBudget.totalBytes() >> 20) << "/" << mRt.mBudget.budgetMb() << "MB"
              << " total_models=" << mRt.mModels.size();
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus LlamaBackend::loadEmbed(const std::string& modelPath, int64_t* _aidl_return) {
    // v0.6.9: mRt.mLock shrunk around slow ctor (see load() / runtime/load_registry.h).
    const std::string key = "llama-emb:" + modelPath;
    std::unique_lock<std::mutex> lk(mRt.mLock);

    // Idempotent same-path detect — same file loaded as generation model is a distinct handle.
    for (auto& [h, m] : mRt.mModels) {
        if (m.path == modelPath && m.isEmbedding) {
            LOG(INFO) << "oird: embed model already loaded path=" << modelPath << " handle=" << h;
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

    // Snapshot tunables + reserve budget under lock.
    const int32_t embNCtx = mTextEmbedNCtx;
    const int32_t poolSize = std::max(1, mTextEmbedContextsPerModel);
    mRt.mBudget.addResident(newSize);

    lk.unlock();

    LOG(INFO) << "oird: loading (embed mode) " << modelPath;

    const int32_t totalCores = std::max(2, (int32_t)sysconf(_SC_NPROCESSORS_ONLN));
    int32_t threads = std::max(1, totalCores / poolSize);

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    llama_model* model = llama_model_load_from_file(modelPath.c_str(), mparams);
    if (!model) {
        LOG(ERROR) << "oird: llama_model_load_from_file failed for " << modelPath;
        lk.lock();
        mRt.mBudget.subResident(newSize);
        mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, "embed model load failed");
        return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                W_MODEL_ERROR, "embed model load failed");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = embNCtx;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;

    std::vector<PooledContext> pooledCtxs;
    pooledCtxs.reserve(poolSize);
    for (int32_t i = 0; i < poolSize; ++i) {
        llama_context* c = llama_init_from_model(model, cparams);
        if (!c) {
            LOG(ERROR) << "oird: llama_init_from_model (embed) failed at pool slot " << i;
            for (auto& p : pooledCtxs) llama_free(p.ctx);
            llama_model_free(model);
            lk.lock();
            mRt.mBudget.subResident(newSize);
            mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, "embed context init failed");
            return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                    W_MODEL_ERROR, "embed context init failed");
        }
        pooledCtxs.push_back({c, nullptr});
    }

    const int64_t kvPerCtx = estimateKvBytesPerContext(model, embNCtx);
    const int64_t poolKvBytes = kvPerCtx * poolSize;
    const int32_t nEmbd = llama_n_embd(model);

    lk.lock();

    const int64_t handle = mRt.mNextModelHandle++;
    const int64_t now = currentTimeMs();
    LoadedModel lm;
    lm.model = model;
    lm.vocab = llama_model_get_vocab(model);
    lm.ctx = nullptr;
    lm.context_size = embNCtx;
    lm.n_threads = threads;
    lm.handle = handle;
    lm.path = modelPath;
    lm.sizeBytes = newSize + poolKvBytes;
    lm.loadTimestampMs = now;
    lm.lastAccessMs = now;
    lm.isEmbedding = true;
    lm.hasLlamaPool = true;
    mRt.mBudget.addResident(poolKvBytes);
    mRt.mModels[handle] = std::move(lm);
    registerLlamaModelResourceLocked(handle);
    mPools[handle] = std::make_unique<ContextPool>(std::move(pooledCtxs));

    mRt.mLoadRegistry.publish(lk, key, slot, handle, 0, "");

    *_aidl_return = handle;
    LOG(INFO) << "oird: embed model loaded handle=" << handle << " path=" << modelPath
              << " size=" << (newSize >> 20) << "MB"
              << " n_embd=" << nEmbd
              << " pool=" << poolSize << " ctx × " << (kvPerCtx >> 20) << "MB KV"
              << " resident=" << (mRt.mBudget.totalBytes() >> 20) << "/" << mRt.mBudget.budgetMb() << "MB";
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus LlamaBackend::submitEmbed(int64_t modelHandle,
                                 const std::string& text,
                                 const std::shared_ptr<IOirWorkerVectorCallback>& cb,
                                 int64_t* _aidl_return) {
    LoadedModel* lmPtr = nullptr;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isEmbedding) {
            cb->onError(W_INVALID_INPUT, "handle not an embedding model");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
        lmPtr = &it->second;
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue inference body on the cross-backend scheduler.
    // lmPtr stays valid because the InFlightGuard captured in the lambda
    // holds inFlightCount > 0, blocking LRU eviction.
    mRt.mScheduler->enqueue(mTextEmbedPriority,
        [this, modelHandle, text, cb, lmPtr, guard]() {
            // v0.6.8: hold lease + inflight across inference ONLY; fire
            // terminal callback AFTER releasing both. Prior revisions
            // called cb->onVector / cb->onError while the ContextLease
            // was still in scope — a stalled binder callback (dead app,
            // binder thread contention) would pin the pool slot forever,
            // and any subsequent load() needing to evict would hang on
            // ContextPool::shutdown() waiting for the unreleasable lease.
            std::function<void()> terminal;
            {
                // Tokenize — BOS=true, add_special=true (BERT-style)
                std::vector<llama_token> tokens(text.size() + 8);
                int n = llama_tokenize(lmPtr->vocab, text.c_str(), (int)text.size(),
                                       tokens.data(), (int)tokens.size(), true, true);
                if (n < 0) {
                    tokens.resize(-n);
                    n = llama_tokenize(lmPtr->vocab, text.c_str(), (int)text.size(),
                                       tokens.data(), (int)tokens.size(), true, true);
                }
                if (n <= 0) {
                    terminal = [cb]() { cb->onError(W_INVALID_INPUT, "tokenize failed"); };
                    goto done;
                }
                tokens.resize(n);

                ContextPool* pool = nullptr;
                int priority = ContextPool::PRIO_NORMAL;
                std::chrono::milliseconds timeout{10000};
                {
                    std::lock_guard<std::mutex> lk(mRt.mLock);
                    auto pit = mPools.find(modelHandle);
                    if (pit == mPools.end()) {
                        terminal = [cb]() { cb->onError(W_MODEL_ERROR, "embed model has no context pool"); };
                        goto done;
                    }
                    pool = pit->second.get();
                    priority = mTextEmbedPriority;
                    timeout = std::chrono::milliseconds(mTextEmbedAcquireTimeoutMs);
                }
                ContextLease lease(*pool, priority, timeout);
                llama_context* ectx = lease.ctx();
                if (!ectx) {
                    terminal = [cb]() { cb->onError(W_TIMEOUT, "embed pool acquire timed out"); };
                    goto done;
                }

                llama_memory_clear(llama_get_memory(ectx), true);
                llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
                for (int i = 0; i < (int)tokens.size(); ++i) {
                    batch.token[i] = tokens[i];
                    batch.pos[i] = i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = (i == (int)tokens.size() - 1);
                }
                batch.n_tokens = (int)tokens.size();

                int rc = llama_decode(ectx, batch);
                if (rc != 0) {
                    llama_batch_free(batch);
                    terminal = [cb]() { cb->onError(W_MODEL_ERROR, "embed decode failed"); };
                    goto done;
                }

                const float* embeds = llama_get_embeddings_seq(ectx, 0);
                if (!embeds) {
                    embeds = llama_get_embeddings_ith(ectx, (int)tokens.size() - 1);
                }
                if (!embeds) {
                    llama_batch_free(batch);
                    terminal = [cb]() { cb->onError(W_MODEL_ERROR, "no embeddings returned"); };
                    goto done;
                }

                int n_embd = llama_n_embd(lmPtr->model);
                std::vector<float> vec(embeds, embeds + n_embd);

                double sum2 = 0.0;
                for (float v : vec) sum2 += (double)v * (double)v;
                float norm = (float)std::sqrt(sum2);
                if (norm > 1e-8f) {
                    for (float& v : vec) v /= norm;
                }

                llama_batch_free(batch);
                terminal = [cb, vec = std::move(vec)]() { cb->onVector(vec); };

                LOG(INFO) << "oird: embed handle=" << modelHandle << " text_tokens=" << tokens.size()
                          << " n_embd=" << n_embd;
            }
        done:
            // Lease dtor fires at the closing brace above (pool slot released).
            guard->release();  // explicit early release; matches v0.6.8 ordering.
            if (terminal) terminal();
        });
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus LlamaBackend::submitTranslate(int64_t modelHandle,
                                      const std::string& prompt,
                                      int32_t maxTokens,
                                      const std::shared_ptr<IOirWorkerCallback>& cb,
                                      int64_t* _aidl_return) {
    if (maxTokens <= 0) maxTokens = 512;
    // Low temperature gives stable, close-to-greedy translations.
    return submit(modelHandle, prompt, maxTokens, /*temperature=*/0.2f, cb, _aidl_return);
}

::ndk::ScopedAStatus LlamaBackend::submit(int64_t modelHandle,
                            const std::string& prompt,
                            int32_t maxTokens,
                            float temperature,
                            const std::shared_ptr<IOirWorkerCallback>& callback,
                            int64_t* _aidl_return) {
    if (callback == nullptr) {
        return ::ndk::ScopedAStatus::fromExceptionCode(EX_NULL_POINTER);
    }

    int64_t handle;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        // v0.6 Phase A: check for a live pool (was: !it->second.ctx).
        if (it == mRt.mModels.end() || !it->second.model
                || !it->second.hasLlamaPool
                || mPools.find(modelHandle) == mPools.end()) {
            callback->onError(W_MODEL_ERROR, "unknown modelHandle");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
        handle = mRt.mNextRequestHandle++;
        mRt.mActiveRequests[handle] = std::make_shared<std::atomic_bool>(false);
    }

    if (maxTokens <= 0) maxTokens = 256;
    if (temperature < 0.0f) temperature = 0.7f;

    LOG(INFO) << "oird: submit handle=" << handle
              << " prompt.len=" << prompt.size()
              << " maxTokens=" << maxTokens << " temp=" << temperature;

    auto cancelled = mRt.mActiveRequests[handle];
    // v0.6.3: cross-backend scheduler. Was a bare `std::thread(...).detach()`;
    // now enqueues at the capability's configured priority so audio-
    // priority submits on a different backend can still jump ahead.
    const int32_t pri = mTextCompletePriority;
    mRt.mScheduler->enqueue(pri,
        [this, modelHandle, handle, prompt, maxTokens, temperature,
         cb = callback, cancelled, guard]() {
            runInference(modelHandle, handle, prompt, maxTokens, temperature,
                         cb, cancelled, guard);
        });

    *_aidl_return = handle;
    return ::ndk::ScopedAStatus::ok();
}

void LlamaBackend::runInference(int64_t modelHandle,
                  int64_t handle,
                  std::string prompt,
                  int32_t maxTokens,
                  float temperature,
                  std::shared_ptr<IOirWorkerCallback> cb,
                  std::shared_ptr<std::atomic_bool> cancelled,
                  std::shared_ptr<InFlightGuard> guard) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // v0.6.8: onToken streams inline (incremental); onComplete / onError
    // deferred past lease release + releaseInflight + cleanupRequest so
    // a stalled callback can't pin the ContextLease and deadlock
    // subsequent load()/eviction calls.
    std::function<void()> terminal;
    {
    // v0.6 Phase A: per-model ContextPool. Lease a context slot with
    // priority + timeout so audio-priority submits jump ahead within
    // this llama pool, and no request hangs forever if the pool
    // wedges. (The v0.6 "cross-backend scheduler" story is separate
    // from pool-local priority — see ROADMAP.md v0.7 entry.)

    llama_model* model;
    const llama_vocab* vocab;
    int32_t ctxSize;
    ContextPool* pool = nullptr;
    int priority = ContextPool::PRIO_NORMAL;
    std::chrono::milliseconds timeout{30000};
    int32_t batchSize;
    float tempDefault;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end()) {
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "model unloaded mid-flight"); };
            goto done;
        }
        model = it->second.model;
        vocab = it->second.vocab;
        ctxSize = it->second.context_size;
        auto pit = mPools.find(modelHandle);
        if (pit != mPools.end()) pool = pit->second.get();
        priority = mTextCompletePriority;
        timeout = std::chrono::milliseconds(mTextCompleteAcquireTimeoutMs);
        batchSize = mLlamaBatchSize;
        tempDefault = mTextCompleteTemperatureDefault;
    }
    if (!model || !vocab || !pool) {
        terminal = [cb]() { cb->onError(W_MODEL_ERROR, "model torn down"); };
        goto done;
    }

    // Priority-aware bounded acquire. Audio submits on a different model
    // don't interact here; audio.*/text.* priority matters only when
    // sharing a pool (not the common case, but correct when it happens).
    {
    ContextLease lease(*pool, priority, timeout);
    llama_context* ctx = lease.ctx();
    if (!ctx) {
        terminal = [cb]() { cb->onError(W_TIMEOUT, "text.complete pool acquire timed out"); };
        goto done;
    }
    // Clear KV from any previous request that used this slot so the new
    // prompt starts fresh. (Prefix caching across requests is a v0.7+
    // perf feature.)
    llama_memory_clear(llama_get_memory(ctx), true);
    // Fold OEM default temperature when caller didn't specify one.
    if (temperature < 0.0f) temperature = tempDefault;

    int32_t n_tokens = llama_tokenize(
            vocab, prompt.c_str(), (int32_t)prompt.size(),
            nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    if (n_tokens < 0) n_tokens = -n_tokens;

    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(
            vocab, prompt.c_str(), (int32_t)prompt.size(),
            tokens.data(), (int32_t)tokens.size(), true, true);
    if (n_tokens < 0) {
        terminal = [cb]() { cb->onError(W_INVALID_INPUT, "tokenize failed"); };
        goto done;
    }
    tokens.resize(n_tokens);

    // v0.5 V7: caller-provided maxTokens (submit AIDL) wins; else the
    // OEM-configured text.complete.max_tokens default.
    int32_t defaultMax;
    { std::lock_guard<std::mutex> lk(mRt.mLock); defaultMax = mTextCompleteMaxTokens; }
    const int32_t maxGen = maxTokens > 0 ? maxTokens : defaultMax;
    if (n_tokens + maxGen > ctxSize) {
        if (n_tokens >= ctxSize) {
            int32_t keep = ctxSize - maxGen - 1;
            if (keep < 1) keep = 1;
            LOG(WARNING) << "oird: prompt=" << n_tokens
                         << " exceeds ctx=" << ctxSize << "; truncating to " << keep;
            tokens.resize(keep);
            n_tokens = keep;
        }
    }

    // v0.6 Phase A: OEM-tunable batch size via llama.batch_size knob.
    llama_batch batch = llama_batch_init(batchSize, 0, 1);
    int32_t n_eval = 0;
    while (n_eval < n_tokens) {
        if (cancelled->load()) {
            terminal = [cb]() { cb->onError(W_CANCELLED, "cancelled during prefill"); };
            llama_batch_free(batch);
            llama_memory_clear(llama_get_memory(ctx), true);
            goto done;
        }
        int32_t this_batch = std::min(batchSize, n_tokens - n_eval);
        llama_batch_clear_local(batch);
        for (int32_t i = 0; i < this_batch; ++i) {
            bool is_last = (n_eval + i == n_tokens - 1);
            llama_batch_add_local(batch, tokens[n_eval + i],
                    n_eval + i, {0}, is_last);
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG(ERROR) << "oird: prefill decode failed at pos=" << n_eval;
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "prefill decode failed"); };
            llama_batch_free(batch);
            llama_memory_clear(llama_get_memory(ctx), true);
            goto done;
        }
        n_eval += this_batch;
    }

    llama_sampler* sampler = llama_sampler_chain_init(
            llama_sampler_chain_default_params());
    // v0.6 Phase A: top_p tunable via text.complete.top_p knob.
    float topP;
    { std::lock_guard<std::mutex> lk(mRt.mLock); topP = mTextCompleteTopP; }
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));
    }

    int64_t firstTokenMs = -1;
    int outputIndex = 0;
    int32_t emitted = 0;
    int32_t n_cur = n_tokens;

    for (int32_t i = 0; i < maxGen; ++i) {
        if (cancelled->load()) {
            LOG(INFO) << "oird: handle=" << handle << " cancelled step " << i;
            terminal = [cb]() { cb->onError(W_CANCELLED, "cancelled"); };
            llama_sampler_free(sampler);
            llama_batch_free(batch);
            llama_memory_clear(llama_get_memory(ctx), true);
            goto done;
        }

        llama_token next = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) {
            LOG(INFO) << "oird: EOG at step " << i;
            break;
        }
        llama_sampler_accept(sampler, next);

        char piece[256];
        int32_t n = llama_token_to_piece(
                vocab, next, piece, sizeof(piece),
                /*lstrip=*/0, /*special=*/false);
        if (n > 0) {
            if (firstTokenMs < 0) {
                firstTokenMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        clock::now() - t0).count();
            }
            cb->onToken(std::string(piece, n), outputIndex++);
            ++emitted;
        }

        llama_batch_clear_local(batch);
        llama_batch_add_local(batch, next, n_cur, {0}, true);
        ++n_cur;
        if (llama_decode(ctx, batch) != 0) {
            LOG(ERROR) << "oird: decode failed at generated token " << i;
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "decode failed mid-stream"); };
            llama_sampler_free(sampler);
            llama_batch_free(batch);
            llama_memory_clear(llama_get_memory(ctx), true);
            goto done;
        }
    }

    int64_t totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            clock::now() - t0).count();
    {
        const int32_t capturedEmitted = emitted;
        const int64_t capturedFirst = firstTokenMs < 0 ? 0 : firstTokenMs;
        const int64_t capturedTotal = totalMs;
        terminal = [cb, capturedEmitted, capturedFirst, capturedTotal]() {
            cb->onComplete(capturedEmitted, capturedFirst, capturedTotal);
        };
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);
    llama_memory_clear(llama_get_memory(ctx), true);
    }  // ContextLease released
    }
done:
    mRt.cleanupRequest(handle);
    if (guard) guard->release();  // v0.7: explicit early release; preserves
                                  // v0.6.8 ordering (inflight released
                                  // BEFORE terminal callback fires).
    if (terminal) terminal();
}


// ---- Cross-backend hooks + knob dispatch + resource registration ----

void LlamaBackend::eraseModel(int64_t handle) {
    // Caller holds mRt.mLock. erase on absent map keys is a no-op.
    mPools.erase(handle);
    auto it = mRt.mModels.find(handle);
    if (it == mRt.mModels.end()) return;
    // VLM models also have ctx + model set, but their cleanup is owned
    // by VlmBackend / OirdService VLM paths. Only free here when the
    // handle is purely llama.
    if (it->second.isVlm) return;
    if (it->second.ctx) { llama_free(it->second.ctx); it->second.ctx = nullptr; }
    if (it->second.model) { llama_model_free(it->second.model); it->second.model = nullptr; }
}

bool LlamaBackend::setKnobFloat(const std::string& key, float value) {
    // Caller holds mRt.mLock.
    if      (key == "text.complete.n_ctx")        { mTextCompleteNCtx = (int32_t)value; return true; }
    else if (key == "text.complete.max_tokens")   { mTextCompleteMaxTokens = (int32_t)value; return true; }
    else if (key == "text.embed.n_ctx")           { mTextEmbedNCtx = (int32_t)value; return true; }
    else if (key == "text.complete.contexts_per_model") {
        int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
        mTextCompleteContextsPerModel = n; return true;
    }
    else if (key == "text.embed.contexts_per_model") {
        int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
        mTextEmbedContextsPerModel = n; return true;
    }
    else if (key == "text.complete.acquire_timeout_ms") {
        int32_t n = (int32_t)value; if (n < 100) n = 100;
        mTextCompleteAcquireTimeoutMs = n; return true;
    }
    else if (key == "text.embed.acquire_timeout_ms") {
        int32_t n = (int32_t)value; if (n < 100) n = 100;
        mTextEmbedAcquireTimeoutMs = n; return true;
    }
    else if (key == "text.complete.priority") { mTextCompletePriority = (int32_t)value; return true; }
    else if (key == "text.embed.priority")    { mTextEmbedPriority = (int32_t)value; return true; }
    else if (key == "text.complete.temperature") {
        if (value >= 0.0f && value <= 2.0f) mTextCompleteTemperatureDefault = value;
        return true;
    }
    else if (key == "text.complete.top_p") {
        if (value > 0.0f && value <= 1.0f) mTextCompleteTopP = value;
        return true;
    }
    else if (key == "llama.batch_size") {
        int32_t n = (int32_t)value; if (n < 32) n = 32; if (n > 4096) n = 4096;
        mLlamaBatchSize = n; return true;
    }
    return false;
}

void LlamaBackend::registerLlamaModelResourceLocked(int64_t handle) {
    mRt.registerResourceLocked(std::make_unique<ModelResource>(
        mRt, handle,
        [this](int64_t h, LoadedModel& m) {
            // Llama-specific tear-down. eraseModel handles ctx/model
            // freeing for non-VLM handles; for VLM handles the OirdService
            // kitchen-sink lambda still runs (registered separately) until
            // VlmBackend extraction in step 5b.
            eraseModel(h);
            (void)m;
        }
    ));
}

} // namespace oird
