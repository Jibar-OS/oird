// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// backend/vlm_backend.cpp — VlmBackend method bodies.
// Drives vision.describe via libmtmd on top of llama.cpp.

#include "backend/vlm_backend.h"

#include <chrono>

#include <android-base/logging.h>
#include <mtmd-helper.h>

#include "common/error_codes.h"
#include "image_decode.h"
#include "pool/context_pool.h"
#include "runtime/model_resource.h"

namespace oird {

using aidl::com::android::server::oir::IOirWorkerCallback;

::ndk::ScopedAStatus VlmBackend::loadVlm(const std::string& clipPath,
                             const std::string& llmPath,
                             int64_t* _aidl_return) {
    // v0.6.9: mRt.mLock shrunk around slow ctor (LLaVA-1.5-7b llama_model_load
    // + pool of mtmd_init is ~5-8s on cvd CPU; blocking mRt.mLock that long
    // stalled every other OIRService submit path during H6 validation).
    const std::string combined = clipPath + "|" + llmPath;
    const std::string key = "vlm:" + combined;
    std::unique_lock<std::mutex> lk(mRt.mLock);

    for (auto& [h, m] : mRt.mModels) {
        if (m.path == combined && m.isVlm) {
            LOG(INFO) << "oird: VLM already loaded handle=" << h;
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

    // Budget check (sum of both files)
    const int64_t newSize = fileSizeBytes(clipPath) + fileSizeBytes(llmPath);
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
    const int32_t vlmNCtx   = mVisionDescribeNCtx;
    const int32_t vlmNBatch = mVisionDescribeNBatch;
    const int32_t poolSize  = std::max(1, mVisionDescribeContextsPerModel);
    mRt.mBudget.addResident(newSize);

    lk.unlock();

    LOG(INFO) << "oird: loading VLM clip=" << clipPath << " llm=" << llmPath;

    const int32_t totalCores = std::max(2, (int32_t)sysconf(_SC_NPROCESSORS_ONLN));
    int32_t threads = std::max(1, totalCores / poolSize);

    // v0.5 V1: llm-first because mtmd_init_from_file binds the image
    // encoder against the llama_model.
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    llama_model* model = llama_model_load_from_file(llmPath.c_str(), mparams);
    if (!model) {
        LOG(ERROR) << "oird: llama_model_load_from_file failed for " << llmPath;
        lk.lock();
        mRt.mBudget.subResident(newSize);
        mRt.mLoadRegistry.publish(lk, key, slot, 0, W_MODEL_ERROR, "VLM text-model load failed");
        return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(
                W_MODEL_ERROR, "VLM text-model load failed");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = vlmNCtx;
    cparams.n_batch = vlmNBatch;
    cparams.n_ubatch = vlmNBatch;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;

    mtmd_context_params mtmdParams = mtmd_context_params_default();
    mtmdParams.use_gpu = false;
    mtmdParams.print_timings = false;
    mtmdParams.n_threads = threads;

    std::vector<PooledContext> pooledCtxs;
    pooledCtxs.reserve(poolSize);
    auto cleanup_and_fail = [&](const char* what, int32_t code, const std::string& msg) {
        for (auto& p : pooledCtxs) {
            if (p.mctx) mtmd_free(p.mctx);
            if (p.ctx)  llama_free(p.ctx);
        }
        llama_model_free(model);
        lk.lock();
        mRt.mBudget.subResident(newSize);
        mRt.mLoadRegistry.publish(lk, key, slot, 0, code, msg);
        LOG(ERROR) << "oird: VLM " << what << " failed";
        return ::ndk::ScopedAStatus::fromServiceSpecificErrorWithMessage(code, msg.c_str());
    };
    for (int32_t i = 0; i < poolSize; ++i) {
        llama_context* c = llama_init_from_model(model, cparams);
        if (!c) {
            return cleanup_and_fail("llama_init_from_model", W_MODEL_ERROR,
                    "VLM context init failed");
        }
        mtmd_context* mctx = mtmd_init_from_file(clipPath.c_str(), model, mtmdParams);
        if (!mctx) {
            llama_free(c);
            return cleanup_and_fail("mtmd_init_from_file", W_MODEL_ERROR, "mtmd init failed");
        }
        pooledCtxs.push_back({c, mctx});
    }

    const int64_t kvPerCtx = estimateKvBytesPerContext(model, vlmNCtx);
    const int64_t poolKvBytes = kvPerCtx * poolSize;

    lk.lock();

    const int64_t handle = mRt.mNextModelHandle++;
    const int64_t now = currentTimeMs();
    LoadedModel lm;
    lm.model = model;
    lm.vocab = llama_model_get_vocab(model);
    lm.ctx = nullptr;
    lm.context_size = vlmNCtx;
    lm.n_threads = threads;
    lm.mtmdCtx = nullptr;
    lm.handle = handle;
    lm.path = combined;
    lm.sizeBytes = newSize + poolKvBytes;
    lm.loadTimestampMs = now;
    lm.lastAccessMs = now;
    lm.isVlm = true;
    lm.hasLlamaPool = true;
    mRt.mBudget.addResident(poolKvBytes);
    mRt.mModels[handle] = std::move(lm);
    registerVlmModelResourceLocked(handle);
    mLlama.mPools[handle] = std::make_unique<ContextPool>(std::move(pooledCtxs));

    mRt.mLoadRegistry.publish(lk, key, slot, handle, 0, "");

    *_aidl_return = handle;
    LOG(INFO) << "oird: VLM loaded handle=" << handle
              << " size=" << (newSize >> 20) << "MB"
              << " pool=" << poolSize << " (ctx+mtmd) × "
              << (kvPerCtx >> 20) << "MB KV"
              << " resident=" << (mRt.mBudget.totalBytes() >> 20) << "/" << mRt.mBudget.budgetMb() << "MB";
    return ::ndk::ScopedAStatus::ok();
}

::ndk::ScopedAStatus VlmBackend::submitDescribeImage(int64_t modelHandle,
                                         const std::string& imagePath,
                                         const std::string& prompt,
                                         const std::shared_ptr<IOirWorkerCallback>& cb,
                                         int64_t* _aidl_return) {
    LoadedModel* lmPtr = nullptr;
    std::shared_ptr<InFlightGuard> guard;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto it = mRt.mModels.find(modelHandle);
        if (it == mRt.mModels.end() || !it->second.isVlm) {
            cb->onError(W_INVALID_INPUT, "handle not a VLM");
            *_aidl_return = 0;
            return ::ndk::ScopedAStatus::ok();
        }
        it->second.lastAccessMs = currentTimeMs();
        guard = mRt.acquireInflightLocked(it->second, modelHandle);
        lmPtr = &it->second;
    }
    const int64_t reqHandle = mRt.mNextRequestHandle++;
    *_aidl_return = reqHandle;

    // v0.6.4: enqueue on scheduler.
    mRt.mScheduler->enqueue(mVisionDescribePriority,
        [this, modelHandle, imagePath, prompt, cb, lmPtr, guard]() {
    // v0.6.8: onToken stream stays inside the lease (incremental); only
    // onComplete / onError are deferred past lease + releaseInflight.
    std::function<void()> terminal;
    int outputIndex = 0;
    int64_t totalMs = 0;
    {
    // v0.6 Phase A: lease a VLM slot (pair of llama_ctx + mtmd_ctx).
    // Priority + timeout per vision.describe knobs. Default pool=1 so
    // concurrent describe calls on the same VLM queue up honoring
    // priority order, rather than all piling on a single context.
    ContextPool* pool = nullptr;
    int priority = ContextPool::PRIO_NORMAL;
    std::chrono::milliseconds timeout{60000};
    const llama_vocab* vocab = lmPtr->vocab;
    const int n_batch = mVisionDescribeNBatch;
    {
        std::lock_guard<std::mutex> lk(mRt.mLock);
        auto pit = mLlama.mPools.find(modelHandle);
        if (pit == mLlama.mPools.end()) {
            terminal = [cb]() { cb->onError(W_MODEL_ERROR, "VLM has no context pool"); };
            goto done;
        }
        pool = pit->second.get();
        priority = mVisionDescribePriority;
        timeout = std::chrono::milliseconds(mVisionDescribeAcquireTimeoutMs);
    }
    {
    ContextLease lease(*pool, priority, timeout);
    llama_context* ctx = lease.ctx();
    mtmd_context* mctx = lease.mctx();
    if (!ctx || !mctx) {
        terminal = [cb]() { cb->onError(W_TIMEOUT, "vision.describe pool acquire timed out"); };
        goto done;
    }

    int64_t t0 = currentTimeMs();
    int64_t firstTokenMs = -1;

    llama_memory_clear(llama_get_memory(ctx), true);

    // v0.5 V1: VLM input pipeline rebuilt around libmtmd. Build a
    // composite prompt with the default media marker so the tokenizer
    // splits text + image into separate chunks; mtmd_helper_eval_chunks
    // runs both encode + decode passes against the llama_context.
    const std::string userPrompt = prompt.empty()
            ? std::string("Describe this image in detail.")
            : prompt;
    const std::string composed = std::string("USER: ") + mtmd_default_marker()
            + "\n" + userPrompt + "\nASSISTANT:";

    mtmd_input_text textInput;
    textInput.text          = composed.c_str();
    textInput.add_special   = true;
    textInput.parse_special = true;

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(mctx, imagePath.c_str());
    if (!bmp) {
        std::string msg = "mtmd_helper_bitmap_init_from_file failed: " + imagePath;
        terminal = [cb, msg]() { cb->onError(W_INVALID_INPUT, msg.c_str()); };
        goto done;
    }

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (!chunks) {
        mtmd_bitmap_free(bmp);
        terminal = [cb]() { cb->onError(W_MODEL_ERROR, "mtmd_input_chunks_init failed"); };
        goto done;
    }

    const mtmd_bitmap* bitmaps[1] = { bmp };
    int32_t tkRes = mtmd_tokenize(mctx, chunks, &textInput, bitmaps, 1);
    if (tkRes != 0) {
        std::string msg = "mtmd_tokenize failed: " + std::to_string(tkRes);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
        goto done;
    }

    llama_pos n_past_after = 0;
    int32_t evalRes = mtmd_helper_eval_chunks(
            mctx, ctx, chunks,
            /*n_past=*/0,
            /*seq_id=*/0,
            /*n_batch=*/n_batch,
            /*logits_last=*/true,
            &n_past_after);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    if (evalRes != 0) {
        std::string msg = "mtmd_helper_eval_chunks failed: " + std::to_string(evalRes);
        terminal = [cb, msg]() { cb->onError(W_MODEL_ERROR, msg.c_str()); };
        goto done;
    }
    // n_past is now tracked internally by mtmd via the seq_id; the
    // generation loop below pulls logits via llama_sampler_sample(ctx,-1)
    // and feeds tokens back through llama_decode without us tracking
    // an explicit position counter.
    (void)n_past_after;

    // Generation loop
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // v0.5 V7: OEM-tunable via vision.describe.max_tokens.
    int kMaxTokens;
    { std::lock_guard<std::mutex> lk(mRt.mLock); kMaxTokens = mVisionDescribeMaxTokens; }
    for (int i = 0; i < kMaxTokens; ++i) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;

        char buf[128];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) {
            if (firstTokenMs < 0) firstTokenMs = currentTimeMs() - t0;
            cb->onToken(std::string(buf, n), outputIndex++);
        }

        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, b) != 0) break;
    }

    llama_sampler_free(sampler);
    totalMs = currentTimeMs() - t0;
    const int32_t ftMs = firstTokenMs < 0 ? 0 : (int32_t)firstTokenMs;
    const int capturedIdx = outputIndex;
    const int64_t capturedTotal = totalMs;
    terminal = [cb, capturedIdx, ftMs, capturedTotal]() {
        cb->onComplete(capturedIdx, ftMs, capturedTotal);
    };
    }  // ContextLease released
    }
done:
    guard->release();  // explicit early release; matches v0.6.8 ordering.
    if (terminal) terminal();

    LOG(INFO) << "oird: describe handle=" << modelHandle
              << " tokens=" << outputIndex
              << " totalMs=" << totalMs;
        });  // v0.6.4: close mRt.mScheduler->enqueue lambda
    return ::ndk::ScopedAStatus::ok();
}


// ---- Cross-backend hooks + knob dispatch + resource registration ----

void VlmBackend::eraseModel(int64_t handle) {
    // Caller holds mRt.mLock.
    auto it = mRt.mModels.find(handle);
    if (it == mRt.mModels.end()) return;
    if (it->second.mtmdCtx) {
        mtmd_free(it->second.mtmdCtx);
        it->second.mtmdCtx = nullptr;
    }
    // The llama_context + llama_model are freed by LlamaBackend::eraseModel
    // (it inspects isVlm and skips for non-VLM, but for VLMs we want them
    // freed since this handle is going away). Do them here:
    if (it->second.ctx) {
        llama_free(it->second.ctx);
        it->second.ctx = nullptr;
    }
    if (it->second.model) {
        llama_model_free(it->second.model);
        it->second.model = nullptr;
    }
    // Pool entry lives in mLlama.mPools — erase via LlamaBackend.
    mLlama.mPools.erase(handle);
}

bool VlmBackend::setKnobFloat(const std::string& key, float value) {
    if      (key == "vision.describe.n_ctx")      { mVisionDescribeNCtx = (int32_t)value; return true; }
    else if (key == "vision.describe.n_batch")    { mVisionDescribeNBatch = (int32_t)value; return true; }
    else if (key == "vision.describe.max_tokens") { mVisionDescribeMaxTokens = (int32_t)value; return true; }
    else if (key == "vision.describe.contexts_per_model") {
        int32_t n = (int32_t)value; if (n < 1) n = 1; if (n > 16) n = 16;
        mVisionDescribeContextsPerModel = n; return true;
    }
    else if (key == "vision.describe.acquire_timeout_ms") {
        int32_t n = (int32_t)value; if (n < 100) n = 100;
        mVisionDescribeAcquireTimeoutMs = n; return true;
    }
    else if (key == "vision.describe.priority") {
        mVisionDescribePriority = (int32_t)value; return true;
    }
    return false;
}

void VlmBackend::registerVlmModelResourceLocked(int64_t handle) {
    mRt.registerResourceLocked(std::make_unique<ModelResource>(
        mRt, handle,
        [this](int64_t h, LoadedModel& /*m*/) { eraseModel(h); }
    ));
}

} // namespace oird
