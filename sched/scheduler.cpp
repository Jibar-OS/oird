// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// sched/scheduler.cpp — definitions for Scheduler.

#include "sched/scheduler.h"

#include <android-base/logging.h>

namespace oird {

Scheduler::Scheduler(int numWorkers) : mStop(false) {
    if (numWorkers < 1) numWorkers = 1;
    mWorkers.reserve(numWorkers);
    for (int i = 0; i < numWorkers; ++i) {
        mWorkers.emplace_back([this] { workerLoop(); });
    }
}

Scheduler::~Scheduler() {
    {
        std::lock_guard<std::mutex> lk(mMu);
        mStop = true;
    }
    mCv.notify_all();
    for (auto& w : mWorkers) if (w.joinable()) w.join();
}

void Scheduler::enqueue(int32_t priority, Task task) {
    {
        std::lock_guard<std::mutex> lk(mMu);
        mQueue.push({mNextSeq++, priority, std::move(task)});
    }
    mCv.notify_one();
}

int32_t Scheduler::queueDepth() const {
    std::lock_guard<std::mutex> lk(mMu);
    return static_cast<int32_t>(mQueue.size());
}

void Scheduler::workerLoop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(mMu);
            mCv.wait(lk, [this] { return mStop || !mQueue.empty(); });
            if (mStop && mQueue.empty()) return;
            // const_cast because priority_queue::top returns const ref
            // but we want to move out of it. std::priority_queue
            // doesn't expose a non-const top; the standard workaround
            // is this const_cast-then-pop pattern.
            task = std::move(const_cast<Entry&>(mQueue.top()).task);
            mQueue.pop();
        }
        try {
            task();
        } catch (const std::exception& e) {
            LOG(ERROR) << "scheduler task threw: " << e.what();
        } catch (...) {
            LOG(ERROR) << "scheduler task threw unknown exception";
        }
    }
}

} // namespace oird
