// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// sched/scheduler.h — cross-backend priority queue + worker pool.
//
// v0.6.3: cross-backend scheduler.
//
// Before v0.6.3, same-model concurrency was solid (llama ContextPool,
// whisper WhisperPool, ORT inherent thread-safety) but priority only
// applied *within* one backend's pool — a high-priority audio submit
// couldn't jump ahead of queued text submits because they ran on
// different queues.
//
// This scheduler puts every submit onto one global priority-sorted
// ready queue. M worker threads (default = hardware_concurrency()
// clamped to [4, 16]) pull tasks in priority order and run them. Each
// task internally still leases from its backend's pool / calls its
// backend's Run — same-model parallelism and pool-level priority are
// preserved beneath.
//
// Priority convention matches ContextPool::PRIO_* (POSIX niceness —
// lower = higher priority). audio.* submits arrive at PRIO_AUDIO_REALTIME
// (0); text/vision at PRIO_NORMAL (10). Within one priority level,
// submits run FIFO (monotonic seq tiebreaker).
//
// Worker-count sizing: we want more workers than the sum of all pool
// sizes so a pool-bound task never blocks a worker that could
// otherwise pull a runnable task from the queue (head-of-line
// avoidance). Default 8 covers the typical mix of a 4-ctx llama
// pool + 2-ctx whisper + 1-ctx VLM without contention.
//
// Shutdown: scheduler destructor stops the workers; tasks still
// in-flight finish their current iteration then observe mStop.
// Queued-but-not-started tasks are dropped silently — callers see
// the request as cancelled (which matches oird-process-exit semantics).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace oird {

class Scheduler {
public:
    using Task = std::function<void()>;

    struct Entry {
        int64_t seq;
        int32_t priority;
        Task    task;
        bool operator<(const Entry& o) const {
            // std::priority_queue is a max-heap. Invert comparison so
            // lower priority value is popped first; FIFO within same
            // priority via seq.
            if (priority != o.priority) return priority > o.priority;
            return seq > o.seq;
        }
    };

    explicit Scheduler(int numWorkers);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void enqueue(int32_t priority, Task task);

    int32_t queueDepth() const;
    int32_t workerCount() const { return static_cast<int32_t>(mWorkers.size()); }

private:
    void workerLoop();

    mutable std::mutex              mMu;
    std::condition_variable         mCv;
    std::priority_queue<Entry>      mQueue;
    std::vector<std::thread>        mWorkers;
    int64_t                         mNextSeq = 0;
    std::atomic<bool>               mStop;
};

} // namespace oird
