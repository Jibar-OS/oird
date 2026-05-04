/*
 * Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
 * Licensed under the Apache License, Version 2.0
 *
 * oird — OIR native inference worker. Runs in its own process (not
 * system_server) so native crashes can't take down the platform
 * (DESIGN.md §4.1). Registers the "oir_worker" binder service; only
 * system_server is expected to call in.
 *
 * Uses the AIDL NDK backend (std::string, std::shared_ptr,
 * ndk::ScopedAStatus). main() owns process bringup; everything else
 * lives in service/oir_service.{h,cpp} and the backend/ directory.
 */

#include <algorithm>
#include <thread>

#include "service/oir_service.h"

int main(int argc, char** argv) {
    // Footgun guard: oird is a binder daemon, not a CLI. Anyone who
    // runs `adb shell oird capabilities` (or any bare argv-passing
    // invocation) starts a SECOND daemon that registers as `oir_worker`
    // and races the init-started instance — servicemanager warns, and
    // the original instance's in-flight calls stall. Only init should
    // exec oird, and init doesn't pass argv. Reject anything else.
    if (argc > 1) {
        fprintf(stderr,
                "oird: daemon has no CLI; use `cmd oir <subcommand>` instead.\n"
                "See `cmd oir help` for supported subcommands.\n");
        return 2;
    }
    android::base::InitLogging(nullptr);
    LOG(INFO) << "oird starting up";

    std::shared_ptr<oird::OirdService> service =
            ndk::SharedRefBase::make<oird::OirdService>();

    binder_status_t status = AServiceManager_addService(
            service->asBinder().get(), "oir_worker");
    if (status != STATUS_OK) {
        LOG(ERROR) << "oird: AServiceManager_addService failed status=" << status;
        return 1;
    }
    LOG(INFO) << "oird: registered service \"oir_worker\"";

    // Binder dispatch thread pool size. Most AIDL calls return fast
    // (enqueue to Scheduler, return) so a small pool is plenty: the
    // actual inference happens on Scheduler workers, not these threads.
    // Scale lightly with cores and cap at 8 — past that, binder threads
    // sit idle while Scheduler is the real bottleneck. Floor of 4 keeps
    // single-core dev environments responsive. Must be set before
    // startThreadPool(), so it can't be a runtime knob.
    const unsigned cores = std::thread::hardware_concurrency();
    const uint32_t binderThreads = std::clamp<unsigned>(cores / 2, 4u, 8u);
    ABinderProcess_setThreadPoolMaxThreadCount(binderThreads);
    LOG(INFO) << "oird: binder thread pool size=" << binderThreads
              << " (hardware_concurrency=" << cores << ")";
    ABinderProcess_startThreadPool();
    ABinderProcess_joinThreadPool();
    return 0;
}
