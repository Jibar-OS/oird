/*
 * Copyright (C) 2026 The OpenIntelligenceRuntime Project
 * Licensed under the Apache License, Version 2.0
 *
 * oird — OIR native inference worker. Runs in its own process (not
 * system_server) so native crashes can't take down the platform
 * (DESIGN.md §4.1). Registers the "oir_worker" binder service; only
 * system_server is expected to call in.
 *
 * Uses the AIDL NDK backend (std::string, std::shared_ptr, ndk::ScopedAStatus)
 * rather than the legacy CPP backend (String16, sp<>, android::binder::Status).
 *
 * Inference loop adapted from AAOSP's llm_jni.cpp (b4547 API). AAOSP ran
 * inside system_server via JNI; OIR inlines the same logic into this
 * native worker — no JNI needed.
 *
 * v0.7 step 6: this file used to hold the entire OirdService class +
 * supporting state (~4100 lines). It now only contains main() and the
 * binder lifecycle. Class declaration + bodies live in service/oir_service.h.
 * Step 7+ will split bodies out into backend/{llama,whisper,vlm,ort}.cpp.
 */

#include "service/oir_service.h"

int main(int argc, char** argv) {
    // v0.6.9 footgun guard: oird is a binder daemon, not a CLI. Anyone
    // who runs `adb shell oird capabilities` (or any bare argv-passing
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

    ABinderProcess_setThreadPoolMaxThreadCount(4);
    ABinderProcess_startThreadPool();
    ABinderProcess_joinThreadPool();
    return 0;
}
