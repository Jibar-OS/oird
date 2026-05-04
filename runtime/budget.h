// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// runtime/budget.h — resident-memory accounting for loaded models.
//
// v0.7 step 5: field-only extraction. The eviction loop itself still
// lives in OirdService load handlers (it touches per-backend tear-down:
// mLlamaPools, mWhisperPools, mOcrRec). When backends are extracted
// (steps 6+) the eviction will move into Budget as a callback-driven
// service. For now this class is just a typed wrapper around the three
// fields the daemon used to carry inline.
//
// Thread safety: all methods assume the daemon's mLock is held by the
// caller. Budget itself has no internal lock — it's protected by the
// surrounding mLock the same way the bare fields were.
#pragma once

#include <cstdint>

namespace oird {

class Budget {
public:
    static constexpr int64_t kBytesPerMb = 1024 * 1024;

    // Budget cap in MB. 0 = unlimited (default for safety; OEMs set via
    // OIRService.setConfig()).
    int32_t budgetMb() const { return mBudgetMb; }
    void setBudgetMb(int32_t mb) { mBudgetMb = mb; }

    // Cap converted to bytes (or 0 if unlimited).
    int64_t budgetBytes() const {
        return (int64_t)mBudgetMb * kBytesPerMb;
    }

    // Resident bytes — running total of all loaded models' weight
    // footprints + pool KV. Updated inline at load / unload / evict.
    int64_t totalBytes() const { return mTotalBytes; }

    void addResident(int64_t bytes) { mTotalBytes += bytes; }
    void subResident(int64_t bytes) {
        if (mTotalBytes >= bytes) mTotalBytes -= bytes;
        else mTotalBytes = 0;
    }

    // Returns true iff the budget is unlimited or `total + additional`
    // would still fit. Use to gate a load before starting it; on false,
    // caller runs eviction and re-checks.
    bool fitsAfter(int64_t additionalBytes) const {
        if (mBudgetMb <= 0) return true;
        return mTotalBytes + additionalBytes <= budgetBytes();
    }

    int32_t evictionCount() const { return mEvictionCount; }
    void recordEviction() { ++mEvictionCount; }

private:
    int32_t mBudgetMb = 0;          // 0 = unlimited (default for safety)
    int64_t mTotalBytes = 0;
    int32_t mEvictionCount = 0;
};

} // namespace oird
