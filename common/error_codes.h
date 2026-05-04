// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// common/error_codes.h — service-specific error code constants returned to
// callers via ScopedAStatus::fromServiceSpecificErrorWithMessage(). These
// are the wire-level codes the SDK sees; do not renumber once shipped.
#pragma once

#include <cstdint>

namespace oird {

constexpr int32_t W_MODEL_ERROR         = 1;
constexpr int32_t W_CANCELLED           = 2;
constexpr int32_t W_INVALID_INPUT       = 3;
constexpr int32_t W_INSUFFICIENT_MEMORY = 4;
constexpr int32_t W_TIMEOUT             = 5;
// v0.6 Phase A: OEM model shape mismatch (load-time). Distinct from
// W_MODEL_ERROR so SDK can map to typed OirModelIncompatibleException
// and apps can fail gracefully + fall back to OEM-bake guidance.
constexpr int32_t W_MODEL_INCOMPATIBLE  = 11;
// v0.6 Phase A: a capability is declared in capabilities.xml but the
// backend doesn't support its shape on this device. Distinct from
// W_MODEL_INCOMPATIBLE (model shape) and W_CAPABILITY_UNAVAILABLE_NO_MODEL
// (no model file).
constexpr int32_t W_CAPABILITY_UNSUPPORTED = 12;
// v0.5 V8: capability declared but no default model on device — OEM
// must bake one. Maps to OirCapabilityUnavailableException with
// reason=NO_MODEL.
constexpr int32_t W_CAPABILITY_UNAVAILABLE_NO_MODEL = 9;

} // namespace oird
