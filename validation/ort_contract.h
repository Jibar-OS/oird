// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// validation/ort_contract.h — ORT load-time shape validation.
//
// v0.6 Phase A: ORT load-time shape validation.
//
// v0.5 and earlier assumed OEM-supplied ONNX models matched the hardcoded
// shape contract for each capability. Mismatched models SIGSEGV'd at
// inference time or silently produced garbage. v0.6 validates at load
// time and returns MODEL_INCOMPATIBLE with a diagnostic message, so OEMs
// see "this model won't work" before they ship a device.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace oird {

// Validates input count + optional per-input shape. -1 in expected shape
// = wildcard (batch / dynamic spatial dims). Returns empty string on
// success; diagnostic on mismatch.
std::string validateOrtContract(Ort::Session* s,
                                size_t expectedInputs,
                                const std::vector<std::vector<int64_t>>& inputShapes,
                                const char* capability);

} // namespace oird
