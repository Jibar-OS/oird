// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// validation/ort_contract.cpp — definition of validateOrtContract.

#include "validation/ort_contract.h"

namespace oird {

std::string validateOrtContract(Ort::Session* s,
                                size_t expectedInputs,
                                const std::vector<std::vector<int64_t>>& inputShapes,
                                const char* capability) {
    if (!s) return std::string(capability) + ": null session";
    auto toStr = [](const std::vector<int64_t>& v) {
        std::string r = "[";
        for (size_t i = 0; i < v.size(); ++i) { if (i) r += ","; r += std::to_string(v[i]); }
        return r + "]";
    };
    size_t nIn = s->GetInputCount();
    if (nIn != expectedInputs) {
        return std::string(capability) + ": expected "
               + std::to_string(expectedInputs) + " inputs, got "
               + std::to_string(nIn);
    }
    for (size_t i = 0; i < inputShapes.size() && i < nIn; ++i) {
        const auto& expect = inputShapes[i];
        if (expect.empty()) continue;  // skip check for this input
        auto actual = s->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        if (expect.size() != actual.size()) {
            return std::string(capability) + ": input[" + std::to_string(i)
                   + "] rank mismatch: expected " + toStr(expect)
                   + " got " + toStr(actual);
        }
        for (size_t d = 0; d < expect.size(); ++d) {
            if (expect[d] == -1) continue;  // wildcard on expected side
            // v0.6.9: treat dynamic actual dim (-1) as compatible with any
            // concrete expected dim. ONNX models with dynamic H/W (e.g. RT-DETR
            // `pixel_values [batch,3,height,width]` where height/width are
            // ONNX symbolic names) accept the concrete size at Run() time —
            // they're not incompatible with our preprocess pipeline that
            // resizes to kIn×kIn, they just don't pre-fix the spatial dims
            // in the graph.
            if (actual[d] == -1) continue;
            if (expect[d] != actual[d]) {
                return std::string(capability) + ": input[" + std::to_string(i)
                       + "] shape mismatch: expected " + toStr(expect)
                       + " got " + toStr(actual);
            }
        }
    }
    return "";
}

} // namespace oird
