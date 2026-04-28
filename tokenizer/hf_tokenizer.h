// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// tokenizer/hf_tokenizer.h — minimal HuggingFace-style tokenizer used by
// text.classify and text.rerank.
//
// Sidecar schema: `<model-path>.tokenizer.json` — the standard HF tokenizer
// export, of which we consume a narrow subset:
//   { "model": { "vocab": { "<token>": <id>, ... } },
//     "added_tokens": [{ "id": N, "content": "[CLS]" | "[SEP]" | "[UNK]" | "[PAD]" }, ...] }
// Runtime does whitespace pre-tokenization + longest-prefix match against the
// vocab. This is NOT a full WordPiece/BPE implementation — it works cleanly
// for single-word lookups and falls back to <UNK> for subword splits that
// need real WordPiece. OEMs shipping a classifier that requires full
// WordPiece must either bake a pre-tokenized model or link a fuller
// tokenizer in v0.6.1.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace oird {

class HfTokenizer {
public:
    bool loadFromSidecar(const std::string& path);
    std::vector<int64_t> encode(const std::string& text) const;
    std::vector<int64_t> encodePair(const std::string& a, const std::string& b) const;
    std::vector<int64_t> typeIdsForPair(const std::string& a, const std::string& b) const;
    bool valid() const { return !mVocab.empty(); }

private:
    std::unordered_map<std::string, int64_t> mVocab;
    int64_t mCls  = -1;
    int64_t mSep  = -1;
    int64_t mUnk  = -1;
    int64_t mPad  = -1;

    std::vector<int64_t> tokenize(const std::string& text) const;
};

bool loadHfTokenizerSidecar(const std::string& path, HfTokenizer& out);

} // namespace oird
