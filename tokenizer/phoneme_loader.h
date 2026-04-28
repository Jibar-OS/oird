// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// tokenizer/phoneme_loader.h — phoneme map for audio.synthesize.
//
// Sidecar schema: `<piper-model-path>.phonemes.json`
//   { "version": 1,
//     "phoneme_ids": { "<ph>": <int>, ... },      // not used by the simple
//                                                 //   word-level path below
//     "grapheme_map": { "<word-or-grapheme>": [<id>, <id>, ...], ... } }
// Lookup is word-level (case-insensitive, whitespace-tokenized). Unknown words
// fall back to per-character lookup using single-char keys if present in the
// map; otherwise emit the "<unk>" entry if one is declared, else drop the
// word silently. This is intentionally simple — OEM-supplied maps for real
// languages can be produced from espeak-ng dumps.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace oird {

struct PhonemeMap {
    std::unordered_map<std::string, std::vector<int64_t>> graphemeToIds;
    std::vector<int64_t> unkIds;
    bool empty() const { return graphemeToIds.empty() && unkIds.empty(); }
};

bool loadPhonemeSidecar(const std::string& path, PhonemeMap& out);

// Split `text` into lowercase whitespace-delimited tokens and look up each
// in the phoneme map. On miss, try per-character; on total miss, append
// <unk> if defined. Concatenates the ID sequences.
std::vector<int64_t> graphemesToPhonemeIds(const std::string& text,
                                            const PhonemeMap& pm);

} // namespace oird
