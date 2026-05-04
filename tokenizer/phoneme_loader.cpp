// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// tokenizer/phoneme_loader.cpp — definitions for PhonemeMap.

#include "tokenizer/phoneme_loader.h"

#include <cctype>

#include <android-base/logging.h>

#include "common/json_util.h"

namespace oird {

bool loadPhonemeSidecar(const std::string& path, PhonemeMap& out) {
    std::string content;
    if (!readFileToString(path, content)) return false;
    const size_t end = content.size();

    const size_t gmapKey = content.find("\"grapheme_map\"");
    if (gmapKey == std::string::npos) {
        LOG(WARNING) << "oird: phoneme sidecar " << path << " missing \"grapheme_map\"";
        return false;
    }
    size_t p = content.find('{', gmapKey);
    if (p == std::string::npos) return false;
    ++p;
    while (true) {
        p = skipJsonWs(content, p, end);
        if (p >= end) return false;
        if (content[p] == '}') { ++p; break; }
        std::string key;
        size_t nextP = parseJsonString(content, p, end, key);
        if (nextP == std::string::npos) return false;
        p = skipJsonWs(content, nextP, end);
        if (p >= end || content[p] != ':') return false;
        ++p;
        std::vector<int64_t> ids;
        if (!parseIdArray(content, p, end, ids)) return false;
        // Case-fold ASCII for stable lookup.
        for (auto& c : key) c = (char)tolower((unsigned char)c);
        if (key == "<unk>") out.unkIds = std::move(ids);
        else out.graphemeToIds[std::move(key)] = std::move(ids);
    }
    LOG(INFO) << "oird: loaded phoneme map from " << path
              << " entries=" << out.graphemeToIds.size()
              << " has_unk=" << (!out.unkIds.empty() ? "yes" : "no");
    return !out.empty();
}

std::vector<int64_t> graphemesToPhonemeIds(const std::string& text,
                                            const PhonemeMap& pm) {
    std::vector<int64_t> out;
    std::string tok;
    auto flush = [&]() {
        if (tok.empty()) return;
        auto it = pm.graphemeToIds.find(tok);
        if (it != pm.graphemeToIds.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
        } else {
            // Per-character fallback.
            bool anyHit = false;
            for (char c : tok) {
                std::string one(1, c);
                auto cit = pm.graphemeToIds.find(one);
                if (cit != pm.graphemeToIds.end()) {
                    anyHit = true;
                    out.insert(out.end(), cit->second.begin(), cit->second.end());
                }
            }
            if (!anyHit && !pm.unkIds.empty()) {
                out.insert(out.end(), pm.unkIds.begin(), pm.unkIds.end());
            }
        }
        tok.clear();
    };
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { flush(); continue; }
        tok.push_back((char)tolower((unsigned char)c));
    }
    flush();
    return out;
}

} // namespace oird
