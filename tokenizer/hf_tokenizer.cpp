// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// tokenizer/hf_tokenizer.cpp — definitions for HfTokenizer.

#include "tokenizer/hf_tokenizer.h"

#include <cctype>

#include <android-base/logging.h>

#include "common/json_util.h"

namespace oird {

bool HfTokenizer::loadFromSidecar(const std::string& path) {
    std::string content;
    if (!readFileToString(path, content)) return false;
    const size_t end = content.size();

    // Find "vocab" key (accept nested under "model.vocab" or top-level).
    size_t vkey = content.find("\"vocab\"");
    if (vkey == std::string::npos) return false;
    size_t p = content.find('{', vkey);
    if (p == std::string::npos) return false;
    ++p;
    while (true) {
        p = skipJsonWs(content, p, end);
        if (p >= end) return false;
        if (content[p] == '}') { ++p; break; }
        std::string token;
        size_t nextP = parseJsonString(content, p, end, token);
        if (nextP == std::string::npos) return false;
        p = skipJsonWs(content, nextP, end);
        if (p >= end || content[p] != ':') return false;
        ++p;
        p = skipJsonWs(content, p, end);
        const size_t numStart = p;
        if (p < end && (content[p] == '-' || content[p] == '+')) ++p;
        while (p < end && content[p] >= '0' && content[p] <= '9') ++p;
        if (numStart == p) return false;
        try {
            mVocab[std::move(token)] = std::stoll(content.substr(numStart, p - numStart));
        } catch (...) { return false; }
    }

    // Wire special tokens if present.
    auto look = [&](const char* s) -> int64_t {
        auto it = mVocab.find(s);
        return it == mVocab.end() ? -1 : it->second;
    };
    mCls = look("[CLS]");   if (mCls < 0) mCls = look("<s>");
    mSep = look("[SEP]");   if (mSep < 0) mSep = look("</s>");
    mUnk = look("[UNK]");   if (mUnk < 0) mUnk = look("<unk>");
    mPad = look("[PAD]");   if (mPad < 0) mPad = look("<pad>");

    LOG(INFO) << "oird: loaded tokenizer from " << path
              << " vocab=" << mVocab.size()
              << " cls=" << mCls << " sep=" << mSep
              << " unk=" << mUnk << " pad=" << mPad;
    return !mVocab.empty();
}

// Whitespace + longest-prefix match. Case-folds ASCII lowercase before
// lookup — enough for uncased BERT-family tokenizers; OEM tokenizers that
// are case-sensitive or require proper WordPiece need v0.6.1 or a pre-
// tokenized input flow.
std::vector<int64_t> HfTokenizer::tokenize(const std::string& text) const {
    std::vector<int64_t> out;
    std::string word;
    auto flushWord = [&]() {
        if (word.empty()) return;
        // Try whole-word first.
        auto it = mVocab.find(word);
        if (it != mVocab.end()) { out.push_back(it->second); word.clear(); return; }
        // Longest-prefix match walk — approximates WordPiece for single-piece vocab.
        size_t pos = 0;
        bool first = true;
        while (pos < word.size()) {
            size_t bestLen = 0;
            int64_t bestId = -1;
            for (size_t len = word.size() - pos; len > 0; --len) {
                std::string piece = (first ? "" : "##") + word.substr(pos, len);
                auto pit = mVocab.find(piece);
                if (pit != mVocab.end()) { bestLen = len; bestId = pit->second; break; }
            }
            if (bestId < 0) { out.push_back(mUnk >= 0 ? mUnk : 0); word.clear(); return; }
            out.push_back(bestId);
            pos += bestLen;
            first = false;
        }
        word.clear();
    };
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r'
                || c == '.' || c == ',' || c == '!' || c == '?' || c == ';') {
            flushWord();
        } else {
            word.push_back((char)tolower((unsigned char)c));
        }
    }
    flushWord();
    return out;
}

std::vector<int64_t> HfTokenizer::encode(const std::string& text) const {
    std::vector<int64_t> ids;
    if (mCls >= 0) ids.push_back(mCls);
    auto mid = tokenize(text);
    ids.insert(ids.end(), mid.begin(), mid.end());
    if (mSep >= 0) ids.push_back(mSep);
    return ids;
}

std::vector<int64_t> HfTokenizer::encodePair(const std::string& a,
                                              const std::string& b) const {
    std::vector<int64_t> ids;
    if (mCls >= 0) ids.push_back(mCls);
    auto aIds = tokenize(a);
    ids.insert(ids.end(), aIds.begin(), aIds.end());
    if (mSep >= 0) ids.push_back(mSep);
    auto bIds = tokenize(b);
    ids.insert(ids.end(), bIds.begin(), bIds.end());
    if (mSep >= 0) ids.push_back(mSep);
    return ids;
}

std::vector<int64_t> HfTokenizer::typeIdsForPair(const std::string& a,
                                                  const std::string& b) const {
    // Mirror the layout encodePair produces: [CLS] a [SEP] b [SEP]
    std::vector<int64_t> types;
    if (mCls >= 0) types.push_back(0);
    auto aIds = tokenize(a);
    for (size_t i = 0; i < aIds.size(); ++i) types.push_back(0);
    if (mSep >= 0) types.push_back(0);
    auto bIds = tokenize(b);
    for (size_t i = 0; i < bIds.size(); ++i) types.push_back(1);
    if (mSep >= 0) types.push_back(1);
    return types;
}

bool loadHfTokenizerSidecar(const std::string& path, HfTokenizer& out) {
    return out.loadFromSidecar(path);
}

} // namespace oird
