// Copyright (C) 2026 The OpenIntelligenceRuntime Project
// Licensed under the Apache License, Version 2.0
//
// common/json_util.cpp — handwritten JSON helpers.

#include "common/json_util.h"

#include <fstream>
#include <iterator>
#include <sys/stat.h>

namespace oird {

bool fileExists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool readFileToString(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    out.assign((std::istreambuf_iterator<char>(f)),
               std::istreambuf_iterator<char>());
    return true;
}

size_t skipJsonWs(const std::string& s, size_t p, size_t end) {
    while (p < end && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n'
                       || s[p] == '\r' || s[p] == ',')) ++p;
    return p;
}

size_t parseJsonString(const std::string& s, size_t p, size_t end, std::string& out) {
    if (p >= end || s[p] != '"') return std::string::npos;
    ++p;
    out.clear();
    while (p < end && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < end) {
            const char n = s[p + 1];
            if (n == '"')  { out.push_back('"');  p += 2; }
            else if (n == '\\') { out.push_back('\\'); p += 2; }
            else if (n == '/')  { out.push_back('/');  p += 2; }
            else if (n == 'n')  { out.push_back('\n'); p += 2; }
            else if (n == 't')  { out.push_back('\t'); p += 2; }
            else if (n == 'r')  { out.push_back('\r'); p += 2; }
            else if (n == 'b')  { out.push_back('\b'); p += 2; }
            else if (n == 'f')  { out.push_back('\f'); p += 2; }
            else if (n == 'u' && p + 5 < end) {
                unsigned cp = 0;
                for (int i = 0; i < 4; ++i) {
                    char c = s[p + 2 + i];
                    cp <<= 4;
                    if (c >= '0' && c <= '9')       cp |= (c - '0');
                    else if (c >= 'a' && c <= 'f')  cp |= (10 + c - 'a');
                    else if (c >= 'A' && c <= 'F')  cp |= (10 + c - 'A');
                    else return std::string::npos;
                }
                // UTF-8 encode (BMP only; surrogate pairs not supported).
                if (cp < 0x80) {
                    out.push_back((char)cp);
                } else if (cp < 0x800) {
                    out.push_back((char)(0xC0 | (cp >> 6)));
                    out.push_back((char)(0x80 | (cp & 0x3F)));
                } else {
                    out.push_back((char)(0xE0 | (cp >> 12)));
                    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back((char)(0x80 | (cp & 0x3F)));
                }
                p += 6;
            } else {
                out.push_back(n); p += 2;
            }
        } else {
            out.push_back(s[p++]);
        }
    }
    if (p >= end) return std::string::npos;
    return p + 1;  // past closing quote
}

bool parseIdArray(const std::string& s, size_t& p, size_t end,
                  std::vector<int64_t>& out) {
    p = skipJsonWs(s, p, end);
    if (p >= end || s[p] != '[') return false;
    ++p;
    while (true) {
        p = skipJsonWs(s, p, end);
        if (p >= end) return false;
        if (s[p] == ']') { ++p; return true; }
        const size_t numStart = p;
        if (s[p] == '-' || s[p] == '+') ++p;
        while (p < end && s[p] >= '0' && s[p] <= '9') ++p;
        if (numStart == p) return false;
        try {
            out.push_back((int64_t)std::stoll(s.substr(numStart, p - numStart)));
        } catch (...) { return false; }
    }
}

} // namespace oird
