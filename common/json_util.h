// Copyright (C) 2026 The Open Intelligence Runtime Project, a JibarOS project
// Licensed under the Apache License, Version 2.0
//
// common/json_util.h — minimal handwritten JSON parsing primitives shared
// by sidecar loaders (tokenizer, phoneme map, vision.detect class labels).
// Hand-rolled to avoid pulling in a third-party JSON dep into the daemon.
//
// Not a full parser — only the subset oird's sidecars use:
//   - whitespace/comma skipping
//   - string parsing with \", \\, \n, \t, \/, \uXXXX (BMP only)
//   - integer-array parsing
// Plus tiny file utilities (readFileToString, fileExists).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace oird {

bool fileExists(const std::string& path);

// Read a whole file into a string; returns false if unreadable.
bool readFileToString(const std::string& path, std::string& out);

// Skip JSON whitespace + commas. Returns new position.
size_t skipJsonWs(const std::string& s, size_t p, size_t end);

// Parse a JSON string token starting at s[p] (which must be '"'). Returns the
// position after the closing quote and writes the decoded string to `out`.
// Supports \", \\, \n, \t, \/, \uXXXX (BMP only). Returns std::string::npos
// on malformed input.
size_t parseJsonString(const std::string& s, size_t p, size_t end, std::string& out);

// Parse a JSON integer array starting at s[p]. Advances p past the closing
// `]`. Returns false on malformed input.
bool parseIdArray(const std::string& s, size_t& p, size_t end,
                  std::vector<int64_t>& out);

} // namespace oird
