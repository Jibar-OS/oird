// v0.4 H4-B: shared interface for image decode helpers.
#ifndef OIRD_IMAGE_DECODE_H
#define OIRD_IMAGE_DECODE_H

#include <cstdint>
#include <string>
#include <vector>

struct RgbImage {
    std::vector<uint8_t> px;  // HWC, 8-bit per channel, RGB
    int w = 0;
    int h = 0;
};

bool decodeJpeg(const std::string& path, RgbImage& out);
bool decodePng(const std::string& path, RgbImage& out);

#endif
