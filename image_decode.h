// v0.4 H4-B: shared interface for image decode helpers.
//
// v0.7 hardening: caller passes maxPixels (loaded from oir_config.xml
// image.max_pixels knob; 0 = no cap). Decoders return false cleanly
// on dimension overflow OR cap violation OR malformed input — never
// abort the daemon.
#ifndef OIRD_IMAGE_DECODE_H
#define OIRD_IMAGE_DECODE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Default cap: 4096x4096 = 16M pixels (~48 MB RGB). Above any realistic
// VLM/detector input; comfortably below "OOM the daemon" territory.
// OEMs override via /vendor/etc/oir/oir_config.xml image.max_pixels.
constexpr size_t kDefaultMaxImagePixels = 4096u * 4096u;

struct RgbImage {
    std::vector<uint8_t> px;  // HWC, 8-bit per channel, RGB
    int w = 0;
    int h = 0;
};

bool decodeJpeg(const std::string& path, RgbImage& out, size_t maxPixels = kDefaultMaxImagePixels);
bool decodePng(const std::string& path, RgbImage& out, size_t maxPixels = kDefaultMaxImagePixels);

#endif
