// v0.4 H4-B: image decode helpers isolated from oird.cpp so libjpeg/libpng
// macros do not pollute the main TU.
//
// v0.7 hardening:
// - JPEG: custom jpeg_error_mgr + setjmp/longjmp. A malformed JPEG
//   returns false instead of letting libjpeg call exit() and tear
//   the daemon down.
// - JPEG + PNG: dimension cap (caller-supplied via maxPixels) with
//   overflow-safe size_t multiply. Untrusted client cannot cause oird
//   to attempt an unbounded resize().
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <csetjmp>
#include "image_decode.h"

extern "C" {
#include <jpeglib.h>
}

#include <png.h>

namespace {

// Returns true if (w * h * 3) fits in size_t without overflow AND is
// within maxPixels (0 = no cap, only overflow check). bytes_out is set
// only on success.
bool checkDimensions(int w, int h, size_t maxPixels, size_t& bytes_out) {
    if (w <= 0 || h <= 0) return false;
    const size_t uw = static_cast<size_t>(w);
    const size_t uh = static_cast<size_t>(h);
    if (uh > 0 && uw > std::numeric_limits<size_t>::max() / uh) return false;  // pixels overflow
    const size_t pixels = uw * uh;
    if (maxPixels != 0 && pixels > maxPixels) return false;
    if (pixels > std::numeric_limits<size_t>::max() / 3) return false;          // bytes overflow
    bytes_out = pixels * 3;
    return true;
}

// Custom JPEG error manager. Replaces libjpeg default error_exit (which
// calls exit() and kills the process) with a longjmp back to the caller
// so we can clean up and return false.
struct JpegErrorMgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

void jpegErrorExit(j_common_ptr cinfo) {
    auto* err = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
    longjmp(err->setjmp_buffer, 1);
}

}  // namespace

bool decodeJpeg(const std::string& path, RgbImage& out, size_t maxPixels) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    struct jpeg_decompress_struct cinfo;
    JpegErrorMgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpegErrorExit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(f);
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        fclose(f);
        return false;
    }
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    out.w = cinfo.output_width;
    out.h = cinfo.output_height;
    size_t buf_bytes = 0;
    if (!checkDimensions(out.w, out.h, maxPixels, buf_bytes)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(f);
        return false;
    }
    out.px.resize(buf_bytes);
    int row_stride = out.w * 3;
    while (cinfo.output_scanline < (JDIMENSION)out.h) {
        JSAMPROW row = out.px.data() + (size_t)cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return true;
}

bool decodePng(const std::string& path, RgbImage& out, size_t maxPixels) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    png_byte hdr[8];
    if (fread(hdr, 1, 8, f) != 8 || png_sig_cmp(hdr, 0, 8)) { fclose(f); return false; }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(f); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); fclose(f); return false; }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(f);
        return false;
    }
    png_init_io(png, f);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);
    out.w = png_get_image_width(png, info);
    out.h = png_get_image_height(png, info);
    png_byte ct = png_get_color_type(png, info);
    png_byte bd = png_get_bit_depth(png, info);
    if (bd == 16) png_set_strip_16(png);
    if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (ct == PNG_COLOR_TYPE_GRAY && bd < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if (ct == PNG_COLOR_TYPE_RGBA || ct == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_strip_alpha(png);
    png_read_update_info(png, info);
    size_t buf_bytes = 0;
    if (!checkDimensions(out.w, out.h, maxPixels, buf_bytes)) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(f);
        return false;
    }
    out.px.resize(buf_bytes);
    std::vector<png_bytep> rows(out.h);
    for (int y = 0; y < out.h; ++y) rows[y] = out.px.data() + (size_t)y * out.w * 3;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(f);
    return true;
}
