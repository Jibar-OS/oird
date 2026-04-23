// v0.4 H4-B: image decode helpers isolated from oird.cpp so libjpeg/libpng
// macros do not pollute the main TU.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <csetjmp>
#include "image_decode.h"

extern "C" {
#include <jpeglib.h>
}

#include <png.h>


bool decodeJpeg(const std::string& path, RgbImage& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
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
    out.px.resize((size_t)out.w * out.h * 3);
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

bool decodePng(const std::string& path, RgbImage& out) {
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
    out.px.resize((size_t)out.w * out.h * 3);
    std::vector<png_bytep> rows(out.h);
    for (int y = 0; y < out.h; ++y) rows[y] = out.px.data() + (size_t)y * out.w * 3;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(f);
    return true;
}
