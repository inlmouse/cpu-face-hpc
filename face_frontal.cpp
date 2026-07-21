/*
The MIT License (MIT)

Copyright (c) 2026 Patrick J. Hu, Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

// SIMD 
#include <emmintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>


constexpr size_t LANDMARK_POINTS = 68;
constexpr size_t FERN_STAGE_COUNT = 5;
constexpr size_t FERN_FEATURES_PER_STAGE = 340;
constexpr size_t FERN_NODE_SIZE = 868;
constexpr size_t WEIGHT_ROW_SIZE = 21760;

constexpr size_t MAX_FEATURE_CHANNELS = 11;
constexpr size_t ALIGN_BYTES = 16;

alignas(16) const int32_t K_CONST_512[4] = { 512, 512, 512, 512 };

struct DetectionRect {
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    int32_t confidence;
    int32_t reserved[64];
};

struct FaceResult {
    DetectionRect bbox;
    std::vector<std::pair<float, float>> landmarks;
};

struct DetectOptions {
    int min_size = 48;
    int max_size = 0;                 // 0 = no limit
    float scale_step_ratio = 1.2f;
    int min_neighbors = 2;
    bool with_landmarks = true;
};

struct FaceDetection {
    int32_t x = 0;
    int32_t y = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t score = 0;                // GroupRectangles cluster average margin
    int32_t neighbors = 0;            // GroupRectangles hits number
    bool has_landmarks = false;
    std::array<std::pair<float, float>, LANDMARK_POINTS> landmarks{};
};

// -----------------------------------------------------------------
// Cold Data Zone: (x1,y1,x2,y2,ch, thr, l, r, sl, sr)
// Two sampled pixels share the same channel ch
// -----------------------------------------------------------------
struct ModelNode {
    int32_t x1, y1, x2, y2, ch;
    int32_t threshold;
    int32_t left_idx;    // Root node: Child node index (1/2)
    int32_t right_idx;
    int32_t left_score;  // Leaf node: Output score (int16 scale)
    int32_t right_score;
};

struct RuntimeNode {
    const uint8_t* ptrA;
    const uint8_t* ptrB;
    int32_t threshold;
    int32_t left_idx;
    int32_t right_idx;
    int32_t left_score;
    int32_t right_score;
};

// Each depth-2 tree = 3 nodes + 1 early exit threshold.
struct RuntimeTree {
    RuntimeNode nodes[3];
    int32_t reject_threshold; // Tree tail block value (not used during inference)
};

struct CascadeStage {
    int32_t tree_count = 0;
    int32_t threshold = 0;                  // [stage threshold = 16B; first int32 of the stage's tail block
    std::vector<ModelNode>   model_nodes;   // Cold data: 3N items; read-only after loading.
    std::vector<int32_t>     tree_reject;   // Tail block value of each tree (not used during inference; retained only)
};

// A private "hot data" view for each call/scale is used, eliminating the need to write back to the shared `stages_`.
struct PatchedStage {
    int32_t threshold = 0;
    std::vector<RuntimeTree> trees;
};

class FaceDetector {
private:
    std::vector<CascadeStage> stages_;      // [FIX-1] 14 个 stage
    int32_t win_w_ = 24, win_h_ = 24;
    std::vector<float> mean_shape_;
    std::vector<uint8_t> ferns_data_;
    std::vector<uint8_t> weights_data_;

    bool is_loaded_ = false;

    static void read_file(const std::string& path, std::vector<uint8_t>& out) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) throw std::runtime_error("Unable to open the file: " + path);
        fin.seekg(0, std::ios::end);
        size_t sz = static_cast<size_t>(fin.tellg());
        fin.seekg(0, std::ios::beg);
        out.resize(sz);
        if (sz > 0) fin.read(reinterpret_cast<char*>(out.data()), sz);
    }

    static void ParallelBilinearResample(
        const uint8_t* img_base, int img_stride, int img_h,
        uint8_t* out_features, int target_stride, int target_w, int target_h,
        const std::vector<int>& y_map, const std::vector<int>& x_map)
    {
        int prev_idx_y = -1, prev_next_y = -1;
        std::vector<int32_t> buf_a(target_w, 0);
        std::vector<int32_t> buf_b(target_w, 0);
        int32_t* line_curr = buf_a.data();
        int32_t* line_next = buf_b.data();

        uint8_t* curr_out_row = out_features;
        const int* y_map_ptr = y_map.data();
        const int* x_map_ptr = x_map.data();

        for (int y = 0; y < target_h; ++y) {
            int idx_y = y_map_ptr[2 * y];
            int weight_y = y_map_ptr[2 * y + 1];
            int target_next_y = (weight_y > 0 && idx_y < img_h - 1) ? (idx_y + 1) : idx_y;

            if (idx_y != prev_idx_y || target_next_y != prev_next_y) {
                int load_mode = 0;
                if (idx_y == prev_next_y) {
                    std::swap(line_curr, line_next);
                    load_mode = 1;
                }
                while (load_mode < 2) {
                    int32_t* active_buf = (load_mode == 0) ? line_curr : line_next;
                    int active_y = (load_mode == 0) ? idx_y : target_next_y;
                    if (load_mode == 1 && target_next_y == idx_y) {
                        std::memcpy(line_next, line_curr, sizeof(int32_t) * target_w);
                    } else {
                        const uint8_t* img_row = img_base + static_cast<ptrdiff_t>(img_stride) * active_y;
                        for (int x = 0; x < target_w; ++x) {
                            int src_x = x_map_ptr[2 * x];
                            int weight_x = x_map_ptr[2 * x + 1];
                            uint8_t p1 = img_row[src_x];
                            uint8_t p2 = img_row[src_x + 1];
                            active_buf[x] = (static_cast<int>(p1) << 10) + weight_x * (static_cast<int>(p2) - static_cast<int>(p1));
                        }
                    }
                    ++load_mode;
                }
                prev_idx_y = idx_y;
                prev_next_y = target_next_y;
            }

            int x_simd = 0;
            if (idx_y == target_next_y) {
                if (target_w >= 16) {
                    __m128i k_512 = _mm_load_si128(reinterpret_cast<const __m128i*>(K_CONST_512));
                    for (; x_simd <= target_w - 16; x_simd += 16) {
                        __m128i src0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd]));
                        __m128i src1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd + 4]));
                        __m128i src2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd + 8]));
                        __m128i src3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd + 12]));
                        src0 = _mm_srai_epi32(_mm_slli_epi32(_mm_add_epi32(src0, k_512), 10), 20);
                        src1 = _mm_srai_epi32(_mm_slli_epi32(_mm_add_epi32(src1, k_512), 10), 20);
                        src2 = _mm_srai_epi32(_mm_slli_epi32(_mm_add_epi32(src2, k_512), 10), 20);
                        src3 = _mm_srai_epi32(_mm_slli_epi32(_mm_add_epi32(src3, k_512), 10), 20);
                        __m128i pack01 = _mm_packus_epi16(_mm_packs_epi32(src0, src1), _mm_packs_epi32(src2, src3));
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(&curr_out_row[x_simd]), pack01);
                    }
                }
                for (; x_simd < target_w; ++x_simd) {
                    curr_out_row[x_simd] = static_cast<uint8_t>(((line_curr[x_simd] + 512) << 10) >> 20);
                }
            } else {
                if (target_w >= 8) {
                    __m128i k_512 = _mm_load_si128(reinterpret_cast<const __m128i*>(K_CONST_512));
                    __m128i v_weight_y = _mm_shuffle_epi32(_mm_cvtsi32_si128(weight_y), 0);
                    for (; x_simd <= target_w - 8; x_simd += 8) {
                        __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd]));
                        __m128i n0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_next[x_simd]));
                        __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd + 4]));
                        __m128i n1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_next[x_simd + 4]));
                        __m128i mix0 = _mm_add_epi32(_mm_slli_epi32(_mm_add_epi32(c0, k_512), 10), _mm_mullo_epi32(_mm_sub_epi32(n0, c0), v_weight_y));
                        __m128i mix1 = _mm_add_epi32(_mm_slli_epi32(_mm_add_epi32(c1, k_512), 10), _mm_mullo_epi32(_mm_sub_epi32(n1, c1), v_weight_y));
                        mix0 = _mm_srai_epi32(mix0, 20);
                        mix1 = _mm_srai_epi32(mix1, 20);
                        __m128i packed = _mm_packus_epi16(_mm_packs_epi32(mix0, mix1), _mm_packs_epi32(mix0, mix1));
                        std::memcpy(&curr_out_row[x_simd], &packed, 8);
                    }
                }
                for (; x_simd < target_w; ++x_simd) {
                    curr_out_row[x_simd] = static_cast<uint8_t>((((line_curr[x_simd] + 512) << 10) + weight_y * (line_next[x_simd] - line_curr[x_simd])) >> 20);
                }
            }
            curr_out_row += target_stride;
        }
    }

    static void BuildPyramidScale(
        const uint8_t* img_base, int img_w, int img_h, int img_stride,
        uint8_t* out_features, int target_w, int target_h, int target_stride)
    {
        float scale_x = static_cast<float>(img_w) / static_cast<float>(target_w);
        float scale_y = static_cast<float>(img_h) / static_cast<float>(target_h);

        std::vector<int> x_map(target_w * 2);
        std::vector<int> y_map(target_h * 2);

        int last_pixel_x = img_w - 1;
        for (int x = 0; x < target_w; ++x) {
            float src_x = ((static_cast<float>(x) + 0.5f) * scale_x) - 0.5f;
            int idx_x = static_cast<int>(std::floor(src_x));
            float weight_x = src_x - static_cast<float>(idx_x);
            if (idx_x < 0) { weight_x = 0.0f; idx_x = 0; }
            if (idx_x >= last_pixel_x) { weight_x = 0.0f; idx_x = last_pixel_x; }
            x_map[2 * x] = idx_x;
            x_map[2 * x + 1] = static_cast<int>(weight_x * 1024.0f + 0.5f);
        }
        for (int y = 0; y < target_h; ++y) {
            float src_y = ((static_cast<float>(y) + 0.5f) * scale_y) - 0.5f;
            int idx_y = static_cast<int>(std::floor(src_y));
            float weight_y = src_y - static_cast<float>(idx_y);
            if (idx_y < 0) { weight_y = 0.0f; idx_y = 0; }
            y_map[2 * y] = idx_y;
            y_map[2 * y + 1] = static_cast<int>(weight_y * 1024.0f + 0.5f);
        }

        ParallelBilinearResample(img_base, img_stride, img_h, out_features, target_stride, target_w, target_h, y_map, x_map);
    }

    static void Extract11ChannelFeatures(const uint8_t* src_buf, int width, int height, size_t element_w, uint8_t* out_multi_channel)
    {
        size_t row_stride = 11 * element_w;

        // Channel 0
        if (height > 0) {
            const uint8_t* src_ptr = src_buf;
            uint8_t* dst_ptr = out_multi_channel;
            for (int y = 0; y < height; ++y) {
                std::memcpy(dst_ptr, src_ptr, element_w);
                src_ptr += element_w;
                dst_ptr += row_stride;
            }
        }

        // Channel 2x2  pooling
        if (height > 1) {
            int limit_h = height - 1;
            int limit_w = width - 1;
            uint8_t* row_ptr = out_multi_channel;
            for (int y = 0; y < limit_h; ++y) {
                for (int x = 0; x < limit_w; ++x) {
                    uint32_t p00 = row_ptr[x];
                    uint32_t p01 = row_ptr[x + 1];
                    uint32_t p10 = row_ptr[row_stride + x];
                    uint32_t p11 = row_ptr[row_stride + x + 1];
                    row_ptr[element_w + x] = static_cast<uint8_t>((((p00 + p10 + 1) >> 1) + ((p01 + p11 + 1) >> 1) + 1) >> 1);
                }
                row_ptr += row_stride;
            }
        }

        // Channel 2-6
        int limit_h_3 = height - 3;
        if (limit_h_3 > 0) {
            uint8_t* row_ptr = out_multi_channel + element_w;
            for (int y = 0; y < limit_h_3; ++y) {
                int limit_w_3 = width - 3;
                if (limit_w_3 > 0) {
                    uint8_t* pixel_ptr = row_ptr + 2;
                    for (int x = 0; x < limit_w_3; ++x) {
                        uint32_t p_curr = pixel_ptr[-2];
                        uint32_t p_next_stride = pixel_ptr[2 * row_stride - 2];
                        uint32_t p_offset = pixel_ptr[0];
                        pixel_ptr++;
                        uint32_t val = (((p_curr + p_next_stride) >> 1) + ((pixel_ptr[2 * row_stride - 1] + p_offset + 1) >> 1) + 1) >> 1;
                        pixel_ptr[element_w - 3] = static_cast<uint8_t>(val);
                    }
                }
                row_ptr += row_stride;
            }

            uint8_t* grad_row_ptr = out_multi_channel + element_w;
            size_t offset_c2 = 2 * element_w;
            size_t offset_c3 = 3 * element_w;

            for (int y = 0; y < limit_h_3; ++y) {
                int limit_w_3 = width - 3;
                if (limit_w_3 > 0) {
                    uint8_t* p_curr = grad_row_ptr;
                    for (int x = 0; x < limit_w_3; ++x) {
                        int32_t v0 = p_curr[0];              // ch1(y, x)
                        int32_t v2 = p_curr[2];              // ch1(y, x+2)
                        p_curr++;                            // -> ch1(y, x+1)
                        // ch3: 水平梯度 ch1(y,x+2) - ch1(y,x)
                        p_curr[offset_c2 - 1] = static_cast<uint8_t>((v2 - v0 + 255) / 2);
                        // ch4: 垂直梯度 ch1(y+2,x) - ch1(y,x)
                        p_curr[offset_c3 - 1] = static_cast<uint8_t>((p_curr[2 * row_stride - 1] - p_curr[-1] + 255) / 2);
                        // ch5: 主对角梯度 ch1(y+2,x+2) - ch1(y,x) 
                        p_curr[4 * element_w - 1] = static_cast<uint8_t>((p_curr[2 * row_stride + 1] - p_curr[-1] + 255) / 2);
                        // ch6: 副对角梯度 ch1(y+2,x) - ch1(y,x+2)
                        p_curr[5 * element_w - 1] = static_cast<uint8_t>((p_curr[2 * row_stride - 1] - p_curr[1] + 255) / 2);
                    }
                }
                grad_row_ptr += row_stride;
            }
        }

        // channel 7-10
        int limit_h_7 = height - 7;
        if (limit_h_7 > 0) {
            uint8_t* long_row_ptr = out_multi_channel + 2 * element_w;
            int limit_w_7 = width - 7;
            for (int y = 0; y < limit_h_7; ++y) {
                if (limit_w_7 > 0) {
                    uint8_t* p_curr = long_row_ptr;
                    size_t offset_c6 = 6 * element_w;
                    size_t offset_c7 = 7 * element_w;
                    size_t offset_c8 = 8 * element_w;
                    for (int x = 0; x < limit_w_7; ++x) {
                        int32_t v0 = p_curr[0];              // ch2(y, x)
                        int32_t v4 = p_curr[4];              // ch2(y, x+4)
                        p_curr++;                            // -> ch2(y, x+1)
                        // ch7: 长距水平 ch2(y,x+4) - ch2(y,x)
                        p_curr[5 * element_w - 1] = static_cast<uint8_t>((v4 - v0 + 255) / 2);
                        // ch8: 长距垂直 ch2(y+4,x) - ch2(y,x)
                        p_curr[offset_c6 - 1] = static_cast<uint8_t>((p_curr[4 * row_stride - 1] - p_curr[-1] + 255) / 2);
                        // ch9: 长距主对角 ch2(y+4,x+4) - ch2(y,x) 
                        p_curr[offset_c7 - 1] = static_cast<uint8_t>((p_curr[4 * row_stride + 3] - p_curr[-1] + 255) / 2);
                        // ch10: 长距副对角 ch2(y+4,x) - ch2(y,x+4)
                        p_curr[offset_c8 - 1] = static_cast<uint8_t>((p_curr[4 * row_stride - 1] - p_curr[3] + 255) / 2);
                    }
                }
                long_row_ptr += row_stride;
            }
        }
    }

    int EvaluateCascadeWindow(const std::vector<PatchedStage>& patched_stages,
                              int64_t woff, int32_t& out_margin) const
    {
        int stage_idx = 0;
        for (const PatchedStage& stage : patched_stages) {
            int32_t sum = 0;
            const size_t n = stage.trees.size();
            for (size_t t = 0; t < n; ++t) {
                const RuntimeTree& tree = stage.trees[t];
                const RuntimeNode& root = tree.nodes[0];
                int32_t diff = root.ptrA[woff] - root.ptrB[woff];
                const RuntimeNode& child = (diff > root.threshold)
                    ? tree.nodes[root.right_idx] : tree.nodes[root.left_idx];
                int32_t cdiff = child.ptrA[woff] - child.ptrB[woff];
                sum += (cdiff > child.threshold) ? child.right_score : child.left_score;
            }
            if (sum < stage.threshold) return (stage_idx == 0) ? 0 : -stage_idx;
            out_margin = sum - stage.threshold + 1;   // Raw Margin + 1
            ++stage_idx;
        }
        return out_margin;
    }

    static void EstimateShapeSimilarityTransform(float* out_scale, const float* current_shape, const float* mean_shape) {
        float mean_curr_x = 0.0f, mean_curr_y = 0.0f;
        float mean_base_x = 0.0f, mean_base_y = 0.0f;
        for (size_t i = 0; i < LANDMARK_POINTS; ++i) {
            mean_curr_x += current_shape[2 * i];
            mean_curr_y += current_shape[2 * i + 1];
            mean_base_x += mean_shape[2 * i];
            mean_base_y += mean_shape[2 * i + 1];
        }
        mean_curr_x /= LANDMARK_POINTS; mean_curr_y /= LANDMARK_POINTS;
        mean_base_x /= LANDMARK_POINTS; mean_base_y /= LANDMARK_POINTS;

        float sum_xx = 0.0f, sum_yy = 0.0f;
        float sum_uu = 0.0f, sum_vv = 0.0f;
        float A = 0.0f, B = 0.0f;
        for (size_t i = 0; i < LANDMARK_POINTS; ++i) {
            float x = current_shape[2 * i] - mean_curr_x;
            float y = current_shape[2 * i + 1] - mean_curr_y;
            float u = mean_shape[2 * i] - mean_base_x;
            float v = mean_shape[2 * i + 1] - mean_base_y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_uu += u * u;
            sum_vv += v * v;
            A += (u * x + v * y);
            B += (u * y - v * x);
        }
        float current_norm = std::sqrt(sum_xx + sum_yy);
        float mean_norm = std::sqrt(sum_uu + sum_vv);
        float scale = current_norm / (mean_norm + 1e-7f);
        float L = std::sqrt(A * A + B * B);
        float cos_theta = A / (L + 1e-7f);
        float sin_theta = B / (L + 1e-7f);
        out_scale[0] = scale * cos_theta;
        out_scale[1] = -scale * sin_theta;
        out_scale[2] = scale * sin_theta;
        out_scale[3] = scale * cos_theta;
    }

    static bool SimilarRectsEps01(const DetectionRect& a, const DetectionRect& b) {
        int delta_sum = std::min(a.width, b.width) + std::min(a.height, b.height);
        return 10 * std::abs(a.x - b.x) <= delta_sum &&
               10 * std::abs(a.y - b.y) <= delta_sum &&
               10 * std::abs(a.x + a.width  - b.x - b.width)  <= delta_sum &&
               10 * std::abs(a.y + a.height - b.y - b.height) <= delta_sum;
    }

    static std::vector<DetectionRect> GroupRectangles(const std::vector<DetectionRect>& rects, int group_threshold) {
        const int n = (int)rects.size();
        if (n == 0) return {};

        std::vector<int> labels(n);
        for (int i = 0; i < n; ++i) labels[i] = i;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (SimilarRectsEps01(rects[i], rects[j])) {
                    int lo = std::min(labels[i], labels[j]);
                    int hi = std::max(labels[i], labels[j]);
                    if (lo == hi) continue;
                    for (int k = 0; k < n; ++k)
                        if (labels[k] == hi) labels[k] = lo;
                }
            }
        }

        // sum_x, sum_y, sum_w, sum_h, count, sum_score
        struct Accum { int64_t sx = 0, sy = 0, sw = 0, sh = 0, sscore = 0; int32_t count = 0; };
        std::vector<Accum> acc(n);
        for (int i = 0; i < n; ++i) {
            Accum& a = acc[labels[i]];
            a.sx += rects[i].x; a.sy += rects[i].y;
            a.sw += rects[i].width; a.sh += rects[i].height;
            a.sscore += rects[i].confidence;
            a.count += 1;
        }

        // aveage
        struct Cluster { int32_t x, y, w, h, neighbors, score; };
        std::vector<Cluster> clusters;
        for (int i = 0; i < n; ++i) {
            const Accum& a = acc[i];
            if (a.count <= 0) continue;
            int64_t c = a.count;
            Cluster cl;
            cl.x = (int32_t)((c + 2 * a.sx) / (2 * c));
            cl.y = (int32_t)((c + 2 * a.sy) / (2 * c));
            cl.w = (int32_t)((c + 2 * a.sw) / (2 * c));
            cl.h = (int32_t)((c + 2 * a.sh) / (2 * c));
            cl.neighbors = a.count;
            cl.score = (int32_t)((c + 2 * a.sscore) / (2 * c));
            clusters.push_back(cl);
        }

        // groupThreshold filtering
        std::vector<DetectionRect> out;
        const int m = (int)clusters.size();
        for (int i = 0; i < m; ++i) {
            const Cluster& ci = clusters[i];
            int64_t area_i = (int64_t)ci.w * ci.h;
            bool rejected = false;
            for (int j = 0; j < m; ++j) {
                if (i == j) continue;
                const Cluster& cj = clusters[j];
                int ix1 = std::max(ci.x, cj.x), iy1 = std::max(ci.y, cj.y);
                int ix2 = std::min(ci.x + ci.w, cj.x + cj.w);
                int iy2 = std::min(ci.y + ci.h, cj.y + cj.h);
                int iw = ix2 - ix1, ih = iy2 - iy1;
                int64_t inter = (iw > 0 && ih > 0) ? (int64_t)iw * ih : 0;
                int64_t area_j = (int64_t)cj.w * cj.h;
                // Cover IoU >= 50%
                if (2 * inter >= area_i || 2 * inter >= area_j) {
                    if (ci.neighbors < cj.neighbors ||
                        (ci.neighbors == cj.neighbors && i < j)) {
                        rejected = true;
                        break;
                    }
                }
            }
            if (rejected) continue;
            if (ci.neighbors < group_threshold) continue;
            DetectionRect r;
            r.x = ci.x; r.y = ci.y; r.width = ci.w; r.height = ci.h;
            r.confidence = ci.score;
            r.reserved[0] = ci.neighbors;  
            out.push_back(r);
        }
        return out;
    }

    static void ExtractShapeIndexedFeatures(
        std::vector<int32_t>& out_indices,
        const uint8_t* img_base, int img_w, int img_h, int img_stride,
        const std::vector<float>& current_shape,
        const DetectionRect& bbox,
        const float* scale_rot,
        const uint8_t* ferns_stage_base)
    {
        int limit_x = img_w - 1;
        int limit_y = img_h - 1;
        float half_w = static_cast<float>(bbox.width) * 0.5f;
        float half_h = static_cast<float>(bbox.height) * 0.5f;
        float center_x = static_cast<float>(bbox.x) + half_w;
        float center_y = static_cast<float>(bbox.y) + half_h;

        float s0_w = half_w * scale_rot[0];
        float s1_w = half_w * scale_rot[1];
        float s2_h = half_h * scale_rot[2];
        float s3_h = half_h * scale_rot[3];

        out_indices.clear();
        out_indices.reserve(LANDMARK_POINTS * 5);

        int fern_global_idx = 0;
        for (int lmark_idx = 0; lmark_idx < (int)LANDMARK_POINTS; ++lmark_idx) {
            float abs_lmark_x = (half_w * current_shape[2 * lmark_idx]) + center_x;
            float abs_lmark_y = (half_h * current_shape[2 * lmark_idx + 1]) + center_y;

            for (int f = 0; f < 5; ++f) {
                const uint8_t* fern_base = ferns_stage_base + (fern_global_idx * 868);
                int32_t node_idx = 0;
                for (int depth = 0; depth < 5; ++depth) {
                    const uint8_t* node_ptr = fern_base + (node_idx * 28);
                    float f0, f1, f2, f3;
                    std::memcpy(&f0, node_ptr + 0, 4);
                    std::memcpy(&f1, node_ptr + 4, 4);
                    std::memcpy(&f2, node_ptr + 8, 4);
                    std::memcpy(&f3, node_ptr + 12, 4);

                    int p1_x = std::max(0, std::min(limit_x, static_cast<int>((s0_w * f0) + (f1 * s1_w) + abs_lmark_x)));
                    int p1_y = std::max(0, std::min(limit_y, static_cast<int>((s3_h * f1) + (s2_h * f0) + abs_lmark_y)));
                    int p2_x = std::max(0, std::min(limit_x, static_cast<int>((s0_w * f2) + (f3 * s1_w) + abs_lmark_x)));
                    int p2_y = std::max(0, std::min(limit_y, static_cast<int>((s3_h * f3) + (s2_h * f2) + abs_lmark_y)));

                    int pixel_diff = static_cast<int>(img_base[p1_y * img_stride + p1_x]) -
                                     static_cast<int>(img_base[p2_y * img_stride + p2_x]);

                    int16_t threshold;
                    std::memcpy(&threshold, node_ptr + 16, 2);
                    if (pixel_diff >= threshold) {
                        std::memcpy(&node_idx, node_ptr + 24, 4);
                    } else {
                        std::memcpy(&node_idx, node_ptr + 20, 4);
                    }
                }
                int local_leaf_idx = node_idx - 31;
                int absolute_leaf_idx = (fern_global_idx * 32) + local_leaf_idx;
                out_indices.push_back(absolute_leaf_idx);
                fern_global_idx++;
            }
        }
    }

    // (y, ch, x): addr = base + y*(11*stride) + ch*stride + x
    // "Hot" data is written only to the caller's private `patched_stages` and does not touch the shared model;     
    // the detection path allows for concurrent read-only access.
    void PatchPointersForScaleLocal(std::vector<PatchedStage>& patched_stages,
                                    const uint8_t* feature_map_base, int feature_stride) const {
        const int channel_stride = MAX_FEATURE_CHANNELS * feature_stride; // = row_stride
        patched_stages.resize(stages_.size());
        for (size_t s = 0; s < stages_.size(); ++s) {
            const CascadeStage& model_stage = stages_[s];
            PatchedStage& out_stage = patched_stages[s];
            out_stage.threshold = model_stage.threshold;
            out_stage.trees.resize(model_stage.tree_count);
            const size_t node_count = model_stage.model_nodes.size();
            for (size_t i = 0; i < node_count; ++i) {
                const ModelNode& model = model_stage.model_nodes[i];
                RuntimeNode& runtime = out_stage.trees[i / 3].nodes[i % 3];
                runtime.ptrA = feature_map_base + (model.y1 * channel_stride) + (model.ch * feature_stride) + model.x1;
                runtime.ptrB = feature_map_base + (model.y2 * channel_stride) + (model.ch * feature_stride) + model.x2;
                runtime.threshold   = model.threshold;
                runtime.left_idx    = model.left_idx;
                runtime.right_idx   = model.right_idx;
                runtime.left_score  = model.left_score;
                runtime.right_score = model.right_score;
            }
            for (size_t t = 0; t < out_stage.trees.size(); ++t)
                out_stage.trees[t].reject_threshold = model_stage.tree_reject[t];
        }
    }

    FaceResult FaceAlignment68Landmarks(const uint8_t* img_ptr, int img_w, int img_h, int img_stride, const DetectionRect& bbox) const {
        FaceResult result;
        result.bbox = bbox;
        result.landmarks.resize(LANDMARK_POINTS);

        std::vector<float> current_shape = mean_shape_;
        std::vector<int32_t> extracted_fern_indices;
        float shape_sim_scale[4];

        for (int stage = 0; stage < 5; ++stage) {
            EstimateShapeSimilarityTransform(shape_sim_scale, current_shape.data(), mean_shape_.data());
            ExtractShapeIndexedFeatures(
                extracted_fern_indices, img_ptr, img_w, img_h, img_stride,
                current_shape, bbox, shape_sim_scale, ferns_data_.data() + (stage * 340 * 868));

            const float K_WEIGHT_SCALE = 0.0001f;
            float s0 = shape_sim_scale[0] * K_WEIGHT_SCALE;
            float s1 = shape_sim_scale[1] * K_WEIGHT_SCALE;
            float s2 = shape_sim_scale[2] * K_WEIGHT_SCALE;
            float s3 = shape_sim_scale[3] * K_WEIGHT_SCALE;

            const uint8_t* weights_stage_ptr = weights_data_.data() + (stage * 68 * 21760);
            for (int p = 0; p < (int)LANDMARK_POINTS; ++p) {
                int accum_dx = 0, accum_dy = 0;
                const uint8_t* row_weights = weights_stage_ptr + (p * 21760);
                for (int j = 0; j < 340; ++j) {
                    int absolute_leaf_idx = extracted_fern_indices[j];
                    accum_dx += static_cast<int8_t>(row_weights[absolute_leaf_idx]);
                    accum_dy += static_cast<int8_t>(row_weights[absolute_leaf_idx + 10880]);
                }
                float f_dx = static_cast<float>(accum_dx);
                float f_dy = static_cast<float>(accum_dy);
                current_shape[2 * p]     += (f_dx * s0) + (f_dy * s1);
                current_shape[2 * p + 1] += (f_dx * s2) + (f_dy * s3);
            }
        }

        float half_w = static_cast<float>(bbox.width) * 0.5f;
        float half_h = static_cast<float>(bbox.height) * 0.5f;
        float c_x = static_cast<float>(bbox.x) + half_w;
        float c_y = static_cast<float>(bbox.y) + half_h;
        for (int i = 0; i < (int)LANDMARK_POINTS; ++i) {
            result.landmarks[i].first  = (half_w * current_shape[2 * i]) + c_x;
            result.landmarks[i].second = (half_h * current_shape[2 * i + 1]) + c_y;
        }
        return result;
    }

public:
    bool loadModel(const std::string& cascade_path,
                   const std::string& mean_shape_path,
                   const std::string& ferns_path,
                   const std::string& weights_path)
    {
        try {
            // Mean Shape
            std::vector<uint8_t> raw_mean;
            read_file(mean_shape_path, raw_mean);
            if (raw_mean.size() != LANDMARK_POINTS * 2 * sizeof(float))
                throw std::runtime_error("Mean Shape Abnormal model dimensions");
            mean_shape_.resize(LANDMARK_POINTS * 2);
            std::memcpy(mean_shape_.data(), raw_mean.data(), raw_mean.size());

            // Ferns & Weights
            read_file(ferns_path, ferns_data_);
            read_file(weights_path, weights_data_);
            if (ferns_data_.size() != FERN_STAGE_COUNT * FERN_FEATURES_PER_STAGE * FERN_NODE_SIZE)
                throw std::runtime_error("Ferns Abnormal model dimensions");
            if (weights_data_.size() != FERN_STAGE_COUNT * LANDMARK_POINTS * WEIGHT_ROW_SIZE)
                throw std::runtime_error("Weights Abnormal model dimensions");

            // Cascade Classifier
            std::vector<uint8_t> raw_cascade;
            read_file(cascade_path, raw_cascade);

            size_t offset = 0;
            auto read_i32 = [&](int32_t& val) {
                if (offset + 4 > raw_cascade.size()) throw std::runtime_error("EOF: Model file corrupted");
                std::memcpy(&val, raw_cascade.data() + offset, 4);
                offset += 4;
            };
            auto skip_bytes = [&](size_t n) {
                if (offset + n > raw_cascade.size()) throw std::runtime_error("EOF: Model file corrupted");
                offset += n;
            };

            // 全局头: (num_stages, win_w, win_h)
            int32_t num_stages;
            read_i32(num_stages);
            read_i32(win_w_);
            read_i32(win_h_);
            if (num_stages <= 0 || num_stages > 64)
                throw std::runtime_error("stage quantity anomaly");

            stages_.clear();
            stages_.resize(num_stages);

            for (int s = 0; s < num_stages; ++s) {
                CascadeStage& stage = stages_[s];
                read_i32(stage.tree_count);
                if (stage.tree_count <= 0 || stage.tree_count > 4096)
                    throw std::runtime_error("tree quantity anomaly");

                stage.model_nodes.reserve(stage.tree_count * 3);
                stage.tree_reject.reserve(stage.tree_count);

                for (int t = 0; t < stage.tree_count; ++t) {
                    for (int n = 0; n < 3; ++n) {   // Depth-2 fixed 3-node
                        ModelNode mn;
                        read_i32(mn.x1); read_i32(mn.y1);
                        read_i32(mn.x2); read_i32(mn.y2);
                        read_i32(mn.ch);
                        read_i32(mn.threshold);
                        read_i32(mn.left_idx);
                        read_i32(mn.right_idx);
                        read_i32(mn.left_score);
                        read_i32(mn.right_score);
                        skip_bytes(16);             // 16 zero padding
                        stage.model_nodes.push_back(mn);
                    }
                    int32_t reject_thr, reserved0;
                    read_i32(reject_thr);           // Tree Tail: Early Exit Threshold
                    read_i32(reserved0);            // Tree Tail: Retain (Constant 0)
                    stage.tree_reject.push_back(reject_thr);
                }
                // Read 16B stage tail block: first int32 = stage threshold.
                stage.threshold = stage.tree_reject.empty() ? 0 : 0;
                {
                    int32_t dummy;
                    read_i32(stage.threshold);
                    read_i32(dummy); read_i32(dummy); read_i32(dummy);
                }
            }

            if (offset != raw_cascade.size())
                std::cerr << "[WARNING] cascade resolution ends at " << offset
                          << ", file size " << raw_cascade.size() << std::endl;

            is_loaded_ = true;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Model loading failed: " << e.what() << std::endl;
            is_loaded_ = false;
            mean_shape_.clear();
            ferns_data_.clear();
            weights_data_.clear();
            stages_.clear();
            return false;
        }
    }

private:
    // Full-scale scanning + GroupRectangles; returns face bounding boxes after clustering.
    std::vector<DetectionRect> ScanAndGroup(const uint8_t* img_ptr, int width, int height, int stride,
                                            int min_size, int max_size, float scale_step_ratio,
                                            int min_neighbors) const
    {
        if (!is_loaded_ || !img_ptr) return {};

        int base_win_w = win_w_;
        int base_win_h = win_h_;

        int start_win_size = std::max(base_win_w, min_size);
        int max_img_limit = (base_win_w * height) / base_win_h;
        int end_win_size = std::min(width, max_img_limit);

        if (max_size > 0 && max_size < end_win_size) end_win_size = max_size;
        if (end_win_size < start_win_size) return {};

        int aligned_stride = ALIGN_BYTES * (width / ALIGN_BYTES + (width % ALIGN_BYTES > 0 ? 1 : 0));

        std::vector<uint8_t> feature_buf(aligned_stride * height + ALIGN_BYTES, 0);
        std::vector<uint8_t> scaled_buf(MAX_FEATURE_CHANNELS * aligned_stride * height + ALIGN_BYTES, 0);
        std::vector<PatchedStage> patched_stages;   // Originally intended to leverage private resources and enable cross-scale capacity reuse.

        // ((start << 10) + win/2) / win
        int scale_loop_ratio = ((start_win_size << 10) + base_win_w / 2) / base_win_w;
        int target_limit_1024 = (end_win_size << 10) / base_win_w;
        const int initial_ratio = scale_loop_ratio;
        int fixed_scale_step = static_cast<int>(scale_step_ratio * 1024.0f + 0.5f);

        std::vector<DetectionRect> raw_candidates;

        while (scale_loop_ratio <= target_limit_1024) {
            int scaled_w = (scale_loop_ratio / 2 + (width << 10)) / scale_loop_ratio;
            int scaled_h = (scale_loop_ratio / 2 + (height << 10)) / scale_loop_ratio;
            int alignment_16 = ALIGN_BYTES * (scaled_w / ALIGN_BYTES + (scaled_w % ALIGN_BYTES > 0 ? 1 : 0));

            BuildPyramidScale(img_ptr, width, height, stride, feature_buf.data(), scaled_w, scaled_h, alignment_16);
            Extract11ChannelFeatures(feature_buf.data(), scaled_w, scaled_h, alignment_16, scaled_buf.data());

            PatchPointersForScaleLocal(patched_stages, scaled_buf.data(), alignment_16);

            int scan_limit_x = scaled_w - base_win_w;
            int scan_limit_y = scaled_h - base_win_h;

            // Row offset must span 11 channels (row-interleaved layout).
            int64_t current_y_offset = 0;
            const int row_block = MAX_FEATURE_CHANNELS * alignment_16;
            // Adaptive step size: Step size is 2 if the scale ratio is <= 2× the initial ratio; otherwise, it is 1.
            int step = (scale_loop_ratio <= 2 * initial_ratio) ? 2 : 1;

            for (int y = 0; y <= scan_limit_y; y += step) {
                for (int x = 0; x <= scan_limit_x; ) {
                    int32_t margin = 0;
                    // Accepted if all 14 stages are passed; double-step if stage 0 rejects.
                    int ret = EvaluateCascadeWindow(patched_stages, current_y_offset + x, margin);
                    if (ret > 0) {
                        DetectionRect rect;
                        rect.x = (scale_loop_ratio * x + K_CONST_512[0]) >> 10;
                        rect.y = (scale_loop_ratio * y + K_CONST_512[0]) >> 10;
                        rect.width  = (scale_loop_ratio * base_win_w + K_CONST_512[0]) >> 10;
                        rect.height = (scale_loop_ratio * base_win_h + K_CONST_512[0]) >> 10;
                        rect.confidence = margin;
                        raw_candidates.push_back(rect);
                        x += step;
                    } else if (ret == 0) {
                        x += 2 * step;   // Level 0 Rejection: Skip an extra step.
                    } else {
                        x += step;
                    }
                }
                current_y_offset += static_cast<int64_t>(row_block) * step;
            }

            scale_loop_ratio = (fixed_scale_step * scale_loop_ratio + K_CONST_512[0]) >> 10;
        }

        // The GroupRectangles clustering method (based on neighbor hit counts) can be replaced with a more modern NMS approach.
        return GroupRectangles(raw_candidates, min_neighbors);
    }

public:
    // The sole external inference interface: returns a value, with the caller owning the result; 
    // supports concurrent read-only calls after `loadModel` completes.
    std::vector<FaceDetection> detect_face_frontal(const uint8_t* img_ptr, int width, int height, int stride,
                                            const DetectOptions& opt = DetectOptions()) const
    {
        std::vector<DetectionRect> final_faces =
            ScanAndGroup(img_ptr, width, height, stride, opt.min_size, opt.max_size,
                         opt.scale_step_ratio, opt.min_neighbors);
        std::vector<FaceDetection> out;
        out.reserve(final_faces.size());
        for (const auto& r : final_faces) {
            FaceDetection f;
            f.x = r.x; f.y = r.y; f.width = r.width; f.height = r.height;
            f.score = r.confidence;
            f.neighbors = r.reserved[0];
            if (opt.with_landmarks) {
                FaceResult fr = FaceAlignment68Landmarks(img_ptr, width, height, stride, r);
                std::copy(fr.landmarks.begin(), fr.landmarks.end(), f.landmarks.begin());
                f.has_landmarks = true;
            }
            out.push_back(std::move(f));
        }
        return out;
    }

};
