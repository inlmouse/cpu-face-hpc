#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

// SIMD 指令集支持
#include <emmintrin.h> 
#include <xmmintrin.h> 
#include <smmintrin.h>

// =================================================================
// 常量与底层结构定义
// =================================================================

constexpr size_t LANDMARK_POINTS = 68;
constexpr size_t FERN_STAGE_COUNT = 5;
constexpr size_t FERN_FEATURES_PER_STAGE = 340;
constexpr size_t FERN_NODE_SIZE = 868;
constexpr size_t WEIGHT_ROW_SIZE = 21760;

constexpr size_t MAX_FEATURE_CHANNELS = 11; 
constexpr size_t ALIGN_BYTES = 16;

const alignas(16) int32_t K_CONST_512[4] = { 512, 512, 512, 512 }; 
const alignas(16) float K_SCALE_MASK[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

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
    std::vector<std::pair<float, float>> landmarks; // 存放 68 个点的 (x, y) 物理坐标
};

struct WeakTree {
    int32_t num_nodes;
    std::vector<uint8_t> raw_node_bytes; 
};

// -----------------------------------------------------------------
// 【冷数据区】：仅在 LoadModel 和 Scale Patch 期间访问
// -----------------------------------------------------------------
struct ModelNode {
    int32_t ch1, x1, y1;
    int32_t ch2, x2, y2;
    int32_t threshold;
    int32_t left_val;
    int32_t right_val;
};

// -----------------------------------------------------------------
// 【热数据区】：滑动窗口最内层循环的终极数据结构 (完美契合 CPU Cache)
// 64 位系统下：8 + 8 + 4 + 4 + 4 = 28 Bytes (补齐后刚好 32 Bytes)
// 一个 64B 的 L1 Cache Line 完美吞下 2 个 Node！
// -----------------------------------------------------------------
struct RuntimeNode {
    const uint8_t* ptrA;
    const uint8_t* ptrB;
    int32_t threshold;
    int32_t left_val;
    int32_t right_val;
};

// 确保内存布局紧凑
struct CascadeStage {
    int32_t win_w;           // 扫描窗口基准宽度
    int32_t win_h;           // 扫描窗口基准高度
    int32_t stage_threshold;
    int32_t tree_count;       // 树的数量
    std::vector<ModelNode> model_nodes;     // 保存离线蓝图
    std::vector<RuntimeNode> runtime_nodes; // 运行时极速查表缓冲
};



// =================================================================
// 面向对象的引擎封装
// =================================================================

class FaceDetector {
private:
    CascadeStage cascade_stage_;
    std::vector<float> mean_shape_;
    std::vector<uint8_t> ferns_data_;
    std::vector<uint8_t> weights_data_;
    
    bool is_loaded_ = false;

    static void read_file(const std::string& path, std::vector<uint8_t>& out) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) {
            throw std::runtime_error("无法打开文件: " + path);
        }
        
        fin.seekg(0, std::ios::end);
        size_t sz = static_cast<size_t>(fin.tellg());
        fin.seekg(0, std::ios::beg);
        
        out.resize(sz);
        if (sz > 0) {
            fin.read(reinterpret_cast<char*>(out.data()), sz);
        }
    }

    // --- 内部算法组件 ---

    void ParallelBilinearResample(
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
                        // 使用 loadu 规避堆内存未 16 字节对齐引发的 Segfault
                        __m128i src0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd]));
                        __m128i src1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd + 4]));
                        __m128i src2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd + 8]));
                        __m128i src3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&line_curr[x_simd + 12]));
                        
                        src0 = _mm_sra_epi32(_mm_sll_epi32(_mm_add_epi32(src0, k_512), 10), 20);
                        src1 = _mm_sra_epi32(_mm_sll_epi32(_mm_add_epi32(src1, k_512), 10), 20);
                        src2 = _mm_sra_epi32(_mm_sll_epi32(_mm_add_epi32(src2, k_512), 10), 20);
                        src3 = _mm_sra_epi32(_mm_sll_epi32(_mm_add_epi32(src3, k_512), 10), 20);
                        
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
                        
                        // 使用 SSE4.1 核心指令
                        __m128i mix0 = _mm_add_epi32(_mm_sll_epi32(_mm_add_epi32(c0, k_512), 10), _mm_mullo_epi32(_mm_sub_epi32(n0, c0), v_weight_y));
                        __m128i mix1 = _mm_add_epi32(_mm_sll_epi32(_mm_add_epi32(c1, k_512), 10), _mm_mullo_epi32(_mm_sub_epi32(n1, c1), v_weight_y));
                        
                        mix0 = _mm_sra_epi32(mix0, 20); 
                        mix1 = _mm_sra_epi32(mix1, 20);
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

    void BuildPyramidScale(
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

    void Extract11ChannelFeatures(const uint8_t* src_buf, int width, int height, size_t element_w, uint8_t* out_multi_channel) 
    {
        size_t row_stride = 11 * element_w; 
        
        // =================================================================
        // 通道 0：原始灰度投影
        // =================================================================
        if (height > 0) {
            const uint8_t* src_ptr = src_buf; 
            uint8_t* dst_ptr = out_multi_channel;
            for (int y = 0; y < height; ++y) {
                std::memcpy(dst_ptr, src_ptr, element_w);
                src_ptr += element_w;
                dst_ptr += row_stride;
            }
        }

        // =================================================================
        // 通道 1：2x2 微小邻域 Box 均值模糊
        // =================================================================
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
                    
                    // 均值平滑计算
                    row_ptr[element_w + x] = static_cast<uint8_t>((((p00 + p10 + 1) >> 1) + ((p01 + p11 + 1) >> 1) + 1) >> 1);
                }
                row_ptr += row_stride;
            }
        }

        // =================================================================
        // 通道 2 & 通道 3-6 (二次平滑与短距一阶 HOG 边缘差分通道)
        // =================================================================
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
                        int32_t v0 = p_curr[0];
                        int32_t v2 = p_curr[2];
                        p_curr++;

                        // 水平、垂直、主对角线、副对角边缘提取 (加255除2，彻底规避整型下溢)
                        p_curr[offset_c3 - 1] = static_cast<uint8_t>((v2 - v0 + 255) / 2);
                        p_curr[offset_c3] = static_cast<uint8_t>((p_curr[2 * row_stride - 1] - p_curr[-1] + 255) / 2);
                        p_curr[4 * element_w - 1] = static_cast<uint8_t>((p_curr[offset_c2] - p_curr[-1] + 255) / 2);
                        p_curr[5 * element_w - 1] = static_cast<uint8_t>((p_curr[2 * row_stride - 1] - p_curr[1] + 255) / 2);
                    }
                }
                grad_row_ptr += row_stride;
            }
        }

        // =================================================================
        // 通道 7-10 (跨度为4像素的长距离稀疏 LBP 纹理通道)
        // =================================================================
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
                        int32_t v0 = p_curr[0];
                        int32_t v4 = p_curr[4];
                        p_curr++;

                        p_curr[offset_c6 - 1] = static_cast<uint8_t>((v4 - v0 + 255) / 2);
                        p_curr[offset_c6] = static_cast<uint8_t>((p_curr[4 * row_stride - 1] - p_curr[-1] + 255) / 2);
                        p_curr[offset_c7 - 1] = static_cast<uint8_t>((p_curr[4 * row_stride - offset_c6 + 4] - p_curr[-1] + 255) / 2);
                        p_curr[offset_c8 - 1] = static_cast<uint8_t>((p_curr[4 * row_stride - 1] - p_curr[3] + 255) / 2);
                    }
                }
                long_row_ptr += row_stride;
            }
        }
    }

    int32_t EvaluateWeakClassifierTree(int current_window_offset, const CascadeStage& stage) 
    {
        // 直指热数据缓存区，命中率极高
        const RuntimeNode* trees = stage.runtime_nodes.data();
        int score = 0;
        int remain = stage.tree_count;

        // =================================================================
        // 极速闭包 (只有纯粹的指针解引用，无任何乘法或数组维度计算)
        // =================================================================
        auto eval_tree = [&](const RuntimeNode* tree) -> int32_t {
            // Root层
            int32_t diff = tree[0].ptrA[current_window_offset] - tree[0].ptrB[current_window_offset];
            const RuntimeNode* child = &tree[ (diff >= tree[0].threshold) ? tree[0].left_val : tree[0].right_val ];
            
            // Child层
            int32_t c_diff = child->ptrA[current_window_offset] - child->ptrB[current_window_offset];
            return (c_diff >= child->threshold) ? child->left_val : child->right_val;
        };

        // =================================================================
        // 黄金短路 (Top-4 Trees)
        // =================================================================
        int top_k = std::min(4, remain);
        for (int i = 0; i < top_k; ++i) {
            score += eval_tree(trees + i * 3);
        }
        
        if (score < stage.stage_threshold) {
            return 0; 
        }

        trees += top_k * 3;
        remain -= top_k;

        // =================================================================
        // 超标量乱序并发展开 (Unroll by 4)
        // =================================================================
        while (remain >= 4) {
            score += eval_tree(trees + 0);
            score += eval_tree(trees + 3);
            score += eval_tree(trees + 6);
            score += eval_tree(trees + 9);
            
            trees += 12;
            remain -= 4;
        }

        // 尾部残余收尾
        while (remain > 0) {
            score += eval_tree(trees);
            trees += 3;
            remain--;
        }

        return score;
    }

    // 修正：提供基础 2D 形状缩放比例估算 
    void EstimateShapeSimilarityTransform(float* out_scale, const float* current_shape, const float* mean_shape) {
        float mean_curr_x = 0.0f, mean_curr_y = 0.0f;
        float mean_base_x = 0.0f, mean_base_y = 0.0f;

        // 1. 计算两个形状的中心点
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

        // 2. 去中心化并累计二阶统计量 (严格复刻原 DLL 累加器顺序)
        for (size_t i = 0; i < LANDMARK_POINTS; ++i) {
            float x = current_shape[2 * i] - mean_curr_x;
            float y = current_shape[2 * i + 1] - mean_curr_y;
            float u = mean_shape[2 * i] - mean_base_x;
            float v = mean_shape[2 * i + 1] - mean_base_y;

            sum_xx += x * x; // Σxx
            sum_yy += y * y; // Σyy
            
            sum_uu += u * u; // Σuu
            sum_vv += v * v; // Σvv

            A += (u * x + v * y); // Σ(ux + vy)
            B += (u * y - v * x); // Σ(uy - vx)
        }

        // 3. 计算尺度 (Scale)
        float current_norm = std::sqrt(sum_xx + sum_yy);
        float mean_norm = std::sqrt(sum_uu + sum_vv);
        
        // 防除零保护 (1e-7f 用于浮点安全，不影响常规尺度计算)
        float scale = current_norm / (mean_norm + 1e-7f); 

        // 4. 计算旋转 (Rotation) - 彻底剔除三角函数
        float L = std::sqrt(A * A + B * B);
        float cos_theta = A / (L + 1e-7f);
        float sin_theta = B / (L + 1e-7f);

        // 5. 组合最终 Similarity 矩阵
        out_scale[0] = scale * cos_theta;  // S0
        out_scale[1] = -scale * sin_theta; // S1
        out_scale[2] = scale * sin_theta;  // S2
        out_scale[3] = scale * cos_theta;  // S3
    }

    // 替代原始代码中晦涩的邻居合并（GroupRectangles），在现代 CV 中，标准的 IoU（交并比） NMS 速度更快且效果更好。
    std::vector<DetectionRect> NonMaximumSuppression(std::vector<DetectionRect>& boxes, float iou_threshold = 0.4f) {
        if (boxes.empty()) return {};

        // 按照得分(置信度)从大到小排序
        std::sort(boxes.begin(), boxes.end(), [](const DetectionRect& a, const DetectionRect& b) {
            return a.confidence > b.confidence;
        });

        std::vector<DetectionRect> keep;
        std::vector<bool> suppressed(boxes.size(), false);

        for (size_t i = 0; i < boxes.size(); ++i) {
            if (suppressed[i]) continue;
            keep.push_back(boxes[i]);

            float area_i = static_cast<float>(boxes[i].width * boxes[i].height);

            for (size_t j = i + 1; j < boxes.size(); ++j) {
                if (suppressed[j]) continue;

                // 计算两个框的交集区域
                int xx1 = std::max(boxes[i].x, boxes[j].x);
                int yy1 = std::max(boxes[i].y, boxes[j].y);
                int xx2 = std::min(boxes[i].x + boxes[i].width, boxes[j].x + boxes[j].width);
                int yy2 = std::min(boxes[i].y + boxes[i].height, boxes[j].y + boxes[j].height);

                int w = std::max(0, xx2 - xx1);
                int h = std::max(0, yy2 - yy1);
                float inter = static_cast<float>(w * h);
                float area_j = static_cast<float>(boxes[j].width * boxes[j].height);
                
                // 计算交并比 IoU
                float iou = inter / (area_i + area_j - inter);

                // 如果重合度过高，淘汰低分框
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        return keep;
    }

    void ExtractShapeIndexedFeatures(
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
        out_indices.reserve(LANDMARK_POINTS * 5); // 68点 * 5棵树 = 全局 340 棵树

        int fern_global_idx = 0;
        for (int lmark_idx = 0; lmark_idx < LANDMARK_POINTS; ++lmark_idx) {
            float abs_lmark_x = (half_w * current_shape[2 * lmark_idx]) + center_x;
            float abs_lmark_y = (half_h * current_shape[2 * lmark_idx + 1]) + center_y;

            for (int f = 0; f < 5; ++f) {
                // 定位到当前这棵 868 字节的完全二叉树基址
                const uint8_t* fern_base = ferns_stage_base + (fern_global_idx * 868);
                
                int32_t node_idx = 0;
                // 【核心修复】：向下穿透 5 层完全二叉树
                for (int depth = 0; depth < 5; ++depth) {
                    // 每个内部节点刚好 28 字节 (16字节坐标 + 2字节阈值 + 2字节对齐 + 8字节左右分支偏移)
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

                    int pixel_diff = static_cast<int>(img_base[p2_y * img_stride + p2_x]) - 
                                    static_cast<int>(img_base[p1_y * img_stride + p1_x]);

                    int16_t threshold;
                    std::memcpy(&threshold, node_ptr + 16, 2);

                    if (pixel_diff >= threshold) {
                        std::memcpy(&node_idx, node_ptr + 24, 4); // 走右分支
                    } else {
                        std::memcpy(&node_idx, node_ptr + 20, 4); // 走左分支
                    }
                }
                
                // 5层走完，node_idx 必然指向叶子节点。根据原 IDA 反汇编逻辑：
                // Depth-5 的内部节点编号是 0~30，叶子节点存储的 left_val/right_val 实际上是 31~62
                // 减去 31 即可得到 local_leaf_idx (0~31)
                int local_leaf_idx = node_idx - 31; 
                
                // 将局部叶子索引映射到全局 10880 维二进制特征向量的绝对偏移
                int absolute_leaf_idx = (fern_global_idx * 32) + local_leaf_idx;
                
                out_indices.push_back(absolute_leaf_idx);
                fern_global_idx++;
            }
        }
    }

    void PatchPointersForScale(CascadeStage& stage, const uint8_t* feature_map_base, int feature_stride) {
        int channel_stride = MAX_FEATURE_CHANNELS * feature_stride;
        
        // 双路并发遍历：用蓝图生成运行时缓存
        size_t node_count = stage.model_nodes.size();
        for (size_t i = 0; i < node_count; ++i) {
            const auto& model = stage.model_nodes[i];
            auto& runtime = stage.runtime_nodes[i];

            // 彻底算死特征指针绝对物理基址
            runtime.ptrA = feature_map_base + (model.y1 * feature_stride) + (model.ch1 * channel_stride) + model.x1;
            runtime.ptrB = feature_map_base + (model.y2 * feature_stride) + (model.ch2 * channel_stride) + model.x2;
            
            // 直传热数据
            runtime.threshold = model.threshold;
            runtime.left_val  = model.left_val;
            runtime.right_val = model.right_val;
        }
    }

    FaceResult FaceAlignment68Landmarks(const uint8_t* img_ptr, int img_w, int img_h, int img_stride, const DetectionRect& bbox) {
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
                current_shape, bbox, shape_sim_scale, ferns_data_.data() + (stage * 340 * 868)
            );

            float s0 = shape_sim_scale[0], s1 = shape_sim_scale[1];
            float s2 = shape_sim_scale[2], s3 = shape_sim_scale[3];

            // 锁定当前阶段的 68x21760 字节权重矩阵基址
            const uint8_t* weights_stage_ptr = weights_data_.data() + (stage * 68 * 21760);
            
            for (int p = 0; p < LANDMARK_POINTS; ++p) {
                int accum_dx = 0, accum_dy = 0;
                
                // 锁定当前特征点 P 专属的回归矩阵行 (长度 21760 bytes)
                const uint8_t* row_weights = weights_stage_ptr + (p * 21760);
                
                // 【核心修复】：对于当前 Landmark，必须汇总全脸所有 340 棵树的叶子响应 (Global Regression)
                for (int j = 0; j < 340; ++j) {
                    // 取出刚才计算好的绝对叶子索引 (0 ~ 10879)
                    int absolute_leaf_idx = extracted_fern_indices[j];
                    
                    // 强转有符号 int8，前半段(0~10879)为 X 位移权重，后半段(+10880)为 Y 位移权重
                    accum_dx += static_cast<int8_t>(row_weights[absolute_leaf_idx]);
                    accum_dy += static_cast<int8_t>(row_weights[absolute_leaf_idx + 10880]);
                }
                
                float f_dx = static_cast<float>(accum_dx);
                float f_dy = static_cast<float>(accum_dy);
                
                // 使用正向 Similarity Transform，将 Normalized Delta 映射回当前的 Image Scale & Rotation
                current_shape[2 * p]     += (f_dx * s0) + (f_dy * s1);
                current_shape[2 * p + 1] += (f_dx * s2) + (f_dy * s3);
            }
        }

        // 最终还原回绝对图像物理坐标
        float half_w = static_cast<float>(bbox.width) * 0.5f;
        float half_h = static_cast<float>(bbox.height) * 0.5f;
        float c_x = static_cast<float>(bbox.x) + half_w;
        float c_y = static_cast<float>(bbox.y) + half_h;

        for (int i = 0; i < LANDMARK_POINTS; ++i) {
            result.landmarks[i].first  = (half_w * current_shape[2 * i]) + c_x;
            result.landmarks[i].second = (half_h * current_shape[2 * i + 1]) + c_y;
        }

        return result;
    }

public:
    // 安全的模型加载器，完全基于 RAII
    bool loadModel(const std::string& cascade_path, 
                   const std::string& mean_shape_path, 
                   const std::string& ferns_path, 
                   const std::string& weights_path) 
    {
        try {
            // 1. 加载 Mean Shape (544 字节)
            std::vector<uint8_t> raw_mean;
            read_file(mean_shape_path, raw_mean);
            if (raw_mean.size() != LANDMARK_POINTS * 2 * sizeof(float)) {
                throw std::runtime_error("Mean Shape 模型尺寸异常");
            }
            mean_shape_.resize(LANDMARK_POINTS * 2);
            std::memcpy(mean_shape_.data(), raw_mean.data(), raw_mean.size());

            // 2. 加载 Ferns & Weights
            read_file(ferns_path, ferns_data_);
            read_file(weights_path, weights_data_);

            // 3. 加载 Cascade 分类器
            std::vector<uint8_t> raw_cascade;
            read_file(cascade_path, raw_cascade);
            
            size_t offset = 0;
            auto read_i32 = [&](int32_t& val) {
                if (offset + 4 > raw_cascade.size()) throw std::runtime_error("EOF: 模型文件损坏");
                std::memcpy(&val, raw_cascade.data() + offset, 4);
                offset += 4;
            };

            // 读取全局头部，精确对应原版二进制结构
            read_i32(cascade_stage_.tree_count);
            read_i32(cascade_stage_.win_w);           // 补回了窗口宽高的读取与保存
            read_i32(cascade_stage_.win_h);
            read_i32(cascade_stage_.stage_threshold); // 读取真实的强分类器阈值

            cascade_stage_.model_nodes.clear();
            cascade_stage_.model_nodes.reserve(cascade_stage_.tree_count * 3); // Depth-2 树固定 3 个 Node

            // 扁平化反序列化所有节点
            for (int i = 0; i < cascade_stage_.tree_count; ++i) {
                int32_t num_nodes;
                read_i32(num_nodes); 
                
                for (int j = 0; j < num_nodes; ++j) {
                    ModelNode mn;
                    read_i32(mn.ch1); read_i32(mn.x1); read_i32(mn.y1);
                    read_i32(mn.ch2); read_i32(mn.x2); read_i32(mn.y2);
                    read_i32(mn.threshold);
                    read_i32(mn.left_val);
                    read_i32(mn.right_val);
                    
                    // 跨过 20 字节的老旧废弃指针空间
                    offset += 20; 
                    
                    cascade_stage_.model_nodes.push_back(mn);
                }
            }

            // 提前分配好 Runtime 热数据区的空间，等待 PatchPointersForScale 填充
            cascade_stage_.runtime_nodes.resize(cascade_stage_.model_nodes.size());

            is_loaded_ = true;
            return true;
        } 
        catch (const std::exception& e) {
            std::cerr << "模型加载失败: " << e.what() << std::endl;
            is_loaded_ = false;
            mean_shape_.clear();
            ferns_data_.clear();
            weights_data_.clear();
            cascade_stage_.model_nodes.clear();
            cascade_stage_.runtime_nodes.clear();
            return false;
        }
    }

    std::vector<FaceResult> detect(const uint8_t* img_ptr, int width, int height, int stride, 
                                   int min_size = 48, int max_size = 0, float scale_step_ratio = 1.2f) 
    {
        if (!is_loaded_ || !img_ptr) return {};
        
        // 取出离线模型的基准检测窗口大小
        int base_win_w = cascade_stage_.win_w; 
        int base_win_h = cascade_stage_.win_h;
        
        int start_win_size = std::max(base_win_w, min_size);
        int max_img_limit = (base_win_w * height) / base_win_h;
        int end_win_size = std::min(width, max_img_limit);
        
        if (max_size > 0 && max_size < end_win_size) end_win_size = max_size;
        if (end_win_size < start_win_size) return {};

        // 统一计算 16 字节对齐步长
        int aligned_stride = ALIGN_BYTES * (width / ALIGN_BYTES + (width % ALIGN_BYTES > 0 ? 1 : 0));
        
        // 预先一次性分配最大所需的特征缓冲区，避免循环内重新 malloc 导致内存碎片化
        std::vector<uint8_t> feature_buf(aligned_stride * height + ALIGN_BYTES, 0); 
        std::vector<uint8_t> scaled_buf(MAX_FEATURE_CHANNELS * aligned_stride * height + ALIGN_BYTES, 0); 

        // 转换为定点数循环变量
        int scale_loop_ratio = (start_win_size << 10) / base_win_w; 
        int target_limit_1024 = (end_win_size << 10) / base_win_w;
        int fixed_scale_step = static_cast<int>(scale_step_ratio * 1024.0f + 0.5f);

        std::vector<DetectionRect> raw_candidates;

        // -----------------------------------------------------------------
        // 阶段一：图像金字塔与多尺度滑动窗口极速扫描
        // -----------------------------------------------------------------
        while (scale_loop_ratio <= target_limit_1024) {
            // 计算当前尺度层的真实宽高与对齐步长
            int scaled_w = (scale_loop_ratio / 2 + (width << 10)) / scale_loop_ratio;
            int scaled_h = (scale_loop_ratio / 2 + (height << 10)) / scale_loop_ratio;
            int alignment_16 = ALIGN_BYTES * (scaled_w / ALIGN_BYTES + (scaled_w % ALIGN_BYTES > 0 ? 1 : 0));

            // 1. 构建本层金字塔重采样图
            BuildPyramidScale(img_ptr, width, height, stride, feature_buf.data(), scaled_w, scaled_h, alignment_16);
            
            // 2. 提取 11 通道的高频边缘与长距纹理特征图
            Extract11ChannelFeatures(feature_buf.data(), scaled_w, scaled_h, alignment_16, scaled_buf.data());

            // 3. 【绝对核心】滑动扫描前，对本层特征图打一次“运行时指针补丁”！
            PatchPointersForScale(cascade_stage_, scaled_buf.data(), alignment_16);

            // 设置扫描边界
            int scan_limit_x = scaled_w - base_win_w;
            int scan_limit_y = scaled_h - base_win_h;
            
            // X, Y 滑动基准偏移量
            int current_y_offset = 0; 
            int step_x = 1; // 可改为 2 以进一步加速牺牲少许召回率
            int step_y = 1;

            // 4. 2D 滑动窗口密集扫描 (此时内存寻址开销已被彻底榨干)
            for (int y = 0; y <= scan_limit_y; y += step_y) {
                for (int x = 0; x <= scan_limit_x; x += step_x) {
                    
                    // 传入一维 offset，函数内部纯指针取值计算，无任何乘法或多维寻址！
                    int score = EvaluateWeakClassifierTree(current_y_offset + x, cascade_stage_);
                    
                    if (score > 0) {
                        // 利用定点数反推映射回原始图像绝对物理坐标系
                        DetectionRect rect;
                        rect.x = (scale_loop_ratio * x + K_CONST_512[0]) >> 10;
                        rect.y = (scale_loop_ratio * y + K_CONST_512[0]) >> 10;
                        rect.width  = (scale_loop_ratio * base_win_w + K_CONST_512[0]) >> 10;
                        rect.height = (scale_loop_ratio * base_win_h + K_CONST_512[0]) >> 10;
                        rect.confidence = score; // 保留真实得分供 NMS 筛选
                        
                        raw_candidates.push_back(rect);
                    }
                }
                // 每换一行，当前窗口的基础 Y 偏移仅增加一个单通道的物理跨度
                current_y_offset += alignment_16 * step_y; 
            }
            
            // 尺度演进：定点数乘法推进至下一层金字塔
            scale_loop_ratio = (fixed_scale_step * scale_loop_ratio + K_CONST_512[0]) >> 10;
        }

        // -----------------------------------------------------------------
        // 阶段二：IoU NMS（非极大值抑制）消除高度重叠候选框
        // -----------------------------------------------------------------
        std::vector<DetectionRect> final_faces = NonMaximumSuppression(raw_candidates, 0.4f);

        // -----------------------------------------------------------------
        // 阶段三：针对高置信度检测框，启动 68点主动形状回归对齐
        // -----------------------------------------------------------------
        std::vector<FaceResult> final_results;
        for (const auto& face_box : final_faces) {
            FaceResult res = FaceAlignment68Landmarks(img_ptr, width, height, stride, face_box);
            final_results.push_back(res);
        }

        return final_results;
    }
};

// 暴露给外部 C API
extern "C" {
    FaceDetector* create_detector() { return new FaceDetector(); }
    void destroy_detector(FaceDetector* det) { delete det; }
    int load_detector_models(FaceDetector* det, const char* p1, const char* p2, const char* p3, const char* p4) {
        return det->loadModel(p1, p2, p3, p4) ? 1 : 0;
    }
    void run_detection(FaceDetector* det, uint8_t* img, int w, int h, int stride) {
        det->detect(img, w, h, stride);
    }
}
