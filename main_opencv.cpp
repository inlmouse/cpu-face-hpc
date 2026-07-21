// OpenCV-based usage example: Using the value-returning API detect_faces
// Usage:
//   main_opencv <image_file> [model_dir=.] [output.jpg] [threads=0]
// threads: 0 = use hardware_concurrency; 1 = run single-thread example only
// Note: This is an example of direct inclusion in a single translation unit;
// for production code, it is recommended to split into facedetect.h/.cpp and link.
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "face_frontal.cpp"

static std::string join_path(const std::string& dir, const char* name) {
    if (dir.empty() || dir == ".") return name;
    if (dir.back() == '/' || dir.back() == '\\') return dir + name;
    return dir + "/" + name;
}

static void draw_faces(cv::Mat& canvas, const std::vector<FaceDetection>& faces) {
#if defined(CV_VERSION_MAJOR) && CV_VERSION_MAJOR >= 3
    const int AA = cv::LINE_AA;
#else
    const int AA = CV_AA;
#endif
    cv::Mat overlay = canvas.clone();

    // OpenCV uses BGR; match the RGB aesthetics of main_plain.cpp
    const cv::Scalar kJaw(208, 224, 72);      // Cyan
    const cv::Scalar kBrowR(64, 176, 255);    // Orange
    const cv::Scalar kBrowL(96, 120, 255);    // Coral
    const cv::Scalar kNose1(80, 232, 255);    // Bright Yellow
    const cv::Scalar kNose2(64, 214, 255);    // Golden Yellow
    const cv::Scalar kEyeR(255, 205, 80);     // Sky Blue
    const cv::Scalar kEyeL(255, 132, 190);    // Purple
    const cv::Scalar kMouthO(122, 78, 255);   // Magenta
    const cv::Scalar kMouthI(198, 176, 255);  // Light Pink
    const cv::Scalar kRect(150, 255, 70);     // Soft Green

    auto draw_poly = [&](const FaceDetection& f, int begin, int end, bool closed, const cv::Scalar& color) {
        std::vector<cv::Point> pts;
        pts.reserve(end - begin + 1);
        for (int i = begin; i <= end; ++i) {
            pts.emplace_back(static_cast<int>(f.landmarks[i].first),
                             static_cast<int>(f.landmarks[i].second));
        }
        cv::polylines(overlay, pts, closed, color, 2, AA);
        for (const cv::Point& p : pts) cv::circle(overlay, p, 2, color, -1, AA);
    };

    for (const FaceDetection& f : faces) {
        cv::rectangle(overlay, cv::Rect(f.x, f.y, f.width, f.height), kRect, 2, AA);
        if (!f.has_landmarks) continue;
        draw_poly(f, 0, 16, false, kJaw);       // Jawline
        draw_poly(f, 17, 21, false, kBrowR);    // Right Eyebrow
        draw_poly(f, 22, 26, false, kBrowL);    // Left Eyebrow
        draw_poly(f, 27, 30, false, kNose1);    // Nose Bridge
        draw_poly(f, 31, 35, false, kNose2);    // Nose Base
        draw_poly(f, 36, 41, true,  kEyeR);     // Right Eye
        draw_poly(f, 42, 47, true,  kEyeL);     // Left Eye
        draw_poly(f, 48, 59, true,  kMouthO);   // Outer Lip
        draw_poly(f, 60, 67, true,  kMouthI);   // Inner Lip
    }

    // Semi-transparent overlay to avoid harsh blocks of color,
    // approximating the soft blend look of main_plain
    cv::addWeighted(overlay, 0.78, canvas, 0.22, 0.0, canvas);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <image_file_name> [model_dir=.] [output.jpg] [threads=0]\n", argv[0]);
        return -1;
    }
    const std::string image_path = argv[1];
    const std::string model_dir = argc > 2 ? argv[2] : ".";
    const std::string output_path = argc > 3 ? argv[3] : "";
    int threads = argc > 4 ? std::atoi(argv[4]) : 0;
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::fprintf(stderr, "Can not load the image file %s.\n", image_path.c_str());
        return -1;
    }
    cv::Mat gray;
#if defined(CV_VERSION_MAJOR) && CV_VERSION_MAJOR >= 3
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
#else
    cv::cvtColor(image, gray, CV_BGR2GRAY);   // Compatibility for OpenCV 2 macros
#endif

    FaceDetector detector;
    if (!detector.loadModel(join_path(model_dir, "face_detector_cascade.bin"),
                            join_path(model_dir, "face_landmark_mean_shape.bin"),
                            join_path(model_dir, "face_landmark_ferns.bin"),
                            join_path(model_dir, "face_landmark_weights.bin"))) {
        std::fprintf(stderr, "Can not load models from %s.\n", model_dir.c_str());
        return -1;
    }

    DetectOptions opt;
    opt.scale_step_ratio = 1.2f;
    opt.min_neighbors = 2;
    opt.min_size = 48;
    opt.max_size = 0;
    opt.with_landmarks = true;

    // ---------------- Single-Threaded Example ----------------
    std::vector<FaceDetection> faces = detector.detect_faces(
        gray.ptr<uint8_t>(0), gray.cols, gray.rows, static_cast<int>(gray.step), opt);

    std::printf("[single] %zu faces detected.\n", faces.size());
    for (size_t i = 0; i < faces.size(); ++i) {
        const FaceDetection& f = faces[i];
        std::printf("[single] face %zu: rect=[%d, %d, %d, %d], neighbors=%d, score=%d\n",
                    i, f.x, f.y, f.width, f.height, f.neighbors, f.score);
    }
    cv::Mat result = image.clone();
    draw_faces(result, faces);
    if (!output_path.empty()) {
        cv::imwrite(output_path, result);
        std::printf("[single] saved: %s\n", output_path.c_str());
    }

    // ---------------- Multi-Threaded Example ----------------
    // detect_faces is const and the model is read-only during inference;
    // a single detector instance can be shared across multiple threads.
    // Note: Here we share the read-only 'gray' Mat; each thread writes to its own
    // times/counts arrays to avoid concurrent modifications.
    if (threads > 1) {
        const int rounds_per_thread = 20;
        std::vector<std::thread> workers;
        std::vector<std::vector<double>> times(threads);
        std::vector<int> counts(threads, 0);

        const uint8_t* gray_ptr = gray.ptr<uint8_t>(0);
        const int cols = gray.cols;
        const int rows = gray.rows;
        const int step = static_cast<int>(gray.step);

        auto t_begin = std::chrono::steady_clock::now();
        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t] {
                times[t].reserve(rounds_per_thread);
                for (int r = 0; r < rounds_per_thread; ++r) {
                    auto t0 = std::chrono::steady_clock::now();
                    std::vector<FaceDetection> fs = detector.detect_faces(gray_ptr, cols, rows, step, opt);
                    auto t1 = std::chrono::steady_clock::now();
                    times[t].push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    counts[t] = static_cast<int>(fs.size());
                }
            });
        }
        for (auto& th : workers) th.join();
        auto t_end = std::chrono::steady_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
        std::printf("[mt] threads=%d rounds/thread=%d wall=%.2f ms throughput=%.1f img/s\n",
                    threads, rounds_per_thread, total_ms,
                    1000.0 * threads * rounds_per_thread / total_ms);
        for (int t = 0; t < threads; ++t) {
            double sum = 0.0;
            for (double v : times[t]) sum += v;
            std::printf("[mt] thread %d: avg=%.3f ms faces=%d\n", t, sum / times[t].size(), counts[t]);
        }
    }

    cv::imshow("Results_frontal", result);
    cv::waitKey();
    return 0;
}
