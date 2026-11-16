// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "Dataset.h"

#include "DigitNet.h"
#include "Log.h"
#include "Vision.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>

namespace spc {
namespace fs = std::filesystem;
namespace {

constexpr int kPanel = 160;          // review panel side (full-resolution image pixels)
constexpr double kPanelPxPerMm = 2.1;  // full-resolution pixels per mm on the table, about

cv::Mat jitter(const cv::Mat& src, cv::RNG& rng)
{
    const cv::Point2f c(static_cast<float>(src.cols) / 2.0f, static_cast<float>(src.rows) / 2.0f);
    cv::Mat m = cv::getRotationMatrix2D(c, rng.uniform(-8.0, 8.0), rng.uniform(0.94, 1.06));
    m.at<double>(0, 2) += rng.uniform(-4.0, 4.0);
    m.at<double>(1, 2) += rng.uniform(-4.0, 4.0);
    cv::Mat out;
    cv::warpAffine(src, out, m, src.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    out.convertTo(out, -1, rng.uniform(0.8, 1.2), rng.uniform(-25.0, 25.0));
    if (rng.uniform(0, 2) == 1) {
        const double s = rng.uniform(0.3, 1.2);
        cv::GaussianBlur(out, out, cv::Size(0, 0), s);
    }
    cv::Mat noise(out.size(), out.type());
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(4));
    out += noise;
    return out;
}

// Paste the yellow part of `other` into `base` beside the base tile (heap neighbours).
void pasteNeighbour(cv::Mat& base, const cv::Mat& other, cv::RNG& rng)
{
    cv::Mat o;
    cv::rotate(other, o, rng.uniform(0, 3));
    const double a = rng.uniform(0.0, 2.0 * std::numbers::pi);
    const double d = rng.uniform(32.0, 42.0) / kPatchMmPerPx;  // centre distance in patch px
    cv::Mat m = (cv::Mat_<double>(2, 3) << 1, 0, std::cos(a) * d, 0, 1, std::sin(a) * d);
    cv::Mat shifted;
    cv::warpAffine(o, shifted, m, base.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat mask = yellowMask(shifted);
    // Keep the base tile itself (30 x 40 mm, upright) uncovered: the neighbour lies beside it.
    const int w = static_cast<int>(30.0 / kPatchMmPerPx);
    const int h = static_cast<int>(40.0 / kPatchMmPerPx);
    mask(cv::Rect((base.cols - w) / 2, (base.rows - h) / 2, w, h)).setTo(0);
    shifted.copyTo(base, mask);
}

}  // namespace

std::vector<Sample> loadSamples(const std::vector<fs::path>& dirs)
{
    std::vector<Sample> out;
    for (const auto& dir : dirs) {
        if (!fs::is_directory(dir)) {
            continue;
        }
        for (const auto& sub : fs::directory_iterator(dir)) {
            if (!sub.is_directory()) {
                continue;
            }
            const std::string name = sub.path().filename().string();
            int label = -1;
            if (name == "junk") {
                label = DigitNet::kJunk;
            } else if (name.size() == 1 && name[0] >= '0' && name[0] <= '9') {
                label = name[0] - '0';
            } else {
                continue;
            }
            for (const auto& f : fs::directory_iterator(sub.path())) {
                if (f.path().extension() != ".png") {
                    continue;
                }
                cv::Mat p = cv::imread(f.path().string(), cv::IMREAD_COLOR);
                if (p.empty()) {
                    continue;
                }
                if (p.rows != kPatchPx || p.cols != kPatchPx) {
                    cv::resize(p, p, cv::Size(kPatchPx, kPatchPx), 0, 0, cv::INTER_AREA);
                }
                out.push_back({p, label, f.path().string()});
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const Sample& a, const Sample& b) { return a.source < b.source; });
    return out;
}

void saveSample(const fs::path& dir, const std::string& label, const std::string& name, const cv::Mat& patch)
{
    fs::create_directories(dir / label);
    cv::imwrite((dir / label / (name + ".png")).string(), patch);
}

int importPanels(const std::vector<fs::path>& askDirs, const fs::path& outDir)
{
    int written = 0;
    cv::RNG rng(12345);
    for (const auto& dir : askDirs) {
        if (!fs::is_directory(dir)) {
            continue;
        }
        for (const auto& f : fs::directory_iterator(dir)) {
            if (f.path().extension() != ".txt") {
                continue;
            }
            fs::path png = f.path();
            png.replace_extension(".png");
            const cv::Mat sheet = cv::imread(png.string(), cv::IMREAD_COLOR);
            if (sheet.empty()) {
                continue;
            }
            std::ifstream in(f.path());
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ls(line);
                int i = 0;
                std::string digit;
                int k = 0;
                double cx = 0;
                double cy = 0;
                double t = 0;
                if (!(ls >> i >> digit >> k >> cx >> cy) || digit.size() != 1 || digit[0] < '0' || digit[0] > '9') {
                    continue;
                }
                ls >> t;  // optional tilt, degrees
                const cv::Rect cell(kPanel * k, kPanel * i, kPanel, kPanel);
                if ((cell & cv::Rect(0, 0, sheet.cols, sheet.rows)) != cell) {
                    continue;
                }
                cv::Mat panel = sheet(cell).clone();
                // Remove the overlay (magenta cross, red labels): copy a median-blurred background in.
                cv::Mat hsv;
                cv::cvtColor(panel, hsv, cv::COLOR_BGR2HSV);
                cv::Mat magenta;
                cv::Mat red1;
                cv::Mat red2;
                cv::inRange(hsv, cv::Scalar(140, 120, 120), cv::Scalar(170, 255, 255), magenta);
                cv::inRange(hsv, cv::Scalar(0, 150, 100), cv::Scalar(8, 255, 255), red1);
                cv::inRange(hsv, cv::Scalar(172, 150, 100), cv::Scalar(180, 255, 255), red2);
                cv::Mat overlay = magenta | red1 | red2;
                cv::dilate(overlay, overlay, cv::Mat::ones(3, 3, CV_8U));
                cv::Mat bg;
                cv::medianBlur(panel, bg, 9);
                bg.copyTo(panel, overlay);
                // The reader's tilt t was applied about the panel centre before reading cx, cy.
                const cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(80.0f, 80.0f), t, 1.0);
                cv::warpAffine(panel, panel, rot, panel.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
                cv::Mat pad;
                const int b = 80;
                cv::copyMakeBorder(panel, pad, b, b, b, b, cv::BORDER_REPLICATE);
                const int side = static_cast<int>(std::lround(kPatchPx * kPatchMmPerPx * kPanelPxPerMm));
                auto crop = [&](double ox, double oy) {
                    const cv::Rect r(static_cast<int>(std::lround(cx + ox)) + b - side / 2,
                                     static_cast<int>(std::lround(cy + oy)) + b - side / 2, side, side);
                    cv::Mat p;
                    cv::resize(pad(r & cv::Rect(0, 0, pad.cols, pad.rows)), p, cv::Size(kPatchPx, kPatchPx), 0, 0,
                               cv::INTER_AREA);
                    return p;
                };
                const std::string name =
                    dir.filename().string() + "_" + f.path().stem().string() + "_" + std::to_string(i);
                saveSample(outDir, digit, name, crop(0, 0));
                const double a = rng.uniform(0.0, 2.0 * std::numbers::pi);
                const double d = rng.uniform(16.0, 22.0) * kPanelPxPerMm;
                saveSample(outDir, "junk", name, crop(std::cos(a) * d, std::sin(a) * d));
                ++written;
            }
        }
    }
    return written;
}

void buildTrainingSet(const std::vector<Sample>& samples, int perSample, unsigned seed, cv::Mat& x, cv::Mat& y)
{
    cv::RNG rng(seed);
    std::vector<const Sample*> tiles;
    for (const auto& s : samples) {
        if (s.label < 10) {
            tiles.push_back(&s);
        }
    }
    x.release();
    y.release();
    auto add = [&](const cv::Mat& patch, int label) {
        x.push_back(DigitNet::features(patch));
        y.push_back(label);
    };
    // Balance the digits: a digit with fewer photos gets more variations of each (up to 4x).
    std::array<int, 12> perLabel{};
    for (const auto& s : samples) {
        ++perLabel[static_cast<size_t>(s.label)];
    }
    int most = 1;
    for (int d = 0; d < 10; ++d) {
        most = std::max(most, perLabel[static_cast<size_t>(d)]);
    }
    const int rotFlags[3] = {cv::ROTATE_90_CLOCKWISE, cv::ROTATE_180, cv::ROTATE_90_COUNTERCLOCKWISE};
    for (const auto& s : samples) {
        const int n = s.label < 10 ? std::min(4 * perSample, perSample * most / std::max(1, perLabel[static_cast<size_t>(s.label)]))
                                   : perSample / 3;
        for (int a = 0; a < n; ++a) {
            cv::Mat p = s.patch.clone();
            if (!tiles.empty() && rng.uniform(0, 2) == 1) {
                pasteNeighbour(p, tiles[static_cast<size_t>(rng.uniform(0, static_cast<int>(tiles.size())))]->patch,
                               rng);
            }
            if (s.label < 10 && DigitNet::halfTurnSymmetric(s.label) && a % 2 == 1) {
                cv::rotate(p, p, cv::ROTATE_180);  // the same digit either way up
            }
            add(jitter(p, rng), s.label);
            if (s.label < 10 && a % 3 == 0) {
                int flag = rotFlags[(a / 3) % 3];
                if (flag == cv::ROTATE_180 && DigitNet::halfTurnSymmetric(s.label)) {
                    flag = (a / 3) % 2 == 0 ? cv::ROTATE_90_CLOCKWISE : cv::ROTATE_90_COUNTERCLOCKWISE;
                }
                cv::Mat r;
                cv::rotate(p, r, flag);
                add(jitter(r, rng), DigitNet::kRotated);
            }
        }
    }
    log("training set: ", x.rows, " rows from ", samples.size(), " samples");
}

}  // namespace spc
