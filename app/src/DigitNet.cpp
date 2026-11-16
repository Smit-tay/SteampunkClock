// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "DigitNet.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace spc {

cv::Mat DigitNet::features(const cv::Mat& patch)
{
    cv::Mat g;
    cv::Mat sat;
    if (patch.channels() == 3) {
        cv::cvtColor(patch, g, cv::COLOR_BGR2GRAY);
        cv::Mat hsv;
        cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
        cv::extractChannel(hsv, sat, 1);
    } else {
        g = patch;
        sat = cv::Mat::zeros(patch.size(), CV_8U);
    }
    // Ink: how much darker than the local background (a closing wipes out thin strokes), so
    // lighting, shadows and the tile colour drop out.
    cv::Mat bg;
    cv::morphologyEx(g, bg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    cv::Mat ink = bg - g;  // saturating
    cv::Mat inkS;
    cv::resize(ink, inkS, cv::Size(kSide, kSide), 0, 0, cv::INTER_AREA);
    cv::Mat fi;
    inkS.convertTo(fi, CV_32F);
    double mx = 0.0;
    cv::minMaxLoc(fi, nullptr, &mx);
    fi /= std::max(mx, 10.0);
    // Where the tile is: the yellow is saturated, the table and the ink are not.
    cv::Mat satS;
    cv::resize(sat, satS, cv::Size(kSatSide, kSatSide), 0, 0, cv::INTER_AREA);
    cv::Mat fs;
    satS.convertTo(fs, CV_32F, 1.0 / 255.0);
    // Stroke directions: gradient-orientation histograms on a grid of cells (HOG without the
    // block overlap), tolerant of the few-mm shifts the placement and the fit leave.
    cv::Mat gs;
    cv::resize(g, gs, cv::Size(kHogSide, kHogSide), 0, 0, cv::INTER_AREA);
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(gs, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gs, gy, CV_32F, 0, 1, 3);
    cv::Mat mag;
    cv::Mat ang;
    cv::cartToPolar(gx, gy, mag, ang, true);
    constexpr int kCells = kHogSide / kHogCell;
    cv::Mat hog(1, kCells * kCells * kHogBins, CV_32F, cv::Scalar(0.0f));
    for (int y = 0; y < kHogSide; ++y) {
        for (int x = 0; x < kHogSide; ++x) {
            // Unsigned direction (a stroke has two edges of opposite sign).
            const float a = std::fmod(ang.at<float>(y, x), 180.0f);
            const int bin = std::min(kHogBins - 1, static_cast<int>(a / (180.0f / kHogBins)));
            const int cell = (y / kHogCell) * kCells + x / kHogCell;
            hog.at<float>(0, cell * kHogBins + bin) += mag.at<float>(y, x);
        }
    }
    for (int c = 0; c < kCells * kCells; ++c) {
        cv::Mat h = hog.colRange(c * kHogBins, (c + 1) * kHogBins);
        const double n = cv::norm(h);
        h /= std::max(n, 50.0);
    }
    cv::Mat f;
    cv::hconcat(std::vector<cv::Mat>{fi.reshape(1, 1), fs.reshape(1, 1), hog}, f);
    return f;
}

void DigitNet::train(const cv::Mat& x, const cv::Mat& y, int pcaDims, int hidden, int iterations)
{
    if (x.rows != y.rows || x.rows == 0) {
        throw std::invalid_argument("DigitNet::train: no data");
    }
    pca_ = cv::PCA(x, cv::noArray(), cv::PCA::DATA_AS_ROW, pcaDims);
    const cv::Mat z = pca_.project(x);
    cv::Mat target(x.rows, kClasses, CV_32F, cv::Scalar(-1.0f));
    for (int i = 0; i < y.rows; ++i) {
        target.at<float>(i, y.at<int>(i, 0)) = 1.0f;
    }
    // An ensemble: each member starts from other random weights and sees a random 80 % of the
    // rows; their scores are averaged, which steadies the readings of the rarer digits.
    mlps_.clear();
    cv::RNG pick(7);
    for (int m = 0; m < kMembers; ++m) {
        cv::theRNG().state = 1000 + static_cast<uint64_t>(m);
        cv::Mat zs;
        cv::Mat ts;
        for (int i = 0; i < z.rows; ++i) {
            if (pick.uniform(0.0, 1.0) < 0.8) {
                zs.push_back(z.row(i));
                ts.push_back(target.row(i));
            }
        }
        auto mlp = cv::ml::ANN_MLP::create();
        const cv::Mat layers = (cv::Mat_<int>(1, 3) << z.cols, hidden, kClasses);
        mlp->setLayerSizes(layers);
        mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
        mlp->setTrainMethod(cv::ml::ANN_MLP::RPROP);
        mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, iterations, 1e-5));
        if (!mlp->train(cv::ml::TrainData::create(zs, cv::ml::ROW_SAMPLE, ts))) {
            throw std::runtime_error("DigitNet::train: ANN_MLP did not train");
        }
        mlps_.push_back(mlp);
    }
}

void DigitNet::save(const std::filesystem::path& file) const
{
    cv::FileStorage fs(file.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("cannot write " + file.string());
    }
    fs << "side" << kSide;
    fs << "features" << kFeatureVersion;
    pca_.write(fs);
    fs << "members" << static_cast<int>(mlps_.size());
    for (size_t m = 0; m < mlps_.size(); ++m) {
        fs << "mlp" + std::to_string(m) << "{";
        mlps_[m]->write(fs);
        fs << "}";
    }
}

void DigitNet::load(const std::filesystem::path& file)
{
    cv::FileStorage fs(file.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("cannot read model " + file.string());
    }
    if (static_cast<int>(fs["side"]) != kSide || static_cast<int>(fs["features"]) != kFeatureVersion) {
        throw std::runtime_error("model " + file.string() + " has another input size");
    }
    pca_.read(fs.root());
    mlps_.clear();
    const int members = static_cast<int>(fs["members"]);
    for (int m = 0; m < members; ++m) {
        auto mlp = cv::ml::ANN_MLP::create();
        mlp->read(fs["mlp" + std::to_string(m)]);
        mlps_.push_back(mlp);
    }
}

cv::Mat DigitNet::scores(const cv::Mat& x) const
{
    if (!ready()) {
        throw std::runtime_error("DigitNet: no model loaded");
    }
    const cv::Mat z = pca_.project(x);
    cv::Mat sum(x.rows, kClasses, CV_32F, cv::Scalar(0.0f));
    for (const auto& mlp : mlps_) {
        cv::Mat out;
        mlp->predict(z, out);
        sum += out;
    }
    return sum / static_cast<double>(mlps_.size());
}

DigitNet::Reading DigitNet::read(const std::array<cv::Mat, 4>& patches) const
{
    cv::Mat x;
    for (const auto& p : patches) {
        x.push_back(features(p));
    }
    const cv::Mat s = scores(x);
    // Each digit's best score over the four orientations.
    std::array<float, 10> perDigit{};
    std::array<int, 10> quarter{};
    perDigit.fill(-1e9f);
    for (int k = 0; k < 4; ++k) {
        for (int d = 0; d < 10; ++d) {
            if (s.at<float>(k, d) > perDigit[static_cast<size_t>(d)]) {
                perDigit[static_cast<size_t>(d)] = s.at<float>(k, d);
                quarter[static_cast<size_t>(d)] = k;
            }
        }
    }
    Reading r;
    r.digit = static_cast<int>(std::max_element(perDigit.begin(), perDigit.end()) - perDigit.begin());
    r.quarter = quarter[static_cast<size_t>(r.digit)];
    const float best = perDigit[static_cast<size_t>(r.digit)];
    float second = -1e9f;
    for (int d = 0; d < 10; ++d) {
        if (d != r.digit) {
            second = std::max(second, perDigit[static_cast<size_t>(d)]);
        }
    }
    r.score = best;
    r.margin = best - second;
    r.junk = s.at<float>(r.quarter, kJunk);
    // Not upright in its best orientation, or not a tile: unreadable.
    if (s.at<float>(r.quarter, kRotated) > best || r.junk > best) {
        r.digit = -1;
    }
    return r;
}

}  // namespace spc
