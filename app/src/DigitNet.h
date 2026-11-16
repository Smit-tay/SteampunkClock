// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// DigitNet: recognises a tile from its patch (Vision.h convention). A small neural network
// (OpenCV ml::ANN_MLP) on PCA-reduced, contrast-normalised grey pixels, trained offline on
// labelled patches (spclock train). Classes: digits 0-9 upright, "tile not upright", "not a
// whole tile". A tile is read at its four quarter turns; the best upright reading wins.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#include <array>
#include <filesystem>
#include <vector>

namespace spc {

class DigitNet {
public:
    static constexpr int kSide = 32;     // ink map, kSide x kSide
    static constexpr int kSatSide = 12;  // saturation (tile placement) map
    static constexpr int kHogSide = 48;  // gradient histograms: kHogSide px grey, kHogCell px cells
    static constexpr int kHogCell = 8;
    static constexpr int kHogBins = 9;
    static constexpr int kFeatureVersion = 4;
    static constexpr int kMembers = 5;  // models of another version are refused
    static constexpr int kRotated = 10;
    static constexpr int kJunk = 11;
    static constexpr int kClasses = 12;

    // The handwritten 0 (slashed) and 8 look the same turned half round.
    [[nodiscard]] static constexpr bool halfTurnSymmetric(int digit) { return digit == 0 || digit == 8; }

    // One row (CV_32F) for a patch: ink map, saturation map, gradient histograms.
    [[nodiscard]] static cv::Mat features(const cv::Mat& patch);

    // X: rows of features(); y: N x 1 CV_32S class ids.
    void train(const cv::Mat& x, const cv::Mat& y, int pcaDims, int hidden, int iterations);
    void save(const std::filesystem::path& file) const;
    void load(const std::filesystem::path& file);
    [[nodiscard]] bool ready() const { return !mlps_.empty() && mlps_.front()->isTrained(); }

    // Class scores for feature rows (N x kClasses); higher is better.
    [[nodiscard]] cv::Mat scores(const cv::Mat& x) const;

    struct Reading {
        int digit = -1;       // -1: not readable
        int quarter = 0;      // which patch (k quarter turns from the guess) was upright
        float score = 0.0f;   // its digit score
        float margin = 0.0f;  // over the next best digit in any orientation
        float junk = 0.0f;    // "not a whole tile" score of the upright patch
    };
    // patches[k]: the tile with up turned k quarter turns counter-clockwise from the guess.
    [[nodiscard]] Reading read(const std::array<cv::Mat, 4>& patches) const;

private:
    cv::PCA pca_;
    std::vector<cv::Ptr<cv::ml::ANN_MLP>> mlps_;
};

}  // namespace spc
