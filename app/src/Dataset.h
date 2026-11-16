// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Dataset: labelled tile patches on disk and their augmentation for training.
// Layout: <dir>/<label>/<name>.png with label "0".."9" (an upright tile of that digit) or "junk"
// (a patch not centred on one whole tile). Patches follow the Vision.h convention.
#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <vector>

namespace spc {

struct Sample {
    cv::Mat patch;  // kPatchPx x kPatchPx BGR
    int label = 0;  // 0..9, or DigitNet::kJunk
    std::string source;
};

[[nodiscard]] std::vector<Sample> loadSamples(const std::vector<std::filesystem::path>& dirs);
void saveSample(const std::filesystem::path& dir, const std::string& label, const std::string& name,
                const cv::Mat& patch);

// The review panels of the 2026-11-15 hand-guided sort (sorter/ask*/<name>.png with answer lines
// "i digit k cx cy [t]" in <name>.txt): one upright patch per answered tile plus a junk patch
// beside it. Returns the number of tiles written.
int importPanels(const std::vector<std::filesystem::path>& askDirs, const std::filesystem::path& outDir);

// Feature rows and class ids for training: each upright sample `perSample` jittered times, its
// three quarter turns as "not upright", junk samples as junk; neighbouring tiles are pasted in
// at random so heap scenes are covered.
void buildTrainingSet(const std::vector<Sample>& samples, int perSample, unsigned seed, cv::Mat& x, cv::Mat& y);

}  // namespace spc
