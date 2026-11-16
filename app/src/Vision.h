// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Vision basics: tile colour masks, upright tile patches, a lone tile's exact pose.
// All functions take the full-resolution photo (calibration image_size, 1600x1200) unless an
// image scale is given (0.5 for 800x600).
#pragma once

#include "Geometry.h"

#include <opencv2/core.hpp>

#include <optional>

namespace spc {

// Patch convention shared by training and recognition: 44 x 44 mm around the tile centre,
// 0.5 mm per pixel, digit top up.
inline constexpr int kPatchPx = 88;
inline constexpr double kPatchMmPerPx = 0.5;

[[nodiscard]] cv::Mat yellowMask(const cv::Mat& bgr);
// Yellow plus the ink written on it.
[[nodiscard]] cv::Mat tileMask(const cv::Mat& bgr);

[[nodiscard]] cv::Mat extractPatch(const cv::Mat& img, const Geometry& geo, const Pose& pose,
                                   double imageScale = 1.0);

// The tile lying alone nearest to `expect` (within searchMm): centre and up direction (the
// tile's long axis, the end nearest expect.up). nullopt if no tile-sized blob is there.
[[nodiscard]] std::optional<Pose> findLoneTile(const cv::Mat& img, const Geometry& geo, const Pose& expect,
                                               double searchMm, double imageScale = 1.0);

// Mean absolute grey change of a ~10 mm square around p between two photos (same scale).
[[nodiscard]] double changeAt(const cv::Mat& before, const cv::Mat& after, const Geometry& geo, cv::Point2d p,
                              double imageScale = 1.0);

}  // namespace spc
