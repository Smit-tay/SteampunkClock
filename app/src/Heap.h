// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Heap vision: find tiles that can be lifted from a jumbled heap. The photo is resampled to a
// top-down map in the robot frame; a 40 x 30 mm rectangle is swept over the yellow area, and a
// pose scores by how much of its outline shows a tile edge (yellow inside, table or a shadow
// outside). A tile with another tile on top loses the covered part of its outline, so the
// best-scoring poses are tiles with nothing on them.
#pragma once

#include "DigitNet.h"
#include "Geometry.h"

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

namespace spc {

struct HeapTile {
    Pose pose;              // up: along the long axis, the digit's top after reading
    double border = 0.0;    // fraction of the outline showing an edge (1: nothing on top)
    double fill = 0.0;      // fraction of the inside that is yellow
    bool lone = false;      // a separate tile, not touching any other
    DigitNet::Reading reading;
    double score = 0.0;     // pick priority, higher first
};

struct HeapRegion {
    cv::Rect2d robot{60.0, -280.0, 300.0, 560.0};  // x 60..360  // x, y, width, height in mm, robot frame
    std::vector<cv::Point2d> exclude;               // stacks, display slots: ignore yellow around these
    double excludeMm = 32.0;
};

// Candidates, best first. net may be null (no reading). img: full-resolution photo.
[[nodiscard]] std::vector<HeapTile> findHeapTiles(const cv::Mat& img, const Geometry& geo, const HeapRegion& region,
                                                  const DigitNet* net);

// The whole tile nearest `near` (within maxMm), found by the same outline search in a small
// window: works beside other tiles and on stacks, where a colour blob would merge them.
[[nodiscard]] std::optional<HeapTile> locateTile(const cv::Mat& img, const Geometry& geo, cv::Point2d near,
                                                 double maxMm, const DigitNet* net);

// Outlines and labels on a copy of img, for checking by eye.
[[nodiscard]] cv::Mat drawHeapTiles(const cv::Mat& img, const Geometry& geo, const std::vector<HeapTile>& tiles);

}  // namespace spc
