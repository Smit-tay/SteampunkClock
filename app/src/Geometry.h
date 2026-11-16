// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Geometry: robot frame <-> table grid <-> camera image, table height, cup offset, wrist planning.
// The robot frame is the uArm frame (x forward, y left, z up). The camera image is mirror-imaged
// relative to it, so angles are always kept in the robot frame and the image is used only for vision.
#pragma once

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core.hpp>

#include <vector>

namespace spc {

// A tile lying flat: centre and the direction its digit top points (robot frame, degrees).
struct Pose {
    cv::Point2d xy;
    double up = 0.0;
};

// Wrap an angle to [-180, 180).
double wrap180(double deg);

class Geometry {
public:
    // cal: the "calibration" object of the config; tile: the "tile" object.
    Geometry(const nlohmann::json& cal, const nlohmann::json& tile);

    [[nodiscard]] cv::Point2d robotToGrid(cv::Point2d r) const;
    [[nodiscard]] cv::Point2d gridToRobot(cv::Point2d g) const;
    // Image pixels of the full-resolution (image_size) photo; divide by 2 for 800x600.
    [[nodiscard]] cv::Point2d robotToImage(cv::Point2d r) const;
    [[nodiscard]] cv::Point2d imageToRobot(cv::Point2d px) const;

    // Robot-frame direction of a grid-frame direction, and back (degrees).
    [[nodiscard]] double gridDirToRobot(double gridDeg) const;
    [[nodiscard]] double robotDirToGrid(double robotDeg) const;
    [[nodiscard]] static double baseDeg(cv::Point2d r);

    // Table top under the cup (fit 2026-11-15), and the cup z at which a tile lying
    // `level` tiles above the table seals (plane-corrected from measured seals).
    [[nodiscard]] double surface(cv::Point2d r) const;
    [[nodiscard]] double sealZ(cv::Point2d r, int level = 0) const;
    // Cup z at which a held tile's bottom is ~1 mm above a surface `level` tiles high.
    [[nodiscard]] double releaseZ(cv::Point2d r, int level) const;

    // Arm command for the cup to be at `cup` with the wrist at `wristDeg`: the cup sits
    // off the wrist axis (cup_offset, fixed in the arm frame), so turning the wrist swings it.
    [[nodiscard]] cv::Point2d cupToCommand(cv::Point2d cup, double wristDeg) const;

    // Wrist angles to carry a tile from src to dst so it arrives with dst.up.
    struct WristPlan {
        double pick = 90.0;
        double place = 90.0;
        int steps = 1;  // the turn needs more than the servo's 0..180: re-grip and turn again, each step pick -> place
    };
    [[nodiscard]] static WristPlan planWrist(const Pose& src, const Pose& dst, double wristScale = 1.0);

    // Map from a size x size patch (mmPerPx, digit top up, centred on the pose) into image
    // pixels; use with cv::remap. Patches are independent of the camera's perspective.
    void patchMaps(const Pose& pose, double mmPerPx, int size, cv::Mat& mapX, cv::Mat& mapY,
                   double imageScale = 1.0) const;

    [[nodiscard]] double tileLong() const { return tileLong_; }
    [[nodiscard]] double tileShort() const { return tileShort_; }
    [[nodiscard]] double tileHeight() const { return tileHeight_; }

private:
    cv::Matx23d robotToGrid_;
    cv::Matx33d gridToRobot_;   // inverse of robotToGrid_ (homogeneous)
    cv::Matx33d robotToImage_;  // projective
    cv::Matx33d imageToRobot_;
    double surfZ0_ = 0.0;
    double surfPerMm_ = 0.0;
    double sealAbove_ = 0.0;
    double plane_[3] = {0.0, 0.0, 0.0};  // seal residual = p0 + p1 x + p2 y
    cv::Point2d cupOffset_;
    double hang_ = 0.0;
    double tileLong_ = 40.0;
    double tileShort_ = 30.0;
    double tileHeight_ = 3.5;
};

}  // namespace spc
