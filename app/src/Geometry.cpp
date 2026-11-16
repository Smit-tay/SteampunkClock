// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "Geometry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace spc {
namespace {

constexpr double kDeg = std::numbers::pi / 180.0;

cv::Point2d apply(const cv::Matx33d& m, cv::Point2d p)
{
    const cv::Vec3d q = m * cv::Vec3d(p.x, p.y, 1.0);
    return {q[0] / q[2], q[1] / q[2]};
}

cv::Matx33d homogeneous(const cv::Matx23d& a)
{
    return {a(0, 0), a(0, 1), a(0, 2), a(1, 0), a(1, 1), a(1, 2), 0.0, 0.0, 1.0};
}

}  // namespace

double wrap180(double deg)
{
    double a = std::fmod(deg + 180.0, 360.0);
    if (a < 0.0) {
        a += 360.0;
    }
    return a - 180.0;
}

Geometry::Geometry(const nlohmann::json& cal, const nlohmann::json& tile)
{
    const auto& rg = cal.at("robot_to_grid");
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            robotToGrid_(r, c) = rg.at(r).at(c).get<double>();
        }
    }
    cv::Matx33d imageToGrid;
    const auto& ig = cal.at("image_to_grid");
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            imageToGrid(r, c) = ig.at(r).at(c).get<double>();
        }
    }
    const cv::Matx33d r2g = homogeneous(robotToGrid_);
    gridToRobot_ = r2g.inv();
    robotToImage_ = imageToGrid.inv() * r2g;
    imageToRobot_ = robotToImage_.inv();

    surfZ0_ = cal.at("surface_z0").get<double>();
    surfPerMm_ = cal.at("surface_per_mm").get<double>();
    sealAbove_ = cal.at("seal_above_surface").get<double>();
    const auto& co = cal.at("cup_offset");
    cupOffset_ = {co.at(0).get<double>(), co.at(1).get<double>()};
    hang_ = cal.at("hang").get<double>();
    tileLong_ = tile.at("long").get<double>();
    tileShort_ = tile.at("short").get<double>();
    tileHeight_ = tile.at("height").get<double>();

    // Plane through the residuals of the measured seal heights (least squares; mean below 4 points).
    const auto& seals = cal.at("seals");
    std::vector<cv::Vec3d> rows;
    for (const auto& s : seals) {
        rows.emplace_back(s.at(0).get<double>(), s.at(1).get<double>(), s.at(2).get<double>());
    }
    if (!rows.empty()) {
        cv::Mat a(static_cast<int>(rows.size()), 3, CV_64F);
        cv::Mat b(static_cast<int>(rows.size()), 1, CV_64F);
        for (int i = 0; i < a.rows; ++i) {
            const auto& s = rows[static_cast<size_t>(i)];
            a.at<double>(i, 0) = 1.0;
            a.at<double>(i, 1) = s[0];
            a.at<double>(i, 2) = s[1];
            b.at<double>(i, 0) = s[2] - (surface({s[0], s[1]}) + sealAbove_);
        }
        if (rows.size() < 4) {
            plane_[0] = cv::mean(b)[0];
        } else {
            cv::Mat p;
            cv::solve(a, b, p, cv::DECOMP_SVD);
            for (int i = 0; i < 3; ++i) {
                plane_[i] = p.at<double>(i, 0);
            }
        }
    }
}

cv::Point2d Geometry::robotToGrid(cv::Point2d r) const
{
    const cv::Vec2d g = robotToGrid_ * cv::Vec3d(r.x, r.y, 1.0);
    return {g[0], g[1]};
}

cv::Point2d Geometry::gridToRobot(cv::Point2d g) const { return apply(gridToRobot_, g); }
cv::Point2d Geometry::robotToImage(cv::Point2d r) const { return apply(robotToImage_, r); }
cv::Point2d Geometry::imageToRobot(cv::Point2d px) const { return apply(imageToRobot_, px); }

double Geometry::gridDirToRobot(double gridDeg) const
{
    // Linear part of grid -> robot.
    const double gx = std::cos(gridDeg * kDeg);
    const double gy = std::sin(gridDeg * kDeg);
    const double rx = gridToRobot_(0, 0) * gx + gridToRobot_(0, 1) * gy;
    const double ry = gridToRobot_(1, 0) * gx + gridToRobot_(1, 1) * gy;
    return std::atan2(ry, rx) / kDeg;
}

double Geometry::robotDirToGrid(double robotDeg) const
{
    const double rx = std::cos(robotDeg * kDeg);
    const double ry = std::sin(robotDeg * kDeg);
    const double gx = robotToGrid_(0, 0) * rx + robotToGrid_(0, 1) * ry;
    const double gy = robotToGrid_(1, 0) * rx + robotToGrid_(1, 1) * ry;
    return std::atan2(gy, gx) / kDeg;
}

double Geometry::baseDeg(cv::Point2d r) { return std::atan2(r.y, r.x) / kDeg; }

double Geometry::surface(cv::Point2d r) const { return surfZ0_ + surfPerMm_ * std::hypot(r.x, r.y); }

double Geometry::sealZ(cv::Point2d r, int level) const
{
    return surface(r) + sealAbove_ + plane_[0] + plane_[1] * r.x + plane_[2] * r.y + tileHeight_ * level;
}

double Geometry::releaseZ(cv::Point2d r, int level) const { return sealZ(r, level) - hang_; }

cv::Point2d Geometry::cupToCommand(cv::Point2d cup, double wristDeg) const
{
    const double a = (wristDeg - 90.0) * kDeg;
    const double c = std::cos(a);
    const double s = std::sin(a);
    // cmd = cup - (R(w - 90) - I) * offset
    const double dx = (c - 1.0) * cupOffset_.x - s * cupOffset_.y;
    const double dy = s * cupOffset_.x + (c - 1.0) * cupOffset_.y;
    return {cup.x - dx, cup.y - dy};
}

Geometry::WristPlan Geometry::planWrist(const Pose& src, const Pose& dst, double wristScale)
{
    // Carried at a fixed wrist angle, a tile turns with the base; turning the wrist by dw
    // turns it by dw more (both counter-clockwise in the robot frame). wristScale: the wrist
    // turns about 9.5 % less than commanded (measured 2026-11-15), so command dw * scale.
    const double dw = wrap180(dst.up - src.up - (baseDeg(dst.xy) - baseDeg(src.xy))) * wristScale;
    // A half turn needs ~199 degrees of servo: more than its range. Then the turn is split into
    // equal steps, with the tile set down and picked up again between them.
    WristPlan p;
    p.steps = std::max(1, static_cast<int>(std::ceil(std::abs(dw) / 180.0 - 1e-9)));
    const double step = dw / p.steps;
    p.pick = std::clamp(90.0 - step / 2.0, 0.0, 180.0);
    p.place = std::clamp(p.pick + step, 0.0, 180.0);
    return p;
}

void Geometry::patchMaps(const Pose& pose, double mmPerPx, int size, cv::Mat& mapX, cv::Mat& mapY,
                         double imageScale) const
{
    if (size <= 0 || mmPerPx <= 0.0) {
        throw std::invalid_argument("patchMaps: bad size");
    }
    // Seen from above, the robot frame is right-handed (x, y counter-clockwise), so the
    // digit's right-hand side is its up direction turned clockwise by 90 degrees.
    const cv::Point2d up{std::cos(pose.up * kDeg), std::sin(pose.up * kDeg)};
    const cv::Point2d right{up.y, -up.x};
    mapX.create(size, size, CV_32F);
    mapY.create(size, size, CV_32F);
    const double half = (size - 1) / 2.0;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            const cv::Point2d p = pose.xy + right * ((i - half) * mmPerPx) + up * ((half - j) * mmPerPx);
            const cv::Point2d px = robotToImage(p) * imageScale;
            mapX.at<float>(j, i) = static_cast<float>(px.x);
            mapY.at<float>(j, i) = static_cast<float>(px.y);
        }
    }
}

}  // namespace spc
