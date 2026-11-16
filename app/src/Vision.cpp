// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "Vision.h"

#include "Log.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <vector>

namespace spc {
namespace {

constexpr double kDeg = std::numbers::pi / 180.0;

}  // namespace

cv::Mat yellowMask(const cv::Mat& bgr)
{
    cv::Mat hsv;
    cv::Mat m;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(18, 90, 120), cv::Scalar(40, 255, 255), m);
    return m;
}

cv::Mat tileMask(const cv::Mat& bgr)
{
    const cv::Mat y = yellowMask(bgr);
    cv::Mat hsv;
    cv::Mat dark;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 110), dark);
    cv::Mat nearY;
    cv::dilate(y, nearY, cv::Mat::ones(9, 9, CV_8U));
    cv::Mat ink = dark & nearY;
    cv::dilate(ink, ink, cv::Mat::ones(5, 5, CV_8U));
    cv::Mat t = y | ink;
    cv::morphologyEx(t, t, cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8U));
    return t;
}

cv::Mat extractPatch(const cv::Mat& img, const Geometry& geo, const Pose& pose, double imageScale)
{
    cv::Mat mx;
    cv::Mat my;
    geo.patchMaps(pose, kPatchMmPerPx, kPatchPx, mx, my, imageScale);
    cv::Mat out;
    cv::remap(img, out, mx, my, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

std::optional<Pose> findLoneTile(const cv::Mat& img, const Geometry& geo, const Pose& expect, double searchMm,
                                 double imageScale)
{
    // Yellow only (closed over the ink): pencil lines next to a tile would join a tile+ink mask.
    // Looser than yellowMask in V and S: a tile in the arm's shadow drops to V ~90, while the bare
    // table stays below S ~60.
    cv::Mat hsv;
    cv::Mat t;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(18, 80, 70), cv::Scalar(40, 255, 255), t);
    cv::morphologyEx(t, t, cv::MORPH_CLOSE, cv::Mat::ones(7, 7, CV_8U));
    // and opened: thin yellowish spurs along pencil lines would stretch the fitted rectangle.
    cv::morphologyEx(t, t, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    cv::Mat lab;
    cv::Mat stats;
    cv::Mat cent;
    const int n = cv::connectedComponentsWithStats(t, lab, stats, cent);
    const cv::Point2d e = geo.robotToImage(expect.xy) * imageScale;
    // Pixels per mm here, from a 10 mm step.
    const double pxPerMm = cv::norm(geo.robotToImage(expect.xy + cv::Point2d(10.0, 0.0)) * imageScale - e) / 10.0;
    const cv::Point2d dx = geo.robotToImage(expect.xy + cv::Point2d(10.0, 0.0)) * imageScale - e;
    const cv::Point2d dy = geo.robotToImage(expect.xy + cv::Point2d(0.0, 10.0)) * imageScale - e;
    const double px2PerMm2 = std::abs(dx.x * dy.y - dx.y * dy.x) / 100.0;
    const double tileArea = geo.tileLong() * geo.tileShort() * px2PerMm2;
    int best = -1;
    double bestD = searchMm * pxPerMm;
    for (int i = 1; i < n; ++i) {
        const double area = stats.at<int>(i, cv::CC_STAT_AREA);
        const double d = std::hypot(cent.at<double>(i, 0) - e.x, cent.at<double>(i, 1) - e.y);
        if (d < 2.0 * searchMm * pxPerMm) {
            log("findLoneTile: blob ", i, " area ", area / tileArea, " tile areas, ", d / pxPerMm, " mm away");
        }
        if (area < 0.7 * tileArea || area > 1.4 * tileArea) {
            continue;
        }
        if (d < bestD) {
            best = i;
            bestD = d;
        }
    }
    if (best < 0) {
        return std::nullopt;
    }
    std::vector<std::vector<cv::Point>> contours;
    const cv::Mat blob = (lab == best);
    cv::findContours(blob, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) {
        return std::nullopt;
    }
    // Fit in the robot frame (mm): the camera looks down at an angle, so image pixels per mm
    // depend on direction.
    std::vector<cv::Point2f> mm;
    mm.reserve(contours.front().size());
    for (const auto& q : contours.front()) {
        const cv::Point2d r = geo.imageToRobot(cv::Point2d(q) / imageScale);
        mm.emplace_back(static_cast<float>(r.x), static_cast<float>(r.y));
    }
    const cv::RotatedRect rr = cv::minAreaRect(mm);
    if (std::getenv("SPC_DEBUG") != nullptr) {
        for (size_t i = 0; i < mm.size(); i += 10) {
            log("contour ", contours.front()[i].x, " ", contours.front()[i].y, " -> ", mm[i].x, " ", mm[i].y);
        }
    }
    const double longMm = std::max(rr.size.width, rr.size.height);
    const double shortMm = std::min(rr.size.width, rr.size.height);
    log("findLoneTile: rectangle ", longMm, " x ", shortMm, " mm");
    // Loose: shadows and glare bulge the mask by a few mm; the area check above already
    // rejects two tiles touching.
    if (longMm < geo.tileLong() - 6.0 || longMm > geo.tileLong() + 8.0 || shortMm < geo.tileShort() - 6.0 ||
        shortMm > geo.tileShort() + 10.0) {
        return std::nullopt;
    }
    double a = rr.angle * kDeg;
    if (rr.size.width < rr.size.height) {
        a += std::numbers::pi / 2.0;
    }
    const cv::Point2d p0(rr.center.x, rr.center.y);
    double up = std::atan2(std::sin(a), std::cos(a)) / kDeg;
    if (std::abs(wrap180(up - expect.up)) > 90.0) {
        up = wrap180(up + 180.0);
    }
    return Pose{p0, up};
}

double changeAt(const cv::Mat& before, const cv::Mat& after, const Geometry& geo, cv::Point2d p, double imageScale)
{
    const cv::Point2d c = geo.robotToImage(p) * imageScale;
    const int r = static_cast<int>(std::lround(10.0 * imageScale));  // ~5 mm each way at 2 px/mm
    const cv::Rect box(static_cast<int>(std::lround(c.x)) - r, static_cast<int>(std::lround(c.y)) - r, 2 * r + 1,
                       2 * r + 1);
    const cv::Rect roi = box & cv::Rect(0, 0, before.cols, before.rows);
    if (roi.area() == 0 || before.size() != after.size()) {
        return std::numeric_limits<double>::infinity();
    }
    cv::Mat a;
    cv::Mat b;
    cv::cvtColor(before(roi), a, cv::COLOR_BGR2GRAY);
    cv::cvtColor(after(roi), b, cv::COLOR_BGR2GRAY);
    cv::Mat d;
    cv::absdiff(a, b, d);
    return cv::mean(d)[0];
}

}  // namespace spc
