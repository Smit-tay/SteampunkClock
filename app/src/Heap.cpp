// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "Heap.h"

#include "Vision.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string>

namespace spc {
namespace {

constexpr double kDeg = std::numbers::pi / 180.0;
constexpr double kPxPerMm = 2.0;    // top-down map resolution
constexpr double kEdgeMm = 2.0;     // in/out sample distance from the outline
constexpr double kCornerMm = 3.0;   // outline samples skip the rounded, blurred corners
constexpr int kEdgeS = 35;          // saturation step at a tile edge on the table
constexpr int kEdgeV = 18;          // brightness step at a tile edge on another tile (its shadow)
constexpr int kInk = 110;           // HSV value below this: pencil or ink, not tile or table
constexpr double kMinSide = 0.3;
constexpr double kMinFill = 0.85;
constexpr double kMinBorder = 0.75;

struct Map {
    cv::Mat s;  // HSV saturation
    cv::Mat v;  // HSV value
    cv::Mat y;  // yellow, closed over the ink
    cv::Point2d origin;  // robot mm at pixel (0, 0); robot x along columns, y along rows

    [[nodiscard]] cv::Point2d toPx(cv::Point2d r) const { return (r - origin) * kPxPerMm; }
    [[nodiscard]] cv::Point2d toRobot(cv::Point2d px) const { return origin + px / kPxPerMm; }
};

Map makeMap(const cv::Mat& img, const Geometry& geo, const HeapRegion& region)
{
    Map m;
    m.origin = {region.robot.x, region.robot.y};
    const int w = static_cast<int>(std::lround(region.robot.width * kPxPerMm));
    const int h = static_cast<int>(std::lround(region.robot.height * kPxPerMm));
    const double imageScale = static_cast<double>(img.cols) / 1600.0;
    cv::Mat mx(h, w, CV_32F);
    cv::Mat my(h, w, CV_32F);
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            const cv::Point2d px = geo.robotToImage(m.toRobot({static_cast<double>(c), static_cast<double>(r)})) *
                                   imageScale;
            mx.at<float>(r, c) = static_cast<float>(px.x);
            my.at<float>(r, c) = static_cast<float>(px.y);
        }
    }
    cv::Mat top;
    cv::remap(img, top, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat hsv;
    cv::cvtColor(top, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    m.s = ch[1];
    m.v = ch[2];
    cv::inRange(hsv, cv::Scalar(18, 80, 70), cv::Scalar(40, 255, 255), m.y);
    cv::morphologyEx(m.y, m.y, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    for (const auto& e : region.exclude) {
        const cv::Point2d p = m.toPx(e);
        cv::circle(m.y, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)),
                   static_cast<int>(region.excludeMm * kPxPerMm), cv::Scalar(0), cv::FILLED);
    }
    return m;
}

// Sample offsets (mm, in the tile's own frame: u along the long axis, w across).
struct Template {
    std::vector<cv::Point2d> edge;    // points on the outline
    std::vector<cv::Point2d> normal;  // outward normals
    std::vector<int> side;            // 0..3
    std::vector<cv::Point2d> inside;
};

Template makeTemplate(double longMm, double shortMm)
{
    Template t;
    const double hl = longMm / 2.0;
    const double hw = shortMm / 2.0;
    for (double u = -hl + kCornerMm; u <= hl - kCornerMm; u += 1.0) {
        t.edge.emplace_back(u, hw);
        t.normal.emplace_back(0.0, 1.0);
        t.side.push_back(0);
        t.edge.emplace_back(u, -hw);
        t.normal.emplace_back(0.0, -1.0);
        t.side.push_back(1);
    }
    for (double w = -hw + kCornerMm; w <= hw - kCornerMm; w += 1.0) {
        t.edge.emplace_back(hl, w);
        t.normal.emplace_back(1.0, 0.0);
        t.side.push_back(2);
        t.edge.emplace_back(-hl, w);
        t.normal.emplace_back(-1.0, 0.0);
        t.side.push_back(3);
    }
    for (double u = -hl + 4.0; u <= hl - 4.0; u += 4.0) {
        for (double w = -hw + 4.0; w <= hw - 4.0; w += 4.0) {
            t.inside.emplace_back(u, w);
        }
    }
    return t;
}

struct Eval {
    double fill = 0.0;
    double border = 0.0;
    double weakest = 0.0;
    double contrast = 0.0;  // mean edge step along the outline: peaks when the outline sits on the edges
};

// Pose at map pixel c, long axis at angle a (radians, map frame = robot frame).
Eval evaluate(const Map& m, const Template& t, cv::Point2d c, double a)
{
    const cv::Point2d u(std::cos(a), std::sin(a));
    const cv::Point2d w(-u.y, u.x);
    auto at = [&](const cv::Mat& img, cv::Point2d mm) -> int {
        const cv::Point2d p = c + (u * mm.x + w * mm.y) * kPxPerMm;
        const int x = static_cast<int>(std::lround(p.x));
        const int y = static_cast<int>(std::lround(p.y));
        if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) {
            return -1;
        }
        return img.at<uchar>(y, x);
    };
    Eval e;
    int in = 0;
    for (const auto& p : t.inside) {
        in += at(m.y, p) > 0 ? 1 : 0;
    }
    e.fill = static_cast<double>(in) / static_cast<double>(t.inside.size());
    if (e.fill < kMinFill) {
        return e;
    }
    // Per side: the fraction of its points where the yellow inside steps to something else
    // outside: the table (saturation drops), a shadow, or the tile's lit edge face (brightness
    // changes). Ink is not an edge: dark pixels on either side do not count.
    std::array<int, 4> hit{};
    std::array<int, 4> total{};
    for (size_t i = 0; i < t.edge.size(); ++i) {
        const auto sd = static_cast<size_t>(t.side[i]);
        ++total[sd];
        const cv::Point2d pin = t.edge[i] - t.normal[i] * kEdgeMm;
        const cv::Point2d pout = t.edge[i] + t.normal[i] * kEdgeMm;
        if (at(m.y, pin) <= 0) {
            continue;
        }
        const int sIn = at(m.s, pin);
        const int sOut = at(m.s, pout);
        const int vIn = at(m.v, pin);
        const int vOut = at(m.v, pout);
        if (sOut < 0 || vOut < 0 || vIn < kInk || vOut < kInk) {
            continue;
        }
        if (sIn - sOut > kEdgeS || std::abs(vIn - vOut) > kEdgeV) {
            ++hit[sd];
        }
        e.contrast += std::max(sIn - sOut, std::abs(vIn - vOut));
    }
    e.contrast /= static_cast<double>(t.edge.size());
    std::array<double, 4> f{};
    for (size_t k = 0; k < 4; ++k) {
        f[k] = total[k] > 0 ? static_cast<double>(hit[k]) / total[k] : 0.0;
    }
    std::sort(f.begin(), f.end());
    e.weakest = f[0];
    // One side may lie against a tile of the same height with no visible step; a tile with
    // something on top loses part of two sides, or most of one.
    e.border = (f[1] + f[2] + f[3]) / 3.0;
    if (f[0] < kMinSide) {
        e.border *= 0.5;
    }
    return e;
}

}  // namespace

std::vector<HeapTile> findHeapTiles(const cv::Mat& img, const Geometry& geo, const HeapRegion& region,
                                    const DigitNet* net)
{
    const Map m = makeMap(img, geo, region);
    const Template t = makeTemplate(geo.tileLong(), geo.tileShort());

    // Centres: deep enough inside the yellow for a whole tile around them.
    cv::Mat dist;
    cv::distanceTransform(m.y, dist, cv::DIST_L2, 3);
    const double minInside = (geo.tileShort() / 2.0 - 4.0) * kPxPerMm;
    const int step = static_cast<int>(2.0 * kPxPerMm);

    struct Hit {
        cv::Point2d c;
        double a = 0.0;
        Eval e;
    };
    std::vector<Hit> hits;
    for (int r = 0; r < dist.rows; r += step) {
        for (int c = 0; c < dist.cols; c += step) {
            if (dist.at<float>(r, c) < minInside) {
                continue;
            }
            Hit best{{static_cast<double>(c), static_cast<double>(r)}, 0.0, {}};
            for (int k = 0; k < 36; ++k) {
                const double a = k * 5.0 * kDeg;
                const Eval e = evaluate(m, t, best.c, a);
                if (e.fill >= kMinFill && e.border > best.e.border) {
                    best.a = a;
                    best.e = e;
                }
            }
            if (best.e.border >= kMinBorder * 0.8) {
                hits.push_back(best);
            }
        }
    }
    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.e.border > b.e.border; });

    // Refine the strongest few, then keep one per tile.
    const double nmsPx = 15.0 * kPxPerMm;
    std::vector<Hit> kept;
    for (const auto& h : hits) {
        if (kept.size() >= 40) {
            break;
        }
        bool near = false;
        for (const auto& k : kept) {
            near = near || cv::norm(k.c - h.c) < nmsPx;
        }
        if (near) {
            continue;
        }
        // Refine: the outline that sits best on the edges (contrast), among poses that keep the
        // edge fraction; coordinate steps, coarse to fine.
        Hit best = h;
        auto better = [&](const Eval& e) {
            return e.fill >= kMinFill && e.border >= h.e.border - 0.05 && e.contrast > best.e.contrast;
        };
        for (const double stepMm : {1.0, 0.5, 0.25}) {
            const double stepA = stepMm * 2.0 * kDeg;  // 2, 1, 0.5 degrees
            bool moved = true;
            for (int it = 0; it < 12 && moved; ++it) {
                moved = false;
                const Hit start = best;
                for (const auto& [dx, dy, da] : std::array<std::array<int, 3>, 6>{
                         {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}}) {
                    const cv::Point2d c = start.c + cv::Point2d(dx, dy) * (stepMm * kPxPerMm);
                    const double a = start.a + da * stepA;
                    const Eval e = evaluate(m, t, c, a);
                    if (better(e)) {
                        best = {c, a, e};
                        moved = true;
                    }
                }
            }
        }
        kept.push_back(best);
    }

    // Separate tiles: their yellow component is about one tile.
    cv::Mat lab;
    cv::Mat stats;
    cv::Mat cent;
    cv::connectedComponentsWithStats(m.y, lab, stats, cent);
    const double tilePx = geo.tileLong() * geo.tileShort() * kPxPerMm * kPxPerMm;

    std::vector<HeapTile> out;
    for (const auto& k : kept) {
        if (k.e.border < kMinBorder) {
            continue;
        }
        HeapTile tile;
        tile.pose = {m.toRobot(k.c), wrap180(k.a / kDeg)};
        tile.border = k.e.border;
        tile.fill = k.e.fill;
        const int id = lab.at<int>(static_cast<int>(std::lround(k.c.y)), static_cast<int>(std::lround(k.c.x)));
        if (id > 0) {
            const double area = stats.at<int>(id, cv::CC_STAT_AREA);
            tile.lone = area > 0.75 * tilePx && area < 1.35 * tilePx;
        }
        tile.score = tile.border + (tile.lone ? 1.0 : 0.0);
        if (net != nullptr && net->ready()) {
            std::array<cv::Mat, 4> patches;
            for (int q = 0; q < 4; ++q) {
                patches[static_cast<size_t>(q)] =
                    extractPatch(img, geo, {tile.pose.xy, tile.pose.up + 90.0 * q}, static_cast<double>(img.cols) / 1600.0);
            }
            tile.reading = net->read(patches);
            if (tile.reading.digit >= 0) {
                tile.pose.up = wrap180(tile.pose.up + 90.0 * tile.reading.quarter);
                tile.score += 0.5 * std::min(1.0, static_cast<double>(tile.reading.margin));
            } else {
                tile.score -= 0.5;
            }
        }
        out.push_back(tile);
    }
    std::sort(out.begin(), out.end(), [](const HeapTile& a, const HeapTile& b) { return a.score > b.score; });
    return out;
}

std::optional<HeapTile> locateTile(const cv::Mat& img, const Geometry& geo, cv::Point2d near, double maxMm,
                                   const DigitNet* net)
{
    HeapRegion region;
    const double half = maxMm + geo.tileLong();
    region.robot = cv::Rect2d(near.x - half, near.y - half, 2.0 * half, 2.0 * half);
    std::optional<HeapTile> best;
    for (const auto& t : findHeapTiles(img, geo, region, net)) {
        const double d = cv::norm(t.pose.xy - near);
        if (d <= maxMm && (!best || d < cv::norm(best->pose.xy - near))) {
            best = t;
        }
    }
    return best;
}

cv::Mat drawHeapTiles(const cv::Mat& img, const Geometry& geo, const std::vector<HeapTile>& tiles)
{
    cv::Mat out = img.clone();
    const double imageScale = static_cast<double>(img.cols) / 1600.0;
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& t = tiles[i];
        const cv::Point2d u(std::cos(t.pose.up * kDeg), std::sin(t.pose.up * kDeg));
        const cv::Point2d w(-u.y, u.x);
        const double hl = geo.tileLong() / 2.0;
        const double hw = geo.tileShort() / 2.0;
        std::vector<cv::Point> poly;
        for (const auto& [a, b] : std::array<std::pair<double, double>, 4>{{{hl, hw}, {hl, -hw}, {-hl, -hw}, {-hl, hw}}}) {
            const cv::Point2d p = geo.robotToImage(t.pose.xy + u * a + w * b) * imageScale;
            poly.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
        }
        const cv::Scalar col = t.lone ? cv::Scalar(0, 200, 0) : (i == 0 ? cv::Scalar(255, 0, 255) : cv::Scalar(255, 80, 0));
        cv::polylines(out, poly, true, col, 2);
        // Digit top: a tick from the centre towards the up edge.
        const cv::Point2d c = geo.robotToImage(t.pose.xy) * imageScale;
        const cv::Point2d tip = geo.robotToImage(t.pose.xy + u * (hl - 4.0)) * imageScale;
        cv::line(out, c, tip, col, 2);
        std::string label = std::to_string(i) + ":" +
                            (t.reading.digit >= 0 ? std::to_string(t.reading.digit) : std::string("?")) + " " +
                            std::to_string(static_cast<int>(std::lround(t.border * 100.0)));
        cv::putText(out, label, c + cv::Point2d(-20, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
        cv::putText(out, label, c + cv::Point2d(-20, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    return out;
}

}  // namespace spc
