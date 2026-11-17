// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// sort: clear the heap into the digit stacks, one tile per photo. Each round takes the best
// candidate from heap vision (a separate tile first, else a tile with nothing on top), reads
// it, lifts it (feeling for its height on the heap), and puts it on a stack of its digit with
// the digit's top along the stacks' direction. The next photo shows whether it arrived.
// Tiles far out (placement there is inaccurate) or unreadable on the heap go to the temporary
// spot first, are read there alone, and continue from there.
#include "Commands.h"

#include "DigitNet.h"
#include "Heap.h"
#include "Log.h"
#include "State.h"
#include "Vision.h"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace spc {
namespace {

constexpr double kMinMargin = 0.3;      // digit score over the runner-up for a trusted reading
constexpr double kMinMarginHeap = 0.8;  // on the heap (neighbours in the patch): read again alone below this
constexpr double kFarMm = 285.0;      // beyond this, go via the temporary spot
constexpr double kChanged = 12.0;     // mean grey change at a spot that means a tile came or went
constexpr int kMaxFailsAtSpot = 2;
constexpr int kMaxRounds = 120;

DigitNet::Reading readAt(const cv::Mat& img, const Geometry& geo, const DigitNet& net, const Pose& pose)
{
    std::array<cv::Mat, 4> patches;
    for (int k = 0; k < 4; ++k) {
        patches[static_cast<size_t>(k)] = extractPatch(img, geo, {pose.xy, wrap180(pose.up + 90.0 * k)});
    }
    return net.read(patches);
}

HeapRegion regionFor(const App& app, const State& st)
{
    HeapRegion region;
    // Only where tiles are known to lie: a heap tile may well sit on an empty slot.
    for (const auto& s : st.stacks) {
        if (s.n > 0) {
            region.exclude.push_back(s.xy);
        }
    }
    for (const auto& d : st.display) {
        if (d.digit >= 0) {
            region.exclude.push_back(d.xy);
        }
    }
    region.exclude.push_back(app.tempSpot);
    return region;
}

double secondsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

class Sorter {
public:
    Sorter(App& app, DigitNet& net) : app_(app), net_(net), st_(State::load(app.stateFile)) {}

    void run()
    {
        const auto t0 = std::chrono::steady_clock::now();
        int moved = 0;
        app_.mover.park();  // once: nothing known yet about where the heap is
        // A tile left on the temporary spot (an earlier run stopped) goes first.
        if (const cv::Mat img = app_.cam.grab(Camera::Res::Hi); locateTile(img, app_.geo, app_.tempSpot, 25.0, &net_)) {
            log("sort: a tile waits on the temporary spot");
            if (fromTemp(img, t0)) {
                ++moved;
            }
        }
        for (int round = 0; round < kMaxRounds; ++round) {
            app_.mover.clearView(heapSeen_, 25.0);
            const cv::Mat img = app_.cam.grab(Camera::Res::Hi);
            const auto tiles = findHeapTiles(img, app_.geo, regionFor(app_, st_), &net_);
            heapSeen_.clear();
            for (const auto& t : tiles) {
                heapSeen_.push_back(t.pose.xy);
            }
            const HeapTile* pick = choose(tiles);
            if (pick == nullptr) {
                log("sort: no liftable tile left (", tiles.size(), " candidates rejected)");
                if (!tiles.empty()) {
                    cv::imwrite((app_.dataDir / "photos" / "sort_stuck.jpg").string(),
                                drawHeapTiles(img, app_.geo, tiles));
                }
                break;
            }
            log("sort: round ", round, ": ", pick->lone ? "lone" : "top", " tile \"", pick->reading.digit, "\" at (",
                pick->pose.xy.x, ", ", pick->pose.xy.y, ") border ", pick->border, " margin ", pick->reading.margin);
            if (moveOne(*pick, img)) {
                ++moved;
            }
        }
        app_.mover.clearView(app_.displaySlots, 25.0);
        log("sort: ", moved, " tiles stacked in ", secondsSince(t0), " s");
    }

private:
    [[nodiscard]] std::string spotKey(cv::Point2d p) const
    {
        return std::to_string(static_cast<int>(std::lround(p.x / 10.0))) + "," +
               std::to_string(static_cast<int>(std::lround(p.y / 10.0)));
    }

    [[nodiscard]] const HeapTile* choose(const std::vector<HeapTile>& tiles) const
    {
        for (const auto& t : tiles) {  // best first
            if (!app_.mover.inReach(t.pose.xy)) {  // motion.reach_min .. reach_max
                continue;
            }
            const auto f = fails_.find(spotKey(t.pose.xy));
            if (f != fails_.end() && f->second >= kMaxFailsAtSpot) {
                continue;
            }
            return &t;
        }
        return nullptr;
    }

    [[nodiscard]] bool readable(const DigitNet::Reading& r) const
    {
        return r.digit >= 0 && r.margin >= kMinMargin;
    }

    // Lift the tile (known height when it lies alone on the table), true when held.
    bool lift(const HeapTile& t, double wrist)
    {
        if (t.lone) {
            app_.mover.pickKnown(t.pose, 0, wrist);
            return true;
        }
        // On the heap: feel down from just above a three-high pile (the pump runs only from there).
        const double zStart = app_.geo.sealZ(t.pose.xy, 3) + 1.0;
        return app_.mover.pickSearch(t.pose, zStart, wrist, std::nullopt);
    }

    bool moveOne(const HeapTile& t, const cv::Mat& before)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const double r = std::hypot(t.pose.xy.x, t.pose.xy.y);
        const bool sure = readable(t.reading) && (t.lone || t.reading.margin >= kMinMarginHeap);
        const bool viaTemp = !sure || r > kFarMm;
        Stack* dst = nullptr;
        if (!viaTemp) {
            dst = st_.putOn(t.reading.digit, app_.maxStack, t.pose.xy);
            if (dst == nullptr) {
                log("sort: no stack with room for \"", t.reading.digit, "\"");
                fails_[spotKey(t.pose.xy)] = kMaxFailsAtSpot;  // not again this run
                return false;
            }
        }
        const Pose target = viaTemp ? Pose{app_.tempSpot, app_.upStack} : Pose{dst->xy, app_.upStack};
        const int level = viaTemp ? 0 : dst->n;
        // Unread tiles keep their long-axis guess; the reading on the temporary spot fixes it.
        const auto w = app_.mover.plan(t.pose, target);
        if (!lift(t, w.pick)) {
            log("sort: no grip at (", t.pose.xy.x, ", ", t.pose.xy.y, ")");
            ++fails_[spotKey(t.pose.xy)];
            return false;
        }
        app_.mover.place(target, level, w.place);
        app_.mover.regrip(target, level, w);
        app_.mover.clearView({t.pose.xy, target.xy}, 25.0);
        const cv::Mat after = app_.cam.grab(Camera::Res::Hi);
        const double left = changeAt(before, after, app_.geo, t.pose.xy);
        const double arrived = changeAt(before, after, app_.geo, target.xy);
        if (arrived < kChanged) {
            log("sort: nothing arrived (change ", arrived, " there, ", left, " at the pick)");
            ++fails_[spotKey(t.pose.xy)];
            return false;
        }
        if (viaTemp) {
            return fromTemp(after, t0);
        }
        // Read the stack top where it actually landed (placement is a few mm and degrees off).
        const auto found = locateTile(after, app_.geo, target.xy, 15.0, &net_);
        const Pose landed = found ? found->pose : target;
        const auto check = found ? found->reading : readAt(after, app_.geo, net_, target);
        ++dst->n;
        st_.save(app_.stateFile);
        // locateTile has already turned the pose to the digit's top (its quarter is relative to
        // its first guess); readAt reads at the target pose, so its quarter is the offset.
        const double off = found ? std::abs(wrap180(landed.up - target.up)) : 90.0 * check.quarter;
        const bool upright = off < 45.0 || (DigitNet::halfTurnSymmetric(check.digit) && off > 135.0 && off < 225.0);
        if (check.digit != t.reading.digit || !upright) {
            log("sort: WARNING stack ", dst->key, " top reads \"", check.digit, "\" quarter ", check.quarter,
                " (margin ", check.margin, ")");
        }
        log("sort: \"", t.reading.digit, "\" onto stack ", dst->key, " (now ", dst->n, ") in ", secondsSince(t0),
            " s, landed ", cv::norm(landed.xy - target.xy), " mm and ", wrap180(landed.up - target.up), " deg off");
        return true;
    }

    // The tile now lies alone on the temporary spot: find it, read it, stack it.
    bool fromTemp(const cv::Mat& img, std::chrono::steady_clock::time_point t0)
    {
        const auto found = locateTile(img, app_.geo, app_.tempSpot, 25.0, &net_);
        if (!found) {
            cv::imwrite((app_.dataDir / "photos" / "sort_temp_lost.jpg").string(), img);
            throw std::runtime_error("sort: the tile on the temporary spot is not where expected");
        }
        const auto& rd = found->reading;
        if (!readable(rd)) {
            cv::imwrite((app_.dataDir / "photos" / "sort_unreadable.jpg").string(), img);
            throw std::runtime_error("sort: cannot read the tile on the temporary spot (best \"" +
                                     std::to_string(rd.digit) + "\", margin " + std::to_string(rd.margin) + ")");
        }
        const Pose src = found->pose;  // up already turned to the digit's top by the reading
        Stack* dst = st_.putOn(rd.digit, app_.maxStack, src.xy);
        if (dst == nullptr) {
            throw std::runtime_error("sort: no stack with room for \"" + std::to_string(rd.digit) + "\"");
        }
        app_.mover.carry(src, 0, {dst->xy, app_.upStack}, dst->n);
        ++dst->n;
        st_.save(app_.stateFile);
        log("sort: \"", rd.digit, "\" via the temporary spot onto stack ", dst->key, " (now ", dst->n, ") in ",
            secondsSince(t0), " s");
        return true;
    }

    App& app_;
    DigitNet& net_;
    State st_;
    std::map<std::string, int> fails_;
    std::vector<cv::Point2d> heapSeen_;  // candidates of the last photo: keep them in view  // per 10 mm cell: failed picks there
};

}  // namespace

void sort(App& app)
{
    if (!std::filesystem::exists(app.modelFile)) {
        throw std::runtime_error("sort: no digit model (" + app.modelFile.string() + "); run train first");
    }
    DigitNet net;
    net.load(app.modelFile);
    const ArmLock lock(app.lockFile, 5.0);
    Sorter(app, net).run();
}

}  // namespace spc
