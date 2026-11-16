// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Mover: pick and place single tiles with the suction cup.
// The pump's pressure sensor cannot be trusted to report a held tile (2026-11-15: the cup held
// tiles while it reported no seal), so a pick is "blind": press the cup onto the tile, wait for
// the vacuum and lift. The caller checks with the camera that the tile left its spot.
#pragma once

#include "Arm.h"
#include "Geometry.h"

#include <nlohmann/json_fwd.hpp>

#include <optional>
#include <vector>

namespace spc {

class Mover {
public:
    Mover(Arm& arm, const Geometry& geo, const nlohmann::json& motion);

    // Tile whose seal height is known: on a stack `level` tiles up, or flat on the table (0).
    void pickKnown(const Pose& src, int level, double wrist);
    // Tile at an unknown height (heap): lower search_step mm at a time from zStart until the pump reports
    // contact, then press 2 mm (at least down to expectSeal - press when given). Without contact,
    // a cup pressed 1 mm past expectSeal is taken as holding. Returns false (pump off) when
    // neither happened before the table.
    bool pickSearch(const Pose& src, double zStart, double wrist, std::optional<double> expectSeal);
    // Put the held tile down on a surface `level` tiles high (bottom ~1 mm above it) and let go.
    void place(const Pose& dst, int level, double wrist);
    // After a place with plan w: the remaining w.steps - 1 turns (pick the tile up again at dst
    // with the wrist turned back, place it turned once more). Nothing when w.steps is 1.
    void regrip(const Pose& dst, int level, const Geometry::WristPlan& w);
    // pickKnown + place (+ regrip) with the wrist turned so the tile arrives with dst.up.
    void carry(const Pose& src, int srcLevel, const Pose& dst, int dstLevel);
    void park();
    // Wrist angles for carrying src to dst, with the configured wrist scale.
    [[nodiscard]] Geometry::WristPlan plan(const Pose& src, const Pose& dst) const
    {
        return Geometry::planWrist(src, dst, wristScale_);
    }
    // Keep the arm out of the camera's view of `keep` (robot points, each with `radiusMm` around
    // it). Does nothing when it is already clear; otherwise swings the base by the smallest angle
    // that clears them, at travel height. Cheaper than park() between moves.
    void clearView(const std::vector<cv::Point2d>& keep, double radiusMm);

    [[nodiscard]] bool inReach(cv::Point2d p) const;
    // Offset added to a place target (from motion.place_corrections); calplace measures without it.
    [[nodiscard]] cv::Point2d placeCorrection(cv::Point2d p) const;
    void setCorrection(bool on) { correct_ = on; }
    [[nodiscard]] double travelZ() const { return travelZ_; }

private:
    [[nodiscard]] bool touching() const;  // pump status shows contact within ~0.8 s

    Arm& arm_;
    const Geometry& geo_;
    double travelZ_;
    cv::Point3d park_;
    double fast_;
    double near_;
    double final_;
    double approach_;
    double press_;
    double searchStep_;
    double wristScale_;  // commanded wrist turn per wanted tile turn (the wrist falls short)
    std::vector<cv::Vec4d> corrections_;  // motion.place_corrections: x, y, dx, dy (mm, robot frame)
    bool correct_ = true;    // degrees added to every carry's turn (measured placement shortfall)  // mm per step when feeling for a tile
    double vacuumS_;
    double releaseS_;
    double reachMin_;
    double reachMax_;
};

}  // namespace spc
