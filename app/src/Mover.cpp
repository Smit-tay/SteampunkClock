// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "Mover.h"

#include "Log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <thread>

namespace spc {
namespace {

void sleepS(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }

}  // namespace

Mover::Mover(Arm& arm, const Geometry& geo, const nlohmann::json& motion)
    : arm_(arm),
      geo_(geo),
      travelZ_(motion.at("travel_z").get<double>()),
      park_(motion.at("park").at(0).get<double>(), motion.at("park").at(1).get<double>(),
            motion.at("park").at(2).get<double>()),
      fast_(motion.at("speed_fast").get<double>()),
      near_(motion.at("speed_near").get<double>()),
      final_(motion.at("speed_final").get<double>()),
      approach_(motion.at("approach").get<double>()),
      press_(motion.at("press").get<double>()),
      searchStep_(motion.value("search_step", 2.0)),
      wristScale_(motion.value("wrist_scale", 1.0)),
      corrections_([&] {
          std::vector<cv::Vec4d> c;
          for (const auto& e : motion.value("place_corrections", nlohmann::json::array())) {
              c.emplace_back(e.at(0).get<double>(), e.at(1).get<double>(), e.at(2).get<double>(),
                             e.at(3).get<double>());
          }
          return c;
      }()),
      vacuumS_(motion.at("vacuum_s").get<double>()),
      releaseS_(motion.at("release_s").get<double>()),
      reachMin_(motion.at("reach_min").get<double>()),
      reachMax_(motion.at("reach_max").get<double>())
{
}

namespace {

// Where the arm hides the table in the photo, measured 2026-11-15 (spclock armview at four
// poses): the camera looks in at an angle, so the raised arm shows as a band 20-30 mm wide from
// ~20 mm beyond the cup (60 with the cable loop seen 2026-11-15) towards (40, 0.75 y + 10), nearly along -x and not towards the base;
// the base housing always covers about 60 mm around (30, 0).
constexpr double kArmHalfWidthMm = 35.0;
constexpr double kArmBeyondCupMm = 60.0;  // the pump cable loops out past the cup
const cv::Point2d kBaseSeen(30.0, 0.0);
constexpr double kBaseSeenMm = 60.0;

double distToSegment(cv::Point2d p, cv::Point2d a, cv::Point2d b)
{
    const cv::Point2d ab = b - a;
    const double len2 = ab.dot(ab);
    const double t = len2 > 0.0 ? std::clamp((p - a).dot(ab) / len2, 0.0, 1.0) : 0.0;
    return cv::norm(p - (a + ab * t));
}

bool hides(cv::Point2d cup, cv::Point2d p, double radius)
{
    if (cv::norm(p - kBaseSeen) < kBaseSeenMm + radius) {
        return true;  // always covered, nothing to gain by moving
    }
    const cv::Point2d tail(40.0, 0.75 * cup.y + 10.0);
    const cv::Point2d d = cup - tail;
    const double len = cv::norm(d);
    if (len < 1e-6) {
        return false;
    }
    const cv::Point2d tip = cup + d * (kArmBeyondCupMm / len);
    return distToSegment(p, tail, tip) < kArmHalfWidthMm + radius;
}

}  // namespace

void Mover::clearView(const std::vector<cv::Point2d>& keep, double radiusMm)
{
    const cv::Point3d at = arm_.position();
    const cv::Point2d cup(at.x, at.y);
    auto clear = [&](cv::Point2d c) {
        for (const auto& p : keep) {
            if (cv::norm(p - kBaseSeen) < kBaseSeenMm + radiusMm) {
                continue;  // under the base housing: hidden wherever the arm is
            }
            if (hides(c, p, radiusMm)) {
                return false;
            }
        }
        return true;
    };
    if (clear(cup)) {
        log("clearView: arm at (", cup.x, ", ", cup.y, ") already clear");
        return;
    }
    // Candidates: the base swung at the current radius, or pulled in to 200 or 160 mm; the
    // clear one nearest the cup wins (least travel).
    const double r0 = std::hypot(cup.x, cup.y);
    constexpr double kStep = 5.0 * std::numbers::pi / 180.0;
    // Base angles the arm accepts at travel height (E25 below about -65 deg, measured 2026-11-15).
    constexpr double kMinA = -65.0 * std::numbers::pi / 180.0;
    constexpr double kMaxA = 80.0 * std::numbers::pi / 180.0;
    std::vector<cv::Point2d> cands;
    for (const double r : {std::max(r0, reachMin_), 200.0, 160.0}) {
        for (double a = kMinA; a <= kMaxA + 1e-9; a += kStep) {
            const cv::Point2d c(r * std::cos(a), r * std::sin(a));
            if (clear(c)) {
                cands.push_back(c);
            }
        }
    }
    std::sort(cands.begin(), cands.end(),
              [&](cv::Point2d p, cv::Point2d q) { return cv::norm(p - cup) < cv::norm(q - cup); });
    for (const auto& c : cands) {
        try {
            arm_.moveRaw({c.x, c.y, std::max(at.z, travelZ_)}, fast_);
        } catch (const ArmError& e) {
            log("clearView: (", c.x, ", ", c.y, ") refused (", e.what(), "), trying the next");
            continue;  // outside the arm's range there
        }
        log("clearView: moved ", cv::norm(c - cup), " mm to (", c.x, ", ", c.y, ")");
        return;
    }
    park();  // nothing clear by turning: the fixed park pose
}

cv::Point2d Mover::placeCorrection(cv::Point2d p) const
{
    // Inverse-distance blend of the measured corrections (x, y, dx, dy); ~30 mm smoothing.
    if (!correct_ || corrections_.empty()) {
        return {0.0, 0.0};
    }
    cv::Point2d sum(0.0, 0.0);
    double wsum = 0.0;
    for (const auto& c : corrections_) {
        const double d2 = (p.x - c[0]) * (p.x - c[0]) + (p.y - c[1]) * (p.y - c[1]);
        const double w = 1.0 / (d2 + 900.0);
        sum += cv::Point2d(c[2], c[3]) * w;
        wsum += w;
    }
    return sum / wsum;
}

bool Mover::inReach(cv::Point2d p) const
{
    const double r = std::hypot(p.x, p.y);
    return r >= reachMin_ && r <= reachMax_;
}

void Mover::park()
{
    arm_.pump(false);
    arm_.wrist(90.0);
    arm_.move({park_.x, park_.y}, park_.z, fast_);
}

void Mover::pickKnown(const Pose& src, int level, double wrist)
{
    const double seal = geo_.sealZ(src.xy, level);
    arm_.wrist(wrist);
    arm_.move(src.xy, travelZ_, fast_);
    arm_.move(src.xy, seal + approach_, fast_);
    arm_.pump(true);
    arm_.move(src.xy, seal - press_, final_);
    sleepS(vacuumS_);
    arm_.move(src.xy, travelZ_, near_);
}

bool Mover::touching() const
{
    // pump_status updates at 5 Hz: 1 = sealed, 2 = running free. Any 1 means the cup touches.
    // The first reading after a move can be stale, so look a few times; stop at the first seal.
    for (int i = 0; i < 3; ++i) {
        sleepS(0.2);
        if (arm_.pumpStatus() == 1) {
            return true;
        }
    }
    return false;
}

bool Mover::pickSearch(const Pose& src, double zStart, double wrist, std::optional<double> expectSeal)
{
    arm_.wrist(wrist);
    arm_.move(src.xy, travelZ_, fast_);
    const double floor = geo_.surface(src.xy) - 1.0;  // never press far into the table
    double z = std::max(zStart, floor);
    arm_.move(src.xy, z, near_);
    arm_.pump(true);
    sleepS(0.4);  // the first pump_status after switching on is stale (read as a false contact)
    bool contact = false;
    int steps = 0;
    while (true) {
        contact = touching();
        if (contact || z - searchStep_ < floor) {
            break;
        }
        z -= searchStep_;
        ++steps;
        arm_.move(src.xy, z, final_);
    }
    log("pickSearch: ", contact ? "contact" : "no contact", " after ", steps, " steps at z ", z);
    bool held = false;
    if (contact) {
        // A stray contact reading can come ~3 mm above the tile: press on, at least to the
        // predicted seal when the tile's height is known.
        double target = z - 2.0;
        if (expectSeal) {
            target = std::min(target, *expectSeal - press_);
        }
        z = std::max(target, floor);
        arm_.move(src.xy, z, final_);
        held = true;
    } else if (expectSeal && z <= *expectSeal - press_) {
        held = true;  // pressed past where a tile lying there seals: assume it holds
    }
    if (held) {
        sleepS(vacuumS_ + 0.2);
    } else {
        arm_.pump(false);
    }
    arm_.move(src.xy, travelZ_, near_);
    return held;
}

void Mover::place(const Pose& dst, int level, double wrist)
{
    const double z = geo_.releaseZ(dst.xy, level);
    const cv::Point2d at = dst.xy + placeCorrection(dst.xy);
    arm_.wrist(wrist);  // before moving there: the cup correction depends on it
    arm_.move(at, travelZ_, fast_);
    arm_.move(at, z + approach_, fast_);
    arm_.move(at, z, final_);
    arm_.pump(false);
    sleepS(releaseS_);
    arm_.move(at, travelZ_, near_);  // straight up: no drag on the tile
}

void Mover::carry(const Pose& src, int srcLevel, const Pose& dst, int dstLevel)
{
    const auto w = plan(src, dst);
    pickKnown(src, srcLevel, w.pick);
    place(dst, dstLevel, w.place);
    regrip(dst, dstLevel, w);
}

void Mover::regrip(const Pose& dst, int level, const Geometry::WristPlan& w)
{
    for (int k = 1; k < w.steps; ++k) {
        log("regrip: turn ", k + 1, " of ", w.steps);
        pickKnown(dst, level, w.pick);  // the tile lies at dst: place() aims for it
        place(dst, level, w.place);
    }
}

}  // namespace spc
