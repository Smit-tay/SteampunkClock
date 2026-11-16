// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// collect: training data from the sorted tiles. Each tile of each stack goes to the photo spot,
// is photographed at four orientations (arm swung clear of the camera's view) and then waits on a temporary
// stack; when its stack is done, the tiles go back. Every photo gives one upright patch of a
// known digit and some "not a whole tile" patches around it.
#include "Commands.h"

#include "Dataset.h"
#include "Log.h"
#include "State.h"
#include "Vision.h"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace spc {

void collect(App& app, const std::string& digits)
{
    const ArmLock lock(app.lockFile, 5.0);
    State st = State::load(app.stateFile);
    const auto outDir = app.dataDir / "collect";
    const auto photoDir = app.dataDir / "photos";
    std::filesystem::create_directories(photoDir);
    cv::RNG rng(static_cast<uint64_t>(std::time(nullptr)));
    const double up = app.upStack;
    const Pose spot{app.photoSpot, up};
    app.mover.park();

    for (auto& stack : st.stacks) {
        if (stack.n == 0 || digits.find(static_cast<char>('0' + stack.digit)) == std::string::npos) {
            continue;
        }
        int waiting = 0;  // tiles of this stack on the temporary stack
        while (stack.n > 0) {
            const std::string name = stack.key + "_" + std::to_string(stack.n - 1) + "_" + std::to_string(std::time(nullptr));
            app.mover.carry({stack.xy, up}, stack.n - 1, spot, 0);
            --stack.n;
            st.save(app.stateFile);
            Pose at = spot;
            Pose prev{stack.xy, up};  // where the tile on the spot came from (for the wrist angle)
            for (int r = 0; r < 4; ++r) {
                app.mover.clearView({spot.xy}, 25.0);
                const cv::Mat img = app.cam.grab(Camera::Res::Hi);
                const auto found = findLoneTile(img, app.geo, at, 15.0);
                if (!found) {
                    cv::imwrite((photoDir / (name + "_lost.jpg")).string(), img);
                    throw std::runtime_error("collect: tile " + name + " not found on the photo spot");
                }
                const double off = std::hypot(found->xy.x - at.xy.x, found->xy.y - at.xy.y);
                const auto w = app.mover.plan(prev, at);
                log("collect: \"", stack.digit, "\" ", name, " turn ", r, ": placed ", off, " mm off (", found->xy.x - at.xy.x,
                    ", ", found->xy.y - at.xy.y, "), up ", wrap180(found->up - at.up), " deg off, wrist ", w.place);
                at = *found;
                const std::string n = name + "_r" + std::to_string(r);
                saveSample(outDir, std::to_string(stack.digit), n, extractPatch(img, app.geo, at));
                for (int j = 0; j < 3; ++j) {
                    const double a = rng.uniform(0.0, 2.0 * std::numbers::pi);
                    const double d = rng.uniform(15.0, 24.0);
                    const Pose off2{at.xy + cv::Point2d(std::cos(a), std::sin(a)) * d, at.up + rng.uniform(-45.0, 45.0)};
                    saveSample(outDir, "junk", n + "_j" + std::to_string(j), extractPatch(img, app.geo, off2));
                }
                if (r == 0) {
                    cv::imwrite((photoDir / (name + ".jpg")).string(), img);
                }
                if (r < 3) {
                    const Pose next{spot.xy, wrap180(up + 90.0 * (r + 1))};
                    app.mover.carry(at, 0, next, 0);
                    prev = at;
                    at = next;
                }
            }
            app.mover.carry(at, 0, {app.tempSpot, up}, waiting);
            ++waiting;
        }
        while (waiting > 0) {
            --waiting;
            app.mover.carry({app.tempSpot, up}, waiting, {stack.xy, up}, stack.n);
            ++stack.n;
            st.save(app.stateFile);
        }
    }
    app.mover.park();
    log("collect: done");
}

}  // namespace spc
