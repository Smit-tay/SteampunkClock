// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// show: put HHMM on the display. Only changed slots are touched; for each, the old tile goes
// back onto a stack of its digit and the new one comes from the nearest stack holding it. A
// slot whose new digit has no tile left in the stacks waits until the others have returned theirs.
// One photo first gives where the display tiles and the stack tops really lie (picks there, not at
// the nominal spots); one photo after checks every slot.
#include "Commands.h"

#include "DigitNet.h"
#include "Heap.h"
#include "Log.h"
#include "State.h"

#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>

namespace spc {

void show(App& app, const std::string& time, const DigitNet* preloaded, const Progress& progress)
{
    auto say = [&](const std::string& m) {
        if (progress) {
            progress(m);
        }
    };
    // "clear": every display tile back onto the stacks.
    const std::string hhmm = time == "clear" ? "----" : time;
    auto wanted = [&](size_t i) { return hhmm[i] == '-' ? -1 : hhmm[i] - '0'; };
    if (hhmm != "----" && (hhmm.size() != 4 || hhmm.find_first_not_of("0123456789") != std::string::npos || hhmm.substr(0, 2) > "23" ||
        hhmm.substr(2, 2) > "59")) {
        throw std::invalid_argument("show: time must be HHMM, 0000..2359, or clear");
    }
    if (app.displaySlots.size() != 4) {
        throw std::runtime_error("show: layout.display_grid must list 4 slots");
    }
    const ArmLock lock(app.lockFile, 5.0);
    State st = State::load(app.stateFile);
    if (st.display.size() != 4) {
        st.display.assign(4, Slot{});
    }
    for (size_t i = 0; i < 4; ++i) {
        st.display[i].xy = app.displaySlots[i];
    }
    const double up = app.upStack;
    std::deque<size_t> todo;
    for (size_t i = 0; i < 4; ++i) {
        if (st.display[i].digit != wanted(i)) {
            todo.push_back(i);
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (todo.empty()) {
        log("show: ", hhmm, " already shown");
        return;
    }
    std::optional<DigitNet> own;
    const DigitNet* net = preloaded;
    if (net == nullptr && std::filesystem::exists(app.modelFile)) {
        own.emplace();
        own->load(app.modelFile);
        net = &*own;
    }
    // Where things really lie: display tiles and the stack tops this change will use.
    std::map<std::string, Pose> stackTop;  // key -> observed top pose, until that tile is taken
    std::vector<std::optional<Pose>> slotTile(4);
    if (net != nullptr) {
        std::vector<cv::Point2d> keep(app.displaySlots.begin(), app.displaySlots.end());
        for (const auto& sk : st.stacks) {
            keep.push_back(sk.xy);
        }
        say("looking at the tiles");
        app.mover.clearView(keep, 25.0);
        const cv::Mat img = app.cam.grab(Camera::Res::Hi);
        for (const size_t i : todo) {
            if (st.display[i].digit >= 0) {
                if (const auto t = locateTile(img, app.geo, st.display[i].xy, 15.0, net)) {
                    slotTile[i] = Pose{t->pose.xy, up};  // its own up only matters to within a few degrees
                    if (std::abs(wrap180(t->pose.up - up)) < 45.0) {
                        slotTile[i]->up = t->pose.up;
                    }
                }
            }
        }
        for (const auto& sk : st.stacks) {
            bool used = false;  // a digit this change takes
            for (const size_t i : todo) {
                used = used || wanted(i) == sk.digit;
            }
            if (sk.n > 0 && used) {
                if (const auto t = locateTile(img, app.geo, sk.xy, 12.0, net);
                    t && std::abs(wrap180(t->pose.up - up)) < 45.0) {
                    stackTop[sk.key] = t->pose;
                }
            }
        }
        log("show: looked in ", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), " s");
    }
    size_t stuck = 0;
    while (!todo.empty()) {
        const size_t i = todo.front();
        todo.pop_front();
        Slot& slot = st.display[i];
        const int want = wanted(i);
        const auto ts = std::chrono::steady_clock::now();
        say("tile " + std::to_string(i + 1) + " of 4" +
            (want >= 0 ? ": fetching a " + std::to_string(want) : ": clearing"));
        if (slot.digit >= 0) {
            Stack* back = st.putOn(slot.digit, app.maxStack, slot.xy);
            if (back == nullptr) {
                throw std::runtime_error("show: no stack of " + std::to_string(slot.digit) + " has room");
            }
            app.mover.carry(slotTile[i].value_or(Pose{slot.xy, up}), 0, {back->xy, up}, back->n);
            stackTop[back->key] = Pose{back->xy, up};  // just put there, to within ~2 mm
            ++back->n;
            slot.digit = -1;
            st.save(app.stateFile);
        }
        if (want < 0) {
            continue;
        }
        Stack* from = st.takeFrom(want, slot.xy);
        if (from == nullptr) {
            // Every tile of this digit is on the display; others may give one back first.
            if (++stuck > todo.size() + 1) {
                throw std::runtime_error("show: no tile of " + std::to_string(want) + " left");
            }
            todo.push_back(i);
            continue;
        }
        stuck = 0;
        if (net != nullptr && stackTop.find(from->key) == stackTop.end()) {
            // A top not seen yet (the tile under one just taken): look before picking; lower
            // tiles of a stack can lie several mm off its nominal spot.
            app.mover.clearView({from->xy}, 25.0);
            if (const auto t = locateTile(app.cam.grab(Camera::Res::Hi), app.geo, from->xy, 15.0, net);
                t && std::abs(wrap180(t->pose.up - up)) < 45.0) {
                stackTop[from->key] = t->pose;
            }
        }
        const auto seen = stackTop.find(from->key);
        app.mover.carry(seen != stackTop.end() ? seen->second : Pose{from->xy, up}, from->n - 1, {slot.xy, up}, 0);
        stackTop.erase(from->key);
        --from->n;
        slot.digit = want;
        st.save(app.stateFile);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        log("show: slot ", i + 1, " -> ", want, " in ", s, " s");
    }
    st.shown = hhmm == "----" ? "" : hhmm;
    st.save(app.stateFile);
    app.mover.clearView(app.displaySlots, 25.0);  // visitors watch the display through the camera
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    log("show: ", hhmm, " shown, ", total, " s");
    // Check what the camera sees on every slot.
    if (net != nullptr) {
        say("checking the display");
        const cv::Mat img = app.cam.grab(Camera::Res::Hi);
        std::string bad;
        for (size_t i = 0; i < 4; ++i) {
            const int want = wanted(i);
            const auto t = locateTile(img, app.geo, st.display[i].xy, 15.0, net);
            if (want < 0) {
                if (t) {
                    bad += " slot " + std::to_string(i + 1) + " not empty;";
                }
                continue;
            }
            if (!t) {
                bad += " slot " + std::to_string(i + 1) + " empty;";
                continue;
            }
            const auto& r = t->reading;
            const bool upright = std::abs(wrap180(t->pose.up - up)) < 45.0 ||
                                 (DigitNet::halfTurnSymmetric(r.digit) && std::abs(wrap180(t->pose.up - up)) > 135.0);
            log("show: slot ", i + 1, " reads \"", r.digit, "\" ", cv::norm(t->pose.xy - st.display[i].xy), " mm and ",
                wrap180(t->pose.up - up), " deg off");
            if (r.digit != want || !upright) {
                bad += " slot " + std::to_string(i + 1) + " reads " + std::to_string(r.digit) + ";";
            }
        }
        if (!bad.empty()) {
            throw std::runtime_error("show: check failed:" + bad);
        }
    }
}

}  // namespace spc
