// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// spclock: the SteampunkClock. A uArm Swift Pro, driven through topicfs, sorts handwritten digit
// tiles into stacks and shows the time with them; an overhead camera and a small neural network
// (trained offline on the tiles themselves) find and read the tiles. Everything runs locally.
#include "App.h"
#include "Camera.h"
#include "Commands.h"
#include "Dataset.h"
#include "DigitNet.h"
#include "Heap.h"
#include "Log.h"
#include "State.h"
#include "Vision.h"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kUsage = R"(usage: spclock [-c CONFIG] COMMAND [ARGS]
  status                    print the stacks and the display
  photo FILE                save a full-resolution photo
  read X Y                  read the tile lying near robot point (X, Y)
  show HHMM|clear           put the time HHMM on the display, or empty it
  serve [PORT]              web service on 127.0.0.1:PORT (default 8765): status, update to now
  sort                      clear the heap into the digit stacks
  collect [DIGITS]          photograph the stored tiles for training (default: all digits)
  train [--per N] [--final] DIR...   train the digit network on patch folders
  import-panels OUT DIR...  turn the 2026-11-15 review panels into training patches
  fetch X Y KEY [UPOFF]     put the lone tile near (X, Y) on stack KEY; UPOFF: its digit top, degrees
                            from the stored orientation (default 0)
  carry X Y LEVEL KEY        blind move: the upright tile at level LEVEL of the pile at (X, Y) onto stack KEY
  locate X Y                find and read the whole tile nearest robot point (X, Y) on a new photo
  heap IN OUT               find liftable tiles on photo IN (or "-": a new photo), draw them to OUT
  drop KEY X Y UP LEVEL     put the top tile of stack KEY at (X, Y), digit top UP (robot deg), LEVEL
                            tiles above the table (test heaps)
  calplace KEY              measure placement errors with the top tile of stack KEY (spot, display
                            slots, one free point; correction off) and print place_corrections
  init-state FILE           create the state from the Python sorter's tile_state.json
CONFIG defaults to spclock.json in the current directory.)";

int initState(spc::App& app, const std::string& file)
{
    std::ifstream f(file);
    const auto j = nlohmann::json::parse(f);
    spc::State st;
    for (const auto& [key, cell] : j.at("cells").items()) {
        st.stacks.push_back({key, key[0] - '0', {cell.at("xy").at(0).get<double>(), cell.at("xy").at(1).get<double>()},
                             cell.at("n").get<int>()});
    }
    st.save(app.stateFile);
    spc::log("init-state: ", st.stacks.size(), " stacks written to ", app.stateFile.string());
    return 0;
}

int fetch(spc::App& app, cv::Point2d near, const std::string& key, double upOffset)
{
    const spc::ArmLock lock(app.lockFile, 5.0);
    auto st = spc::State::load(app.stateFile);
    spc::Stack* dst = nullptr;
    for (auto& s : st.stacks) {
        if (s.key == key) {
            dst = &s;
        }
    }
    if (dst == nullptr || dst->n >= app.maxStack) {
        throw std::runtime_error("fetch: no stack " + key + " with room");
    }
    app.mover.clearView({near}, 25.0);
    spc::DigitNet net;
    net.load(app.modelFile);
    const auto found = spc::locateTile(app.cam.grab(spc::Camera::Res::Hi), app.geo, near, 25.0, &net);
    if (!found) {
        throw std::runtime_error("fetch: no whole tile near the point");
    }
    // Its digit's top: from the reading when sure, else the long axis nearest the stored
    // orientation plus upOffset.
    std::optional<spc::Pose> tile = found->pose;
    if (found->reading.digit < 0 || found->reading.margin < 0.5) {
        const double guess = spc::wrap180(app.upStack + upOffset);
        tile->up = std::abs(spc::wrap180(found->pose.up - guess)) <= 90.0 ? found->pose.up
                                                                         : spc::wrap180(found->pose.up + 180.0);
        double best = 1e9;
        for (int q = 0; q < 4; ++q) {  // nearest quarter turn of the found axis to the guess
            const double u = spc::wrap180(found->pose.up + 90.0 * q);
            if (std::abs(spc::wrap180(u - guess)) < best) {
                best = std::abs(spc::wrap180(u - guess));
                tile->up = u;
            }
        }
    }
    spc::log("fetch: found \"", found->reading.digit, "\" (margin ", found->reading.margin, ") up ", tile->up);
    // The photo gives the long axis; which end is the digit's top comes from upOffset.
    app.mover.carry(*tile, 0, {dst->xy, app.upStack}, dst->n);
    ++dst->n;
    st.save(app.stateFile);
    spc::log("fetch: tile from (", tile->xy.x, ", ", tile->xy.y, ") onto stack ", key, ", now ", dst->n);
    return 0;
}

int carryTo(spc::App& app, cv::Point2d from, int level, const std::string& key)
{
    const spc::ArmLock lock(app.lockFile, 5.0);
    auto st = spc::State::load(app.stateFile);
    spc::Stack* dst = nullptr;
    for (auto& s : st.stacks) {
        if (s.key == key) {
            dst = &s;
        }
    }
    if (dst == nullptr || dst->n >= app.maxStack) {
        throw std::runtime_error("carry: no stack " + key + " with room");
    }
    app.mover.carry({from, app.upStack}, level, {dst->xy, app.upStack}, dst->n);
    ++dst->n;
    st.save(app.stateFile);
    spc::log("carry: tile from (", from.x, ", ", from.y, ") level ", level, " onto stack ", key, ", now ", dst->n);
    return 0;
}

// Stacks, display slots and the temporary spot: heap vision ignores tiles there.
spc::HeapRegion heapRegion(const spc::App& app)
{
    spc::HeapRegion region;
    const auto st = spc::State::load(app.stateFile);
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

int heap(spc::App& app, const std::string& in, const std::string& out)
{
    std::optional<spc::ArmLock> lock;
    if (in == "-") {
        lock.emplace(app.lockFile, 5.0);
        app.mover.park();  // a view of the whole table
    }
    const cv::Mat img = in == "-" ? app.cam.grab(spc::Camera::Res::Hi) : cv::imread(in, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("heap: cannot read " + in);
    }
    spc::DigitNet net;
    if (std::filesystem::exists(app.modelFile)) {
        net.load(app.modelFile);
    }
    const auto t0 = std::chrono::steady_clock::now();
    // A saved photo may show another layout: no exclusions then.
    const auto tiles = spc::findHeapTiles(img, app.geo, in == "-" ? heapRegion(app) : spc::HeapRegion{}, &net);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& t = tiles[i];
        spc::log("heap: ", i, " at (", t.pose.xy.x, ", ", t.pose.xy.y, ") up ", t.pose.up, " border ", t.border,
                 " fill ", t.fill, t.lone ? " lone" : "", " digit ", t.reading.digit, " margin ", t.reading.margin,
                 " score ", t.score);
    }
    spc::log("heap: ", tiles.size(), " candidates in ", ms, " ms");
    cv::imwrite(out, spc::drawHeapTiles(img, app.geo, tiles));
    return 0;
}

int drop(spc::App& app, const std::string& key, spc::Pose at, int level)
{
    const spc::ArmLock lock(app.lockFile, 5.0);
    auto st = spc::State::load(app.stateFile);
    for (auto& s : st.stacks) {
        if (s.key == key && s.n > 0) {
            app.mover.carry({s.xy, app.upStack}, s.n - 1, at, level);
            --s.n;
            st.save(app.stateFile);
            spc::log("drop: tile from stack ", key, " to (", at.xy.x, ", ", at.xy.y, ") up ", at.up, " level ", level);
            return 0;
        }
    }
    throw std::runtime_error("drop: stack " + key + " is empty or unknown");
}

int calplace(spc::App& app, const std::string& key)
{
    const spc::ArmLock lock(app.lockFile, 5.0);
    spc::DigitNet net;
    net.load(app.modelFile);
    auto st = spc::State::load(app.stateFile);
    spc::Stack* stack = nullptr;
    for (auto& s : st.stacks) {
        if (s.key == key && s.n > 0) {
            stack = &s;
        }
    }
    if (stack == nullptr) {
        throw std::runtime_error("calplace: stack " + key + " is empty or unknown");
    }
    for (const auto& d : st.display) {
        if (d.digit >= 0) {
            throw std::runtime_error("calplace: the display must be empty");
        }
    }
    std::vector<cv::Point2d> targets{app.photoSpot};
    targets.insert(targets.end(), app.displaySlots.begin(), app.displaySlots.end());
    targets.emplace_back(205.0, -70.0);
    app.mover.setCorrection(false);
    // Start on the photo spot.
    app.mover.carry({stack->xy, app.upStack}, stack->n - 1, {app.photoSpot, app.upStack}, 0);
    --stack->n;
    st.save(app.stateFile);
    cv::Point2d last = app.photoSpot;
    std::vector<std::vector<cv::Point2d>> errs(targets.size());
    for (int round = 0; round < 2; ++round) {
        for (size_t i = 0; i < targets.size(); ++i) {
            const size_t k = (i + 1) % targets.size();  // spot last in each round
            app.mover.clearView({last}, 25.0);
            const auto here = spc::locateTile(app.cam.grab(spc::Camera::Res::Hi), app.geo, last, 20.0, &net);
            if (!here) {
                throw std::runtime_error("calplace: lost the tile");
            }
            const spc::Pose src = spc::nearerTop(here->pose, app.upStack,
                                                 spc::DigitNet::halfTurnSymmetric(here->reading.digit));
            app.mover.carry(src, 0, {targets[k], app.upStack}, 0);
            app.mover.clearView({targets[k]}, 25.0);
            const auto there = spc::locateTile(app.cam.grab(spc::Camera::Res::Hi), app.geo, targets[k], 20.0, &net);
            if (!there) {
                throw std::runtime_error("calplace: lost the tile after placing");
            }
            const cv::Point2d e = there->pose.xy - targets[k];
            errs[k].push_back(e);
            spc::log("calplace: (", targets[k].x, ", ", targets[k].y, ") landed (", e.x, ", ", e.y, ") off, up ",
                     spc::wrap180(there->pose.up - app.upStack), " deg");
            last = there->pose.xy;
        }
    }
    // Back on the stack, from where it lies now (the photo spot).
    app.mover.clearView({last}, 25.0);
    const auto here = spc::locateTile(app.cam.grab(spc::Camera::Res::Hi), app.geo, last, 20.0, &net);
    app.mover.setCorrection(true);
    app.mover.carry(here ? spc::nearerTop(here->pose, app.upStack, spc::DigitNet::halfTurnSymmetric(here->reading.digit))
                         : spc::Pose{last, app.upStack},
                    0, {stack->xy, app.upStack}, stack->n);
    ++stack->n;
    st.save(app.stateFile);
    std::cout << "\"place_corrections\": [";
    for (size_t k = 0; k < targets.size(); ++k) {
        cv::Point2d m(0.0, 0.0);
        for (const auto& e : errs[k]) {
            m += e;
        }
        m /= static_cast<double>(std::max<size_t>(1, errs[k].size()));
        std::cout << (k ? ", " : "") << "[" << targets[k].x << ", " << targets[k].y << ", " << -m.x << ", " << -m.y
                  << "]";
    }
    std::cout << "]\n";
    return 0;
}

int status(spc::App& app)
{
    const auto st = spc::State::load(app.stateFile);
    for (const auto& s : st.stacks) {
        std::cout << "stack " << s.key << ": " << s.n << " x \"" << s.digit << "\" at (" << s.xy.x << ", " << s.xy.y
                  << ")\n";
    }
    for (size_t i = 0; i < st.display.size(); ++i) {
        std::cout << "slot " << i + 1 << ": " << (st.display[i].digit < 0 ? std::string("empty")
                                                                          : std::to_string(st.display[i].digit))
                  << '\n';
    }
    std::cout << "temp spot: (" << app.tempSpot.x << ", " << app.tempSpot.y << ")\n";
    std::cout << "shown: " << (st.shown.empty() ? "nothing" : st.shown) << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> args(argv + 1, argv + argc);
    std::string config = "spclock.json";
    if (args.size() >= 2 && args[0] == "-c") {
        config = args[1];
        args.erase(args.begin(), args.begin() + 2);
    }
    if (args.empty()) {
        std::cerr << kUsage << '\n';
        return 2;
    }
    try {
        const spc::CurlGlobal curl;
        spc::App app(config);
        const std::string& cmd = args[0];
        if (cmd == "status") {
            return status(app);
        }
        if (cmd == "photo" && args.size() == 2) {
            cv::imwrite(args[1], app.cam.grab(spc::Camera::Res::Hi));
            return 0;
        }
        if (cmd == "read" && args.size() == 3) {
            spc::readTile(app, {std::stod(args[1]), std::stod(args[2])});
            return 0;
        }
        if (cmd == "show" && args.size() == 2) {
            spc::show(app, args[1]);
            return 0;
        }
        if (cmd == "sort" && args.size() == 1) {
            spc::sort(app);
            return 0;
        }
        if (cmd == "serve" && args.size() <= 2) {
            spc::serve(app, args.size() == 2 ? std::stoi(args[1]) : 8765);
            return 0;
        }
        if (cmd == "collect") {
            spc::collect(app, args.size() >= 2 ? args[1] : "0123456789");
            return 0;
        }
        if (cmd == "train") {
            int per = 30;
            bool final = false;
            std::vector<std::filesystem::path> dirs;
            for (size_t i = 1; i < args.size(); ++i) {
                if (args[i] == "--per" && i + 1 < args.size()) {
                    per = std::stoi(args[++i]);
                } else if (args[i] == "--final") {
                    final = true;
                } else {
                    dirs.emplace_back(args[i]);
                }
            }
            spc::train(app, dirs, per, final);
            return 0;
        }
        if (cmd == "import-panels" && args.size() >= 3) {
            const std::vector<std::filesystem::path> dirs(args.begin() + 2, args.end());
            spc::log("import-panels: ", spc::importPanels(dirs, args[1]), " tiles");
            return 0;
        }
        if (cmd == "fetch" && (args.size() == 4 || args.size() == 5)) {
            return fetch(app, {std::stod(args[1]), std::stod(args[2])}, args[3],
                         args.size() == 5 ? std::stod(args[4]) : 0.0);
        }
        if (cmd == "carry" && args.size() == 5) {
            return carryTo(app, {std::stod(args[1]), std::stod(args[2])}, std::stoi(args[3]), args[4]);
        }
        if (cmd == "lone" && args.size() == 5) {  // debug: findLoneTile on a saved photo
            const auto t = spc::findLoneTile(cv::imread(args[1]), app.geo,
                                             {{std::stod(args[2]), std::stod(args[3])}, std::stod(args[4])}, 20.0);
            spc::log("lone: ", t ? "found" : "not found");
            return t ? 0 : 1;
        }
        if (cmd == "locate" && args.size() == 3) {
            spc::DigitNet net;
            net.load(app.modelFile);
            const cv::Point2d p(std::stod(args[1]), std::stod(args[2]));
            const spc::ArmLock lock(app.lockFile, 5.0);
            app.mover.clearView({p}, 25.0);
            const auto t = spc::locateTile(app.cam.grab(spc::Camera::Res::Hi), app.geo, p, 30.0, &net);
            if (!t) {
                spc::log("locate: no whole tile within 30 mm");
                return 1;
            }
            spc::log("locate: tile at (", t->pose.xy.x, ", ", t->pose.xy.y, ") up ", t->pose.up, " border ", t->border,
                     " digit ", t->reading.digit, " margin ", t->reading.margin);
            return 0;
        }
        if (cmd == "goto" && args.size() == 4) {  // debug: move the cup (pump off, wrist straight)
            const spc::ArmLock lock(app.lockFile, 5.0);
            app.arm.pump(false);
            app.arm.wrist(90.0);
            app.arm.move({std::stod(args[1]), std::stod(args[2])}, std::stod(args[3]), 100.0);
            return 0;
        }
        if (cmd == "gridview" && args.size() == 2) {  // debug: grid lines (grid mm) and display slots
            const spc::ArmLock lock(app.lockFile, 5.0);
            app.mover.park();
            cv::Mat img = app.cam.grab(spc::Camera::Res::Hi);
            auto px = [&](double gx, double gy) {
                const cv::Point2d p = app.geo.robotToImage(app.geo.gridToRobot({gx, gy}));
                return cv::Point(static_cast<int>(p.x), static_cast<int>(p.y));
            };
            for (int v = -200; v <= 400; v += 100) {
                cv::line(img, px(v, -300), px(v, 400), cv::Scalar(255, 0, 0), 1);
                cv::line(img, px(-100, v), px(400, v), cv::Scalar(0, 0, 255), 1);
                cv::putText(img, "x" + std::to_string(v), px(v, 0), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
                cv::putText(img, "y" + std::to_string(v), px(100, v), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
            }
            for (const auto& d : app.displaySlots) {
                const cv::Point2d p = app.geo.robotToImage(d);
                cv::circle(img, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), 8, cv::Scalar(255, 0, 255), 3);
            }
            cv::imwrite(args[1], img);
            return 0;
        }
        if (cmd == "armview") {  // debug: where the arm hides the table, robot frame, 10 mm cells
            const cv::Mat img = app.cam.grab(spc::Camera::Res::Hi);
            cv::Mat grey;
            cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
            const auto at = app.arm.position();
            std::cout << "arm at " << at.x << " " << at.y << " " << at.z << "\n     y:";
            for (int y = 250; y >= -250; y -= 10) {
                std::cout << (y % 50 == 0 ? '|' : ' ');
            }
            std::cout << '\n';
            for (int x = 330; x >= 40; x -= 10) {
                std::cout << "x " << x << (x < 100 ? "  " : " ");
                for (int y = 250; y >= -250; y -= 10) {
                    const cv::Point2d px = app.geo.robotToImage({static_cast<double>(x), static_cast<double>(y)});
                    char c = ' ';
                    if (px.x >= 0 && px.y >= 0 && px.x < grey.cols && px.y < grey.rows) {
                        c = grey.at<uchar>(static_cast<int>(px.y), static_cast<int>(px.x)) < 60 ? '#' : '.';
                    }
                    if (std::hypot(x - at.x, y - at.y) < 7) {
                        c = 'C';
                    }
                    std::cout << c;
                }
                std::cout << '\n';
            }
            return 0;
        }
        if (cmd == "heap" && args.size() == 3) {
            return heap(app, args[1], args[2]);
        }
        if (cmd == "drop" && args.size() == 6) {
            return drop(app, args[1], {{std::stod(args[2]), std::stod(args[3])}, std::stod(args[4])},
                        std::stoi(args[5]));
        }
        if (cmd == "calplace" && args.size() == 2) {
            return calplace(app, args[1]);
        }
        if (cmd == "init-state" && args.size() == 2) {
            return initState(app, args[1]);
        }
        std::cerr << kUsage << '\n';
        return 2;
    } catch (const std::exception& e) {
        spc::log("error: ", e.what());
        return 1;
    }
}
