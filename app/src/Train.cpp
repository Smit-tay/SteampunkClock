// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// train: fit the digit network on patch folders. 15% of the samples (chosen by a hash of the
// file name, so repeatable) are held out to report accuracy on unseen photos; with `final` the
// saved model is then refitted on everything.
// read: recognise the tile lying at a point, for checking the model on the real table.
#include "Commands.h"

#include "Dataset.h"
#include "DigitNet.h"
#include "Log.h"
#include "Vision.h"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <functional>
#include <string>

namespace spc {
namespace {

constexpr int kPcaDims = 80;
constexpr int kHidden = 64;
constexpr int kIterations = 300;

bool heldOut(const Sample& s) { return std::hash<std::string>{}(s.source) % 100 < 15; }

std::array<cv::Mat, 4> quarterTurns(const cv::Mat& upright)
{
    // patches[k]: the guess turned k quarter turns counter-clockwise, so the tile appears turned clockwise.
    std::array<cv::Mat, 4> p;
    p[0] = upright;
    cv::rotate(upright, p[1], cv::ROTATE_90_CLOCKWISE);
    cv::rotate(upright, p[2], cv::ROTATE_180);
    cv::rotate(upright, p[3], cv::ROTATE_90_COUNTERCLOCKWISE);
    return p;
}

}  // namespace

void train(App& app, const std::vector<std::filesystem::path>& dirs, int perSample, bool final)
{
    const auto all = loadSamples(dirs);
    std::vector<Sample> fit;
    std::vector<Sample> test;
    for (const auto& s : all) {
        (heldOut(s) ? test : fit).push_back(s);
    }
    int counts[DigitNet::kClasses] = {};
    for (const auto& s : all) {
        ++counts[s.label];
    }
    std::string summary;
    for (int d = 0; d < 10; ++d) {
        summary += std::to_string(d) + ":" + std::to_string(counts[d]) + " ";
    }
    log("train: ", all.size(), " samples (", summary, "junk:", counts[DigitNet::kJunk], "), ", test.size(),
        " held out");

    cv::Mat x;
    cv::Mat y;
    buildTrainingSet(fit, perSample, 1, x, y);
    DigitNet net;
    log("train: fitting ", x.rows, " rows");
    net.train(x, y, kPcaDims, kHidden, kIterations);

    int ok = 0;
    int tiles = 0;
    int junkOk = 0;
    int junk = 0;
    for (const auto& s : test) {
        if (s.label < 10) {
            const auto r = net.read(quarterTurns(s.patch));
            ++tiles;
            const bool upright = r.quarter == 0 || (r.quarter == 2 && DigitNet::halfTurnSymmetric(s.label));
            if (r.digit == s.label && upright) {
                ++ok;
            } else {
                log("  miss: ", s.source, " is ", s.label, ", read ", r.digit, " turn ", r.quarter, " score ", r.score);
            }
        } else {
            const cv::Mat sc = net.scores(DigitNet::features(s.patch));
            cv::Point mx;
            cv::minMaxLoc(sc, nullptr, nullptr, nullptr, &mx);
            ++junk;
            junkOk += mx.x == DigitNet::kJunk ? 1 : 0;
        }
    }
    log("train: held out tiles read right (digit and upright) ", ok, "/", tiles, ", junk recognised ", junkOk, "/",
        junk);
    if (final) {
        buildTrainingSet(all, perSample, 2, x, y);
        log("train: final fit on all ", x.rows, " rows");
        net.train(x, y, kPcaDims, kHidden, kIterations);
    }
    net.save(app.modelFile);
    log("train: saved ", app.modelFile.string());
}

void readTile(App& app, cv::Point2d xy)
{
    const cv::Mat img = app.cam.grab(Camera::Res::Hi);
    Pose pose{xy, app.upStack};
    if (const auto found = findLoneTile(img, app.geo, pose, 15.0)) {
        pose = *found;
        log("read: lone tile at (", pose.xy.x, ", ", pose.xy.y, ") up ", pose.up);
    }
    std::array<cv::Mat, 4> patches;
    const auto dir = app.dataDir / "debug";
    std::filesystem::create_directories(dir);
    for (int k = 0; k < 4; ++k) {
        patches[static_cast<size_t>(k)] = extractPatch(img, app.geo, {pose.xy, wrap180(pose.up + 90.0 * k)});
        cv::imwrite((dir / ("read_" + std::to_string(k) + ".png")).string(), patches[static_cast<size_t>(k)]);
    }
    if (!std::filesystem::exists(app.modelFile)) {
        log("read: no model yet; patches saved in ", dir.string());
        return;
    }
    DigitNet net;
    net.load(app.modelFile);
    const auto r = net.read(patches);
    log("read: digit ", r.digit, " quarter ", r.quarter, " (up ", wrap180(pose.up + 90.0 * r.quarter), "), score ",
        r.score, " margin ", r.margin, " junk ", r.junk);
}

}  // namespace spc
