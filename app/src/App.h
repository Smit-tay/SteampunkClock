// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// App: configuration and the objects every command needs, built from spclock.json.
// Relative paths in the config are relative to the config file's directory.
#pragma once

#include "Arm.h"
#include "Camera.h"
#include "Geometry.h"
#include "Mover.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <vector>

namespace spc {

class App {
public:
    explicit App(const std::filesystem::path& configFile);
    App(const App&) = delete;
    App& operator=(const App&) = delete;
    App(App&&) = delete;
    App& operator=(App&&) = delete;
    ~App() = default;

    nlohmann::json cfg;
    std::filesystem::path base;
    std::filesystem::path stateFile;
    std::filesystem::path modelFile;
    std::filesystem::path dataDir;
    std::filesystem::path lockFile;
    Geometry geo;
    Camera cam;
    Arm arm;
    Mover mover;
    int maxStack = 4;
    double upStack = 0.0;  // robot-frame direction of a stored or displayed digit's top
    cv::Point2d photoSpot;
    cv::Point2d tempSpot;
    std::vector<cv::Point2d> displaySlots;  // H1 H2 M1 M2, robot frame

    // Robot point of the keypad cell k (0..11).
    [[nodiscard]] cv::Point2d keypadCell(int k) const;
};

}  // namespace spc
