// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Commands: the things spclock does (see main.cpp for the command line).
#pragma once

#include "App.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace spc {

class DigitNet;

// Photograph every stored tile at four orientations on the photo spot and save training patches.
void collect(App& app, const std::string& digits);

// Train the digit network on patch folders, report held-out accuracy, save the model.
void train(App& app, const std::vector<std::filesystem::path>& dirs, int perSample, bool final);

// Read the tile lying near (x, y): digit, orientation, scores; saves its patches under data/debug.
void readTile(App& app, cv::Point2d xy);

// Show the time HHMM: swap only the changed digits (old tile back to its stack, new one in).
// Progress messages for a watcher (the web service); may be empty.
using Progress = std::function<void(const std::string&)>;

// Put HHMM on the display ("clear": empty it). preloaded: the digit network, or null to load it.
void show(App& app, const std::string& hhmm, const DigitNet* preloaded = nullptr, const Progress& progress = {});

// The web service: line-JSON requests on 127.0.0.1:port (see Serve.cpp).
void serve(App& app, int port);

// Clear the heap into the digit stacks (needs a trained model).
void sort(App& app);

}  // namespace spc
