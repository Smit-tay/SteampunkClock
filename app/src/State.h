// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// State: where the tiles are. Stacks of one digit each (max_stack high), the four display
// slots (one tile each) and the time last shown. Saved after every move (atomic rename).
#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace spc {

struct Stack {
    std::string key;  // "0".."9", "3b", ... (a digit may have several stacks)
    int digit = -1;
    cv::Point2d xy;   // robot frame
    int n = 0;        // tiles in it
};

struct Slot {
    cv::Point2d xy;   // robot frame
    int digit = -1;   // -1: empty
};

class State {
public:
    static State load(const std::filesystem::path& file);
    void save(const std::filesystem::path& file) const;

    std::vector<Stack> stacks;
    std::vector<Slot> display;  // H1 H2 M1 M2
    std::string shown;          // "HHMM" or empty

    // The stack of `digit` nearest to `near` that holds a tile / has room; nullptr if none.
    [[nodiscard]] Stack* takeFrom(int digit, cv::Point2d near);
    [[nodiscard]] Stack* putOn(int digit, int maxStack, cv::Point2d near);
    [[nodiscard]] int count(int digit) const;  // in stacks
};

}  // namespace spc
