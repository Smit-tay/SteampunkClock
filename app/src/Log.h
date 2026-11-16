// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Log: one timestamped line to stdout.
#pragma once

#include <chrono>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>

namespace spc {

template <typename... Args>
void log(const Args&... args)
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%H:%M:%S", &tm);
    std::ostringstream os;
    os << stamp << '.' << (ms < 100 ? (ms < 10 ? "00" : "0") : "") << ms << ' ';
    (os << ... << args);
    std::cout << os.str() << std::endl;
}

}  // namespace spc
