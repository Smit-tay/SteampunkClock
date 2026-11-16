// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "State.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace spc {
namespace {

cv::Point2d point(const nlohmann::json& j) { return {j.at(0).get<double>(), j.at(1).get<double>()}; }
nlohmann::json json(cv::Point2d p) { return {std::round(p.x * 100.0) / 100.0, std::round(p.y * 100.0) / 100.0}; }
double dist(cv::Point2d a, cv::Point2d b) { return std::hypot(a.x - b.x, a.y - b.y); }

}  // namespace

State State::load(const std::filesystem::path& file)
{
    std::ifstream f(file);
    if (!f) {
        throw std::runtime_error("cannot read state " + file.string());
    }
    const auto j = nlohmann::json::parse(f);
    State s;
    for (const auto& st : j.at("stacks")) {
        s.stacks.push_back({st.at("key").get<std::string>(), st.at("digit").get<int>(), point(st.at("xy")),
                            st.at("n").get<int>()});
    }
    for (const auto& sl : j.value("display", nlohmann::json::array())) {
        s.display.push_back({point(sl.at("xy")), sl.value("digit", -1)});
    }
    s.shown = j.value("shown", std::string{});
    return s;
}

void State::save(const std::filesystem::path& file) const
{
    nlohmann::json j;
    j["_comment"] = "spclock state: stacks of one digit, display slots H1 H2 M1 M2 (digit -1 = empty), time shown";
    j["stacks"] = nlohmann::json::array();
    for (const auto& st : stacks) {
        j["stacks"].push_back({{"key", st.key}, {"digit", st.digit}, {"xy", json(st.xy)}, {"n", st.n}});
    }
    j["display"] = nlohmann::json::array();
    for (const auto& sl : display) {
        j["display"].push_back({{"xy", json(sl.xy)}, {"digit", sl.digit}});
    }
    j["shown"] = shown;
    const auto tmp = std::filesystem::path(file).concat(".tmp");
    {
        std::ofstream f(tmp);
        f << j.dump(1) << '\n';
        if (!f) {
            throw std::runtime_error("cannot write " + tmp.string());
        }
    }
    std::filesystem::rename(tmp, file);
}

Stack* State::takeFrom(int digit, cv::Point2d near)
{
    Stack* best = nullptr;
    double bestD = std::numeric_limits<double>::max();
    for (auto& st : stacks) {
        if (st.digit == digit && st.n > 0 && dist(st.xy, near) < bestD) {
            best = &st;
            bestD = dist(st.xy, near);
        }
    }
    return best;
}

Stack* State::putOn(int digit, int maxStack, cv::Point2d near)
{
    Stack* best = nullptr;
    double bestD = std::numeric_limits<double>::max();
    for (auto& st : stacks) {
        if (st.digit == digit && st.n < maxStack && dist(st.xy, near) < bestD) {
            best = &st;
            bestD = dist(st.xy, near);
        }
    }
    return best;
}

int State::count(int digit) const
{
    int n = 0;
    for (const auto& st : stacks) {
        if (st.digit == digit) {
            n += st.n;
        }
    }
    return n;
}

}  // namespace spc
