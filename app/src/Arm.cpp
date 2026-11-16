// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "Arm.h"

#include "Log.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <thread>

namespace spc {
namespace {

using namespace std::chrono_literals;

std::string fmt(double v)
{
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", v);
    return buf;
}

}  // namespace

ArmLock::ArmLock(const std::filesystem::path& file, double waitS)
{
    fd_ = ::open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (fd_ < 0) {
        throw ArmError("cannot open lock file " + file.string() + ": " + std::strerror(errno));
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(waitS);
    while (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            ::close(fd_);
            fd_ = -1;
            throw ArmError("arm is busy (lock " + file.string() + " held)");
        }
        std::this_thread::sleep_for(100ms);
    }
}

ArmLock::~ArmLock()
{
    if (fd_ >= 0) {
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
    }
}

Arm::Arm(std::filesystem::path topicfs, const Geometry& geo) : root_(std::move(topicfs)), geo_(geo) {}

std::string Arm::call(const std::string& service, const std::string& payload) const
{
    const auto dir = root_ / service;
    const auto t0 = std::chrono::steady_clock::now();
    {
        // The write blocks until the ROS2 call has returned (or timed out after 10 s).
        std::ofstream cmd(dir / "command", std::ios::binary);
        if (!cmd) {
            throw ArmError("cannot open " + (dir / "command").string());
        }
        cmd << payload;
        cmd.flush();
        if (!cmd) {
            throw ArmError("write to " + service + " failed");
        }
    }
    std::ifstream resp(dir / "response", std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(resp)), std::istreambuf_iterator<char>());
    const auto j = nlohmann::json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.value("success", false)) {
        throw ArmError(service + " " + payload + " failed: " + text);
    }
    if (std::getenv("SPC_TIMING") != nullptr) {
        log("arm: ", service, " ", payload, " ",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), " s");
    }
    return text;
}

std::string Arm::readTopic(const std::string& topic) const
{
    std::ifstream f(root_ / topic / "latest", std::ios::binary);
    if (!f) {
        throw ArmError("cannot read topic " + topic);
    }
    return {(std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()};
}

void Arm::move(cv::Point2d cup, double z, double speed)
{
    const cv::Point2d c = geo_.cupToCommand(cup, wrist_);
    moveRaw({c.x, c.y, z}, speed);
}

void Arm::moveRaw(cv::Point3d cmd, double speed)
{
    const std::string payload = "{\"x\":" + fmt(cmd.x) + ",\"y\":" + fmt(cmd.y) + ",\"z\":" + fmt(cmd.z) +
                                ",\"speed\":" + fmt(speed) + "}";
    try {
        try {
            call("move_to", payload);
        } catch (const ArmError& e) {
            // Seen once (2026-11-15): E25 on a lift that succeeded when sent again. One retry.
            log("arm: ", e.what(), "; retrying once");
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            call("move_to", payload);
        }
    } catch (const ArmError&) {
        try {
            pump(false);
        } catch (const ArmError&) {
            // already failing; report the move
        }
        throw;
    }
}

void Arm::pump(bool on) { call("set_pump", on ? R"({"on":true})" : R"({"on":false})"); }

void Arm::wrist(double deg)
{
    deg = std::clamp(deg, 0.0, 180.0);
    const double turn = std::abs(deg - wrist_);
    call("set_wrist", "{\"angle\":" + fmt(deg) + ",\"speed\":30.0}");
    wrist_ = deg;
    // The service returns at once; the servo needs ~0.2 s per 60 degrees.
    std::this_thread::sleep_for(std::chrono::duration<double>(0.05 + turn / 300.0));
}

int Arm::pumpStatus() const
{
    const auto j = nlohmann::json::parse(readTopic("pump_status"), nullptr, false);
    return j.is_discarded() ? -1 : j.value("data", -1);
}

cv::Point3d Arm::position() const
{
    const auto j = nlohmann::json::parse(readTopic("position"), nullptr, false);
    if (j.is_discarded()) {
        throw ArmError("bad position topic");
    }
    return {j.value("x", 0.0), j.value("y", 0.0), j.value("z", 0.0)};
}

}  // namespace spc
