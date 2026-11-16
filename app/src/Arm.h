// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Arm: the uArm Swift Pro through topicfs (ROS2 services and topics as files).
// A service call writes JSON to <service>/command (the write blocks until the call returns,
// at most 10 s) and reads <service>/response, which is checked for success.
#pragma once

#include "Geometry.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace spc {

class ArmError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Exclusive use of the arm, shared with the movement website (flock on its lock file).
class ArmLock {
public:
    ArmLock(const std::filesystem::path& file, double waitS);
    ~ArmLock();
    ArmLock(const ArmLock&) = delete;
    ArmLock& operator=(const ArmLock&) = delete;
    ArmLock(ArmLock&&) = delete;
    ArmLock& operator=(ArmLock&&) = delete;

private:
    int fd_ = -1;
};

class Arm {
public:
    Arm(std::filesystem::path topicfs, const Geometry& geo);

    // Move the CUP to (x, y, z) in the robot frame; the command is corrected for the cup's
    // offset from the wrist axis at the current wrist angle. Returns when the arm has arrived.
    void move(cv::Point2d cup, double z, double speed);
    void moveRaw(cv::Point3d cmd, double speed);
    void pump(bool on);
    // Wrist servo angle 0..180 (90 = straight). Waits for the servo to turn.
    void wrist(double deg);
    [[nodiscard]] double wristAngle() const { return wrist_; }

    // 0 off, 1 sealed, 2 running without a seal. Unreliable for a held tile (2026-11-15).
    [[nodiscard]] int pumpStatus() const;
    [[nodiscard]] cv::Point3d position() const;

private:
    std::string call(const std::string& service, const std::string& payload) const;
    [[nodiscard]] std::string readTopic(const std::string& topic) const;

    std::filesystem::path root_;
    const Geometry& geo_;
    double wrist_ = 90.0;
};

}  // namespace spc
