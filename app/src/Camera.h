// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// Camera: the overhead phone (IP Webcam app) over HTTP. /shot.jpg returns one JPEG; the
// resolution is switched with /settings/video_size and needs a few seconds to settle.
#pragma once

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace spc {

class CameraError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// libcurl must be initialised once per process (curl_global_init); this does it.
class CurlGlobal {
public:
    CurlGlobal();
    ~CurlGlobal();
    CurlGlobal(const CurlGlobal&) = delete;
    CurlGlobal& operator=(const CurlGlobal&) = delete;
    CurlGlobal(CurlGlobal&&) = delete;
    CurlGlobal& operator=(CurlGlobal&&) = delete;
};

class Camera {
public:
    enum class Res { Hi, Lo };

    // cfg.auth_file: "user:password" for the phone, kept out of the config (and the repo);
    // a relative path is relative to `base`, the config's directory.
    Camera(const nlohmann::json& cfg, const std::filesystem::path& base);

    // One photo at the given resolution. Hi: image_size of the calibration (1600x1200).
    [[nodiscard]] cv::Mat grab(Res res);

private:
    [[nodiscard]] std::string get(const std::string& path, long timeoutS) const;
    void setRes(Res res);

    std::string url_;
    std::string userpwd_;
    std::string hiRes_;
    std::string loRes_;
    double settleS_ = 3.0;
    int current_ = -1;  // -1 unknown, else static_cast<int>(Res)
};

}  // namespace spc
