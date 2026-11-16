// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "Camera.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

namespace spc {
namespace {

size_t collect(char* data, size_t size, size_t n, void* user)
{
    auto* out = static_cast<std::string*>(user);
    out->append(data, size * n);
    return size * n;
}

struct CurlDeleter {
    void operator()(CURL* c) const { curl_easy_cleanup(c); }
};

}  // namespace

CurlGlobal::CurlGlobal()
{
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        throw CameraError("curl_global_init failed");
    }
}

CurlGlobal::~CurlGlobal() { curl_global_cleanup(); }

namespace {

std::string readAuth(const std::filesystem::path& file)
{
    std::ifstream f(file);
    std::string line;
    if (!f || !std::getline(f, line) || line.find(':') == std::string::npos) {
        throw CameraError("cannot read camera user:password from " + file.string());
    }
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    return line;
}

}  // namespace

Camera::Camera(const nlohmann::json& cfg, const std::filesystem::path& base)
    : url_(cfg.at("url").get<std::string>()),
      userpwd_(readAuth(base / cfg.at("auth_file").get<std::string>())),
      hiRes_(cfg.at("hi_res").get<std::string>()),
      loRes_(cfg.at("lo_res").get<std::string>()),
      settleS_(cfg.value("settle_s", 3.0))
{
}

std::string Camera::get(const std::string& path, long timeoutS) const
{
    const std::unique_ptr<CURL, CurlDeleter> c(curl_easy_init());
    if (!c) {
        throw CameraError("curl_easy_init failed");
    }
    std::string body;
    const std::string url = url_ + path;
    curl_easy_setopt(c.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(c.get(), CURLOPT_USERPWD, userpwd_.c_str());
    curl_easy_setopt(c.get(), CURLOPT_TIMEOUT, timeoutS);
    curl_easy_setopt(c.get(), CURLOPT_WRITEFUNCTION, collect);
    curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &body);
    const CURLcode rc = curl_easy_perform(c.get());
    long status = 0;
    curl_easy_getinfo(c.get(), CURLINFO_RESPONSE_CODE, &status);
    if (rc != CURLE_OK || status != 200) {
        throw CameraError(url + ": " + (rc != CURLE_OK ? curl_easy_strerror(rc) : "HTTP " + std::to_string(status)));
    }
    return body;
}

void Camera::setRes(Res res)
{
    if (current_ == static_cast<int>(res)) {
        return;
    }
    (void)get("/settings/video_size?set=" + (res == Res::Hi ? hiRes_ : loRes_), 5);
    std::this_thread::sleep_for(std::chrono::duration<double>(settleS_));
    current_ = static_cast<int>(res);
}

cv::Mat Camera::grab(Res res)
{
    const int wantW = std::stoi(res == Res::Hi ? hiRes_ : loRes_);
    for (int attempt = 0; attempt < 4; ++attempt) {
        setRes(res);
        const std::string jpg = get("/shot.jpg", 15);
        const std::vector<uchar> buf(jpg.begin(), jpg.end());
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (!img.empty() && img.cols == wantW) {
            return img;
        }
        current_ = -1;  // the phone sometimes drops the setting: set it again
    }
    throw CameraError("no " + std::string(res == Res::Hi ? hiRes_ : loRes_) + " image from the camera");
}

}  // namespace spc
