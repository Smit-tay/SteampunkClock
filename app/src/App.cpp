// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
#include "App.h"

#include <fstream>
#include <stdexcept>

namespace spc {
namespace {

nlohmann::json readConfig(const std::filesystem::path& file)
{
    std::ifstream f(file);
    if (!f) {
        throw std::runtime_error("cannot read config " + file.string());
    }
    return nlohmann::json::parse(f);
}

cv::Point2d point(const nlohmann::json& j) { return {j.at(0).get<double>(), j.at(1).get<double>()}; }

}  // namespace

App::App(const std::filesystem::path& configFile)
    : cfg(readConfig(configFile)),
      base(std::filesystem::absolute(configFile).parent_path()),
      stateFile(base / cfg.at("state_file").get<std::string>()),
      modelFile(base / cfg.at("model_file").get<std::string>()),
      dataDir(base / cfg.at("data_dir").get<std::string>()),
      lockFile(cfg.at("lock_file").get<std::string>()),
      geo(cfg.at("calibration"), cfg.at("tile")),
      cam(cfg.at("camera"), base),
      arm(cfg.at("topicfs").get<std::string>(), geo),
      mover(arm, geo, cfg.at("motion"))
{
    const auto& lay = cfg.at("layout");
    maxStack = lay.at("max_stack").get<int>();
    upStack = geo.gridDirToRobot(lay.at("up_grid_deg").get<double>());
    photoSpot = point(lay.at("photo_spot"));
    tempSpot = keypadCell(lay.value("temp_cell", 11));
    for (const auto& s : lay.value("display_grid", nlohmann::json::array())) {
        displaySlots.push_back(geo.gridToRobot(point(s)));
    }
}

cv::Point2d App::keypadCell(int k) const
{
    const auto& lay = cfg.at("layout");
    const double gx = lay.at("row_x0").get<double>() + lay.at("row_pitch").get<double>() * (k / 3);
    const double gy = lay.at("col_y0").get<double>() + lay.at("col_pitch").get<double>() * (k % 3);
    return geo.gridToRobot({gx, gy});
}

}  // namespace spc
