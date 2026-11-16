# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Jack Sidman Smith
cmake_minimum_required(VERSION 3.22)

project(Steampunk_Clock
        VERSION 0.3
        DESCRIPTION "uArm Swift Pro project to make a digital clock in the Steampunk style"
        LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd

# Deploy (app/CMakeLists.txt, target deploy-spclock): run on the HOST, the build container
# has no network. Build in the container first; it matches the worker (Fedora 44).
set(DEPLOY_HOST "worker"         CACHE STRING "ssh host to deploy to")
set(DEPLOY_DIR  "steampunkclock" CACHE STRING "Directory on DEPLOY_HOST, relative to home")

# Vision-guided clock (topicfs, camera, digit network)
add_subdirectory(app)
