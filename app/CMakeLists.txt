# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jack Sidman Smith
#
# spclock: vision-guided SteampunkClock (topicfs arm, overhead camera, local digit network).
# Build in the container (Fedora 44, same as the worker), deploy with the deploy-spclock target.

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs ml)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

add_executable(spclock
    src/main.cpp
    src/App.cpp
    src/Arm.cpp
    src/Camera.cpp
    src/Clock.cpp
    src/Collect.cpp
    src/Dataset.cpp
    src/DigitNet.cpp
    src/Geometry.cpp
    src/Heap.cpp
    src/Mover.cpp
    src/Serve.cpp
    src/Sort.cpp
    src/State.cpp
    src/Train.cpp
    src/Vision.cpp
)
target_compile_options(spclock PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
target_link_libraries(spclock PRIVATE ${OpenCV_LIBS} CURL::libcurl nlohmann_json::nlohmann_json Threads::Threads)

# Copies the binary and the config (not the state, model or data, which live on the worker).
add_custom_target(deploy-spclock
    COMMAND ssh ${DEPLOY_HOST} mkdir -p ${DEPLOY_DIR}
    COMMAND rsync -av ${CMAKE_BINARY_DIR}/app/spclock ${CMAKE_CURRENT_SOURCE_DIR}/config/spclock.json
            ${DEPLOY_HOST}:${DEPLOY_DIR}/
    COMMENT "Copying spclock to ${DEPLOY_HOST}:${DEPLOY_DIR}"
    VERBATIM)
