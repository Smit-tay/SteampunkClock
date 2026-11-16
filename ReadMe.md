<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2024-2026 Jack Sidman Smith -->

# Steampunk Clock

A uArm Swift Pro shows the time with handwritten digit tiles.
The program is `spclock`, a C++20 application in `app/`.
It sees the tiles through an overhead camera and reads them with a neural network.
Visitors to robot.smithjack.net can ask it to set the current time.

## ROS2 and TopicFS

The arm is driven by SwiftRos2, a ROS2 driver for the uArm Swift Pro.
TopicFS (git.smithjack.net/jack/TopicFS) exposes ROS2 topics and services as plain files.
`spclock` writes JSON to a service's `command` file and reads its `response`.
It uses three services under `/mnt/topicfs/swiftpro`: `move_to`, `set_pump` and `set_wrist`.
Position and pump status come from each topic's `latest` file.
`spclock` never opens the serial port, and both containers keep running.

## Computer vision with OpenCV

OpenCV handles every image the overhead camera takes.
The camera is a phone running IP Webcam, fetched over HTTP with libcurl.
A homography maps 1600x1200 image pixels to the 100 mm grid drawn on the table.
An affine transform maps that grid to the robot's own frame.
For the heap, the photo is resampled into a top-down map at 2 px/mm.
A 40 x 30 mm rectangle is then swept over the yellow area in HSV.

Each pose scores by how much of its outline shows a tile edge.
A covered tile loses part of its outline, so free tiles score highest.
The best poses are refined in steps of 1, 0.5 and 0.25 mm.
Each tile found is cut out as an 88 x 88 pixel patch.
The patch covers 44 mm at 0.5 mm per pixel, with the digit's top up.
After every update, a final photo checks that each display slot reads correctly.

## Digit recognition and training

Digits are read by a small neural network, with no cloud service or LLM.
Each patch becomes an ink map, a saturation map and gradient-orientation histograms.
PCA reduces these features to 80 dimensions.
Five OpenCV `ANN_MLP` networks with 64 hidden units each are then averaged.
The classes are the digits 0 to 9, "not upright" and "not a tile".
A tile is read at four quarter turns, and the best upright reading wins.

The arm photographed its own training data with `spclock collect`.
Each of the 33 tiles was placed on a free spot at four quarter turns.
Offset patches beside each tile were saved as "not a tile" examples.
Training adds jitter, quarter turns and neighbouring tiles pasted in at random.
Each network trains on a random 80% of the data, using RPROP for 300 iterations.
A repeatable 15% of the photos is held out to report accuracy.

## How a time update works

The tiles are stored in a keypad of stacks, at most four high.
The display has four slots: hours on the left, minutes on the right.
One photo first shows where the display tiles and stack tops really lie.
Only slots whose digit changes are touched, each in under 10 seconds.
The arm moves only far enough to clear the camera's view between photos.
The model treats 0 and 8 as the same either way up.

## Sharing the arm

`spclock` and the robot.smithjack.net movement page share one arm.
Every command that moves it holds `/var/lib/robot/robot_move.lock`.
The web service listens only on 127.0.0.1:8765, one JSON line per connection.
The page's `clock.php` asks it for status or for an update.
An update runs only when a visitor clicks "Update time".

## Build

Everything builds in a Fedora 44 container that matches the worker.
The host needs only podman.

```sh
podman build -t steampunkclock_dev -f ide/Containerfile.build ide   # only when the Containerfile changes
./ide/cmake-in-container.sh -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
./ide/cmake-in-container.sh --build build
```

Dependencies are OpenCV (core, imgproc, imgcodecs, ml), libcurl and nlohmann-json.
See `ide/README.md` for VS Code and clangd.

## Deploy

Deploy runs on the host, because the build container has no network.

```sh
cmake --build build --target deploy-spclock      # spclock and spclock.json to worker:~/steampunkclock
ssh worker 'sudo install -m 755 ~/steampunkclock/spclock /usr/local/bin/spclock && sudo systemctl restart spclock'
```

The worker keeps its own state, model and data next to `spclock.json`.
These are `spclock_state.json`, `digits.yml`, `data/` and `camera.auth`.
`camera.auth` holds the camera's `user:password` and is never committed.
The service unit is `app/systemd/spclock.service`.

## Commands

Run `spclock [-c CONFIG] COMMAND`; the config defaults to `spclock.json`.

| Command | Purpose |
|---------|---------|
| `status` | print the stacks and the display |
| `show HHMM` / `show clear` | put a time on the display, or empty it |
| `serve [PORT]` | web service for the website (default 8765) |
| `sort` | clear a jumbled heap into the digit stacks |
| `collect [DIGITS]` | photograph the stored tiles for training |
| `train [--per N] [--final] DIR...` | train the digit network |
| `read X Y`, `locate X Y` | read the tile near a robot point |
| `photo FILE`, `heap IN OUT` | save a photo; draw the tiles found on a heap |
| `calplace KEY` | measure placement errors for `place_corrections` |
| `fetch`, `carry`, `drop`, `goto` | single moves for setup and testing |

`spclock` with no arguments prints the full list.

## Configuration

`app/config/spclock.json` holds the calibration, motion settings and table layout.
Lengths are in mm and angles in degrees, in the robot's frame.
The calibration must be redone if the camera or the arm moves.
