# IDE Setup

The code is built in a container (`steampunkclock_dev`, Fedora 44 like the worker, with cmake, ninja, g++ and the
OpenCV, libcurl and nlohmann-json development packages), so the host needs only podman. `~/dev` is mounted at the
same path inside the container, so source paths and `build/compile_commands.json` are valid on the host too, for
clangd.

## Files

- `Containerfile.build`: the build image. Rebuild it only when this file changes:
  `podman build -t steampunkclock_dev -f ide/Containerfile.build ide`
- `cmake-in-container.sh`: runs `cmake` with its arguments inside the image.

## Build

```sh
./ide/cmake-in-container.sh -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
./ide/cmake-in-container.sh --build build
```

Deploy from the host (the container has no network): `cmake --build build --target deploy-spclock`.

## VS Code

`.vscode/tasks.json` has the image build, Build, Clean and Deploy to worker. `.vscode/settings.json` points the CMake
extension at the wrapper script and clangd at `build/`.
