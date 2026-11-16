#!/bin/bash
# SPDX-License-Identifier: MIT
# Runs cmake inside the steampunkclock_dev image. ~/dev is mounted at the same path as on the host,
# so source paths, build paths and compile_commands.json agree on both sides.
# SELinux labelling is switched off for this one container instead of relabelling all of ~/dev.
exec podman run --rm --network=none --userns=keep-id --security-opt label=disable \
    -v "$HOME/dev:$HOME/dev" -w "$PWD" steampunkclock_dev cmake "$@"
