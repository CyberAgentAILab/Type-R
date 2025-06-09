#!/usr/bin/env bash

set -eux

# Apply and build patches.
for package in apex detectron2 MaskTextSpotterV3 DeepSolo craft-text-detector VLMEvalKit; do
    git apply --directory packages/$package patch/$package.patch $@
done
