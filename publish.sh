#!/usr/bin/env bash
cd $(dirname "$0")

uvx hatch clean
uvx codetoprompt \
    --compress \
    --output "./llms.txt" \
    --respect-gitignore \
    --cxml \
    --exclude "*.svg,.specstory,ref,testdata,*.lock,llms.txt" \
    "."
gitnextver .
uvx hatch build
uv publish
