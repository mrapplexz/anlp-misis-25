#!/usr/bin/env bash

python -m sglang.launch_server --model Qwen/Qwen3-4B-Instruct-2507 --context-length 4096 --dtype bfloat16 --grammar-backend llguidance --tool-call-parser qwen