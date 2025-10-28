#!/usr/bin/env bash

vllm serve Qwen/Qwen3-4B-Instruct-2507 --dtype auto --api-key 123 --max-model-len 4K --gpu-memory-utilization 0.7 --enable-auto-tool-choice --tool-call-parser hermes