# Performance Budgets

## Latency Targets
- H-module daily update: <50ms
- L-module minute update: <10ms  
- ACT halting: max 8 steps
- End-to-end inference: <100ms

## Memory Footprints
- Model parameters: ~27M (54MB fp16)
- Activation memory: ~128MB peak
- KV cache: ~32MB per sequence
- GPU memory budget: <2GB total

## Throughput Requirements
- A100: >1000 inferences/sec
- V100: >500 inferences/sec  
- CPU fallback: >50 inferences/sec
- Batch processing: 32-128 samples
