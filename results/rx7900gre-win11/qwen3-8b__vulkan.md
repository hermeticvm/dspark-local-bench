### qwen3-8b · vulkan · AMD Radeon RX 7900 GRE 16GB

llama.cpp `0.7.0-dev (build 11865, commit 9a0fde996)` · Windows-11-10.0.26200-SP0 · target `models/Qwen3-8B-Q8_0.gguf` · ctx 8192 · bench `qualitative` (coding,math,writing,qa, limit 8, osl 512) · updated 2026-10-25T15:02:45+00:00

| run | decode t/s | speedup | prompt t/s | prompt vs base | accept | coding × | math × | qa × | writing × | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| dspark-n3-p0.3 | 115.0 | 1.86× | 1076.2 | 0.92× | 65.1% | 2.28× | 1.89× | 1.51× | 1.77× | 0 |
| dspark-n3-p0 | 111.5 | 1.81× | 1068.9 | 0.91× | 63.3% | 2.27× | 1.70× | 1.44× | 1.82× | 0 |
| dspark-n5-p0 | 109.1 | 1.77× | 1090.2 | 0.93× | 51.7% | 2.33× | 1.71× | 1.31× | 1.73× | 0 |
| dspark-n7-p0 | 100.4 | 1.63× | 1075.7 | 0.92× | 43.0% | 2.30× | 1.57× | 1.19× | 1.44× | 0 |
| dflash-n4 | 95.1 | 1.54× | 1113.7 | 0.95× | 44.0% | 2.10× | 1.52× | 1.14× | 1.41× | 0 |
| draft-simple-n3 | 83.9 | 1.36× | 1024.3 | 0.87× | 57.8% | 1.52× | 1.29× | 1.26× | 1.36× | 0 |
| baseline | 61.7 | 1.00× | 1173.0 | 1.00× | – | 1.00× | 1.00× | 1.00× | 1.00× | 0 |
