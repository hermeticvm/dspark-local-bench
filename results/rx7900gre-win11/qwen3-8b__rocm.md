### qwen3-8b · rocm · AMD Radeon RX 7900 GRE 16GB

llama.cpp `0.7.0-dev (build 11865, commit 9a0fde996)` · Windows-11-10.0.26200-SP0 · target `models/Qwen3-8B-Q8_0.gguf` · ctx 8192 · bench `qualitative` (coding,math,writing,qa, limit 8, osl 512) · updated 2026-10-25T15:26:46+00:00

| run | decode t/s | speedup | prompt t/s | prompt vs base | accept | coding × | math × | qa × | writing × | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| dspark-n7-p0 | 133.0 | 2.25× | 1328.2 | 0.91× | 43.9% | 3.18× | 2.13× | 1.59× | 2.10× | 0 |
| dspark-n5-p0 | 127.2 | 2.15× | 1333.9 | 0.91× | 52.3% | 2.93× | 2.07× | 1.60× | 2.01× | 0 |
| dflash-n4 | 110.1 | 1.86× | 1327.8 | 0.91× | 44.8% | 2.49× | 1.88× | 1.41× | 1.68× | 0 |
| dspark-n3-p0 | 109.7 | 1.85× | 1329.1 | 0.91× | 63.7% | 2.28× | 1.84× | 1.49× | 1.81× | 0 |
| dspark-n3-p0.3 | 109.0 | 1.84× | 1335.8 | 0.92× | 65.0% | 2.28× | 1.82× | 1.49× | 1.79× | 0 |
| draft-simple-n3 | 76.2 | 1.29× | 1319.2 | 0.90× | 57.9% | 1.56× | 1.26× | 1.11× | 1.22× | 0 |
| baseline | 59.1 | 1.00× | 1459.1 | 1.00× | – | 1.00× | 1.00× | 1.00× | 1.00× | 0 |
