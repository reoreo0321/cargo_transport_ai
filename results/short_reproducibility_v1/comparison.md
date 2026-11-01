# Short-history repeatability confirmation

60 new runs, frozen model and guards. Three process restarts at seed 22301, one each at 22401 and 22501.
All runs retained. Times below are simulation seconds of arrivals only. Budget: 5 mm.

| Cargo | Route | Method | Arrivals within budget | Arrival min–median–max [s] | Peak slip min–median–max [mm] |
|---|---|---|---:|---:|---:|
| experience_weak | AB | short_model | 5/5 | 25.650 / 25.650 / 25.650 | 4.662 / 4.662 / 4.662 |
| experience_weak | AB | memory_zero | 5/5 | 29.350 / 29.350 / 29.350 | 4.366 / 4.366 / 4.366 |
| experience_weak | AC | short_model | 5/5 | 28.000 / 28.000 / 28.000 | 2.195 / 2.195 / 2.195 |
| experience_weak | AC | memory_zero | 5/5 | 30.900 / 30.900 / 30.900 | 4.365 / 4.365 / 4.365 |
| experience_mid | AB | short_model | 5/5 | 24.050 / 24.050 / 24.050 | 1.028 / 1.028 / 1.028 |
| experience_mid | AB | memory_zero | 5/5 | 24.050 / 24.050 / 24.050 | 1.028 / 1.028 / 1.033 |
| experience_mid | AC | short_model | 5/5 | 25.450 / 25.450 / 25.450 | 0.750 / 0.750 / 0.750 |
| experience_mid | AC | memory_zero | 5/5 | 25.450 / 25.450 / 25.450 | 0.750 / 0.750 / 0.750 |
| experience_strong | AB | short_model | 5/5 | 23.900 / 23.900 / 23.900 | 0.186 / 0.186 / 0.186 |
| experience_strong | AB | memory_zero | 5/5 | 23.900 / 23.900 / 23.900 | 0.186 / 0.186 / 0.186 |
| experience_strong | AC | short_model | 5/5 | 25.000 / 25.000 / 25.000 | 0.132 / 0.132 / 0.132 |
| experience_strong | AC | memory_zero | 5/5 | 25.000 / 25.000 / 25.000 | 0.132 / 0.132 / 0.132 |

Stopped runs are never assigned an arrival time. Same-seed details, full-precision statistics, paired results and command audits are in summary.json.
This is confirmation on existing evaluation cargos, not a fresh holdout. Seed repetitions do not establish a safety guarantee.
