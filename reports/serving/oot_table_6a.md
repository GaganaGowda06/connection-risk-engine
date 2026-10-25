| metric | replay | test-block 95% CI | verdict |
|---|---|---|---|
| test2 MAE diff vs persistence at 15 (model minus baseline, minutes) | -0.1216 | [-0.3350, -0.0924] | within |
| test2 MAE diff vs persistence at 60 (model minus baseline, minutes) | -0.1650 | [-0.2248, -0.0788] | within |
| test2 MAE diff vs historical at 15 (model minus baseline, minutes) | -1.9943 | [-2.2719, -1.4203] | within |
| test2 MAE diff vs historical at 60 (model minus baseline, minutes) | -0.5287 | [-0.6080, -0.1531] | within |
| test3 coverage overall (held-out routes) | 0.8010 | [0.7877, 0.8172] | within |
| test3 coverage at 15 | 0.8058 | [0.7952, 0.8189] | within |
| test3 coverage at 60 | 0.7961 | [0.7888, 0.8138] | within |
| test4 Brier P(delay>=15) at 15 | 0.0248 | [0.0147, 0.0275] | within |
| test4 ECE P(delay>=15) at 15 | 0.0103 | [0.0098, 0.0201] | within |
| test4 Brier diff vs historical at 15 (model minus historical) | -0.0441 | [-0.0436, -0.0157] | outside (better) |
| test4 Brier P(delay>=15) at 60 | 0.0485 | [0.0303, 0.0512] | within |
| test4 ECE P(delay>=15) at 60 | 0.0122 | [0.0106, 0.0254] | within |
| test4 Brier diff vs historical at 60 (model minus historical) | -0.0205 | [-0.0186, -0.0039] | outside (better) |
| test5 BHM Brier diff vs gap-bucket, not-yet-arrived (final minus bar) | -0.1003 | [-0.1288, -0.0773] | within |
| test5 BHM ECE, not-yet-arrived | 0.0170 | [0.0025, 0.0095] | outside (worse) |
| test5 MAN Brier diff vs gap-bucket, not-yet-arrived (final minus bar) | -0.0822 | [-0.0981, -0.0679] | within |
| test5 MAN ECE, not-yet-arrived | 0.0227 | [0.0114, 0.0168] | outside (worse) |
| test5 BRI Brier diff vs gap-bucket, not-yet-arrived (final minus bar) | -0.1040 | [-0.1160, -0.0890] | within |
| test5 BRI ECE, not-yet-arrived | 0.0353 | [0.0047, 0.0154] | outside (worse) |
