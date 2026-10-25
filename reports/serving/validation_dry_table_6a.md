| metric | replay | test-block 95% CI | verdict |
|---|---|---|---|
| test2 MAE diff vs persistence at 15 (model minus baseline, minutes) | -0.0536 | [-0.3350, -0.0924] | outside (worse) |
| test2 MAE diff vs persistence at 60 (model minus baseline, minutes) | -0.2776 | [-0.2248, -0.0788] | outside (better) |
| test2 MAE diff vs historical at 15 (model minus baseline, minutes) | -2.0884 | [-2.2719, -1.4203] | within |
| test2 MAE diff vs historical at 60 (model minus baseline, minutes) | -0.9347 | [-0.6080, -0.1531] | outside (better) |
| test3 coverage overall (held-out routes) | 0.8172 | [0.7877, 0.8172] | outside (better) |
| test3 coverage at 15 | 0.8178 | [0.7952, 0.8189] | within |
| test3 coverage at 60 | 0.8166 | [0.7888, 0.8138] | outside (better) |
| test4 Brier P(delay>=15) at 15 | 0.0205 | [0.0147, 0.0275] | within |
| test4 ECE P(delay>=15) at 15 | 0.0136 | [0.0098, 0.0201] | within |
| test4 Brier diff vs historical at 15 (model minus historical) | -0.0500 | [-0.0436, -0.0157] | outside (better) |
| test4 Brier P(delay>=15) at 60 | 0.0412 | [0.0303, 0.0512] | within |
| test4 ECE P(delay>=15) at 60 | 0.0114 | [0.0106, 0.0254] | within |
| test4 Brier diff vs historical at 60 (model minus historical) | -0.0293 | [-0.0186, -0.0039] | outside (better) |
| test5 BHM Brier diff vs gap-bucket, not-yet-arrived (final minus bar) | -0.0083 | [-0.1288, -0.0773] | outside (worse) |
| test5 BHM ECE, not-yet-arrived | 0.0926 | [0.0025, 0.0095] | outside (worse) |
| test5 MAN Brier diff vs gap-bucket, not-yet-arrived (final minus bar) | -0.0129 | [-0.0981, -0.0679] | outside (worse) |
| test5 MAN ECE, not-yet-arrived | 0.0905 | [0.0114, 0.0168] | outside (worse) |
| test5 BRI Brier diff vs gap-bucket, not-yet-arrived (final minus bar) | -0.0169 | [-0.1160, -0.0890] | outside (worse) |
| test5 BRI ECE, not-yet-arrived | 0.0850 | [0.0047, 0.0154] | outside (worse) |
