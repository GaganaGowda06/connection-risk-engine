import numpy as np

from rail_risk.connections import build as B
from rail_risk.connections import data as DD
from rail_risk.connections import models as M
from rail_risk.connections import pipeline as P

cfg = B.load_config()
tr = DD.labelled(DD.flag_groups(DD.load_block("train"), cfg))
tr = tr[~tr.held_group]
va = DD.labelled(DD.flag_groups(DD.load_block("validation"), cfg))
days = sorted(tr.ssd.unique())[::8]
tr = tr[tr.ssd.isin(days)].reset_index(drop=True)
vd = sorted(va.ssd.unique())[:5]
vm = va[va.ssd.isin(vd)]
vok, vhe = vm[~vm.held_group].reset_index(drop=True), vm[vm.held_group].reset_index(drop=True)
P.GRID_C[:] = P.GRID_C[:1]
P.CROUNDS = 40
M.QROUNDS = 30
m = P.fit_all(tr, vok)
p = m.predict(vhe)
y = vhe.miss.to_numpy()
print(len(tr), len(vhe))
print({c: round(float(np.mean((p[c] - y) ** 2)), 4) for c in p})
