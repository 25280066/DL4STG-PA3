# ===== Q1(b) =====
import numpy as np, itertools
K = 8; B = K + 1
phi = {-1: 1, 0: B, +1: B**2}
decode = lambda s: (s % B, (s // B) % B, s // B**2)
triples, sums = [], []
for n_s, n_0, n_p in itertools.product(range(K + 1), repeat=3):
    if not (1 <= n_s + n_0 + n_p <= K): continue
    trends = np.array([-1]*n_s + [0]*n_0 + [1]*n_p)
    S = sum(phi[x] for x in trends)
    assert decode(S) == (n_s, n_0, n_p)
    triples.append((n_s, n_0, n_p)); sums.append(S)
print("phi(-1), phi(0), phi(+1) =", phi[-1], phi[0], phi[1])
print("valid triples:", len(triples), "| distinct sums:", len(set(sums)), "| all decoded correctly: True")
rng = np.random.default_rng(0)
for _ in range(1000):
    x = rng.choice([-1, 0, 1], size=rng.integers(1, K + 1))
    assert (x**2).sum() == np.count_nonzero(x)
print("phi(x)=x^2 counts non-steady neighbours: verified")

# ===== Q1(c): brute-force check of the table =====
from collections import defaultdict
onehot = {-1: np.array([1,0,0]), 0: np.array([0,1,0]), 1: np.array([0,0,1])}
hoods = [c for n in range(1, 9) for c in itertools.combinations_with_replacement([-1,0,1], n)]
tasks = {"(i) #steady": lambda h: h.count(0),
         "(ii) >half slowing": lambda h: h.count(-1) > len(h)/2,
         "(iii) any slowing": lambda h: -1 in h}
aggs = {"Sum": lambda M: M.sum(0), "Mean": lambda M: M.mean(0), "Max": lambda M: M.max(0)}
for tname, task in tasks.items():
    for aname, agg in aggs.items():
        seen = defaultdict(set)
        for h in hoods:
            key = tuple(np.round(agg(np.stack([onehot[x] for x in h])), 10))
            seen[key].add(task(list(h)))
        print(f"{tname:22s} {aname:5s} possible = {all(len(v) == 1 for v in seen.values())}")

# ===== Q2 =====
import os, matplotlib.pyplot as plt
import torch, torch.nn.functional as F
os.makedirs("figs", exist_ok=True)
N = 40
true_step = lambda x: np.roll(x, -1)                       # x_i(t+1) = x_{i+1}(t)

# (a)(i) least squares on the isolated slowdown
x0 = np.zeros(N); x0[20] = 1.0
y = true_step(x0)
X = np.stack([x0, np.roll(x0, 1) + np.roll(x0, -1)], axis=1)
(a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"Q2(a)(i): a = {a:.4f}, b = {b:.4f}, min SSE = {np.sum((X @ [a, b] - y)**2):.4f}")

# (a)(ii) 20-step rollout vs true motion
sym_step = lambda x: a*x + b*(np.roll(x, 1) + np.roll(x, -1))
i = np.arange(N); bump = np.exp(-(i - 20)**2 / 8); T = 20
pred, true = [bump], [bump]
for _ in range(T):
    pred.append(sym_step(pred[-1])); true.append(true_step(true[-1]))
pred, true = np.array(pred), np.array(true)
vmin, vmax = min(pred.min(), true.min()), max(pred.max(), true.max())
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, M, title in zip(axes, [pred, true], ["Symmetric layer (a=0, b=1/2)", "True motion (np.roll(x,-1))"]):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=[-0.5, N-0.5, -0.5, T+0.5])
    ax.set_title(title); ax.set_xlabel("sensor index i")
    ax.set_yticks(range(0, T + 1, 2)); ax.set_xticks(range(0, N, 5))
axes[0].set_ylabel("time step t")
fig.colorbar(im, ax=axes, label="slowdown $x_i(t)$")
plt.savefig("figs/q2a_heatmaps.png", dpi=200, bbox_inches="tight"); plt.show()
c = (i * pred[-1]).sum() / pred[-1].sum()
print(f"t=20 predicted: peak {pred[-1].max():.3f}, centre {c:.2f}, variance {((i-c)**2*pred[-1]).sum()/pred[-1].sum():.2f}")
print(f"t=20 true: peak {true[-1].max():.3f} at sensor {true[-1].argmax()}")

# (b) kernel [0,0,1] with circular padding
rng = np.random.default_rng(0); xr = rng.normal(size=N)
xt = torch.tensor(xr, dtype=torch.float32).view(1, 1, N)
out = F.conv1d(F.pad(xt, (1, 1), mode="circular"), torch.tensor([[[0., 0., 1.]]])).view(-1).numpy()
print("Q2(b) conv1d exact:", np.allclose(out, true_step(xr), atol=1e-6))

# (c) a=0, b=1, c=-1 for both directions
hist_step = lambda x_t, x_prev: (np.roll(x_t, 1) + np.roll(x_t, -1)) - x_prev
for name, step in [("backward", lambda x: np.roll(x, -1)), ("forward", lambda x: np.roll(x, 1))]:
    errs = []
    for _ in range(100):
        xp = rng.normal(size=N); xt_ = step(xp)
        errs.append(np.abs(hist_step(xt_, xp) - step(xt_)).max())
    print(f"Q2(c) {name}: max |error| = {max(errs):.2e}")