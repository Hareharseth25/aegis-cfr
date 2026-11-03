"""Value network export/inference consistency and the PyTorch river solver."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

import aegis  # noqa: E402
from evaluation.torch_river import TorchRiverCFR  # noqa: E402
from models.value_net import ValueNet, zero_sum_project, zero_sum_residual  # noqa: E402


def test_exported_network_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    model = ValueNet((64, 32)).eval()
    x = torch.rand(5, aegis.NN_INPUT_SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    net = model.to_aegis()
    out = net.forward(x.numpy())
    assert np.max(np.abs(out - ref)) < 1e-4
    path = str(tmp_path / "net.bin")
    net.save(path)
    assert np.array_equal(aegis.MLP.load(path).forward(x.numpy()), out)


def test_zero_sum_projection():
    torch.manual_seed(1)
    x = torch.zeros(3, aegis.NN_INPUT_SIZE)
    x[:, 52:52 + 1326] = torch.rand(3, 1326)
    x[:, 52:52 + 1326] /= x[:, 52:52 + 1326].sum(1, keepdim=True)
    x[:, 52 + 1326:52 + 2652] = torch.rand(3, 1326)
    x[:, 52 + 1326:52 + 2652] /= x[:, 52 + 1326:52 + 2652].sum(1, keepdim=True)
    y = torch.randn(3, aegis.NN_OUTPUT_SIZE)
    assert zero_sum_residual(x, zero_sum_project(x, y)).abs().max() < 1e-5


def test_build_features_layout():
    r0 = np.zeros(1326)
    r0[aegis.parse_combo("AdJd")] = 2.0
    r1 = np.ones(1326)
    f = aegis.build_features(aegis.parse_cards("AsKh8c4d2s"), r0.tolist(), r1.tolist(), 100.0, 150.0)
    assert f.shape == (aegis.NN_INPUT_SIZE,)
    assert f[:52].sum() == 5
    assert abs(f[52:52 + 1326].sum() - 1) < 1e-6
    assert abs(f[52 + 1326:52 + 2652].sum() - 1) < 1e-5
    assert abs(f[-2] - 100 / 400) < 1e-6


def test_torch_river_matches_cpp_solver():
    c = aegis.TreeConfig()
    c.board = aegis.parse_cards("AsKh8c4d2s")
    c.pot, c.effective_stack = 100, 150
    s = aegis.StreetSizes()
    s.bet_fractions, s.raise_fractions, s.max_raises = [0.5, 1.0], [1.0], 2
    c.set_sizes(s, s)
    tree = aegis.PublicTree.build(c)
    ctx = aegis.SubgameContext(tree, aegis.Range.parse("AA,KK,AK,AQ,88,QJs,T9s").remove(c.board),
                               aegis.Range.parse("AQ,KQ,QQ,99,JTs,A5s").remove(c.board))
    cfg = aegis.CFRConfig.dcfr()
    cfg.tie_tolerance = 0.0
    cpp = aegis.VectorCFR(ctx, aegis.VectorCFRConfig(cfg))
    cpp.iterate(30)
    tr = TorchRiverCFR(tree, ctx, "cpu", torch.float64)
    tr.iterate(30)
    a, b = cpp.average_policy(), tr.average_policy()
    diff = max(np.max(np.abs(a.get(k) - b.get(k))) for k in range(tree.num_decision_nodes))
    assert diff < 1e-6
    assert abs(ctx.exploitability(a)["nash_conv"] - ctx.exploitability(b)["nash_conv"]) < 1e-6
