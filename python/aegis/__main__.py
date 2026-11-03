"""Command-line interface: python -m aegis <command> ...

  solve REQUEST.json [--out RESULT.json]   solve a turn/river spot
  demo                                       solve the README demo spot
  equity --hero AdJd --villain "AK,QQ+" --board AsKh8c [--mc N]
  evaluate As Kh 8c 4d 2s Ad Jd              describe a 5-7 card hand
"""
from __future__ import annotations

import argparse
import json
import sys

from . import aegis_core as ac
from .spot import solve_request

DEMO = {
    "board": ["As", "Kh", "8c", "4d", "2s"], "pot": 100, "effective_stack": 150, "player_to_act": 0,
    "actions": {"bet_pot_fractions": [0.33, 0.75, 1.5], "raise_pot_fractions": [1.0], "allow_all_in": True,
                "max_raises": 3},
    "range_p0": "AA,KK,88,44,22,AK,AQ,AJ,AT,A9s-A2s,KQ,KJs,QJs,JTs,T9s,98s,87s,76s,65s,54s,K8s,Q8s:0.5,J8s:0.5",
    "range_p1": "AQ,AJ,AT,A9-A2,KQ,KJ,KT,K9s,QQ-99,77-55,33,QJs,JTs,T9s,98s,87s",
    "solver": {"algorithm": "dcfr", "iterations": 1000},
    "hero": {"player": 1, "hand": "AdJd", "path": ["Bet 75"]},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aegis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("request")
    s.add_argument("--out")
    d = sub.add_parser("demo")
    d.add_argument("--out")
    e = sub.add_parser("equity")
    e.add_argument("--hero", required=True, help="hand (AdJd) or range notation")
    e.add_argument("--villain", required=True)
    e.add_argument("--board", default="")
    e.add_argument("--mc", type=int, default=0, help="Monte Carlo samples (0 = exact)")
    v = sub.add_parser("evaluate")
    v.add_argument("cards", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd in ("solve", "demo"):
        req = DEMO if args.cmd == "demo" else json.load(open(args.request))
        res = solve_request(req)
        text = json.dumps(res, indent=1)
        if args.out:
            open(args.out, "w").write(text)
        ex = res["exploitability"]
        print(f"solved {res['num_nodes']} public nodes, hands {res['hands']}, {res['solve_seconds']:.2f}s; "
              f"exploitability {ex['exploitability_pct_pot']:.4f}% of pot (in this subgame)")
        if "hero" in res:
            h = res["hero"]
            print(f"hero {h['hand']} at '{h['path']}':")
            for a, p_, evp in zip(h.get("actions", []), h.get("strategy", []), h.get("action_ev_from_here", [])):
                print(f"  {a:>12}: {100 * p_:6.2f}%   EV from here {evp:8.3f} BB")
        return 0
    if args.cmd == "equity":
        board = ac.parse_cards(args.board) if args.board else []
        hero_is_hand = len(args.hero) == 4 and args.hero[1].islower()
        villain = ac.Range.parse(args.villain)
        if hero_is_hand:
            r = ac.equity_vs_range_exact(ac.parse_combo(args.hero), villain, board)
            print(f"{args.hero} vs [{args.villain}]: equity {100 * r.equity():.3f}% "
                  f"(win {100 * r.win:.3f}%, tie {100 * r.tie:.3f}%), {r.samples} boards enumerated")
        else:
            hero = ac.Range.parse(args.hero)
            if args.mc or len(board) < 3:
                r = ac.equity_range_vs_range_mc(hero, villain, board, args.mc or 200000, 1)
                print(f"equity {100 * r.equity():.3f}% ± {196 * r.std_error:.3f}% (95% CI, Monte Carlo)")
            else:
                r, _, _ = ac.equity_range_vs_range_exact(hero, villain, board)
                print(f"equity {100 * r.equity():.4f}% (exact)")
        return 0
    if args.cmd == "evaluate":
        cards = [ac.parse_card(c) for c in args.cards]
        val = ac.evaluate_hand(cards)
        print(f"{ac.describe_hand(val)} (value {val})")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
