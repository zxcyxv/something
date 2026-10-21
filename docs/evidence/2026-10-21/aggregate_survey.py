"""Aggregate the hand survey of ARC tasks (one JSON object per task: id, set, rule, classes, primary, operator,
visible, checked, confidence, missing) and join it with the signature audit (arc_prior_audit.json).

    python -I docs/evidence/2026-10-21/aggregate_survey.py docs/evidence/2026-10-21/arc_operator_survey.jsonl \
        docs/evidence/2026-10-21/arc_prior_audit.json
"""
import collections
import json
import sys

LEVEL = {'A1': 'A', 'A2': 'A', 'A3': 'A', 'B1': 'B', 'B2': 'B', 'C1': 'C', 'C2': 'C', 'C3': 'C', 'C4': 'C',
         'D1': 'D', 'D2': 'D', 'D3': 'D', 'E1': 'E', 'E2': 'E', 'F1': 'F', 'F2': 'F', 'F3': 'F', 'G': 'G'}
ORDER = 'ABCDEFG'


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    audit = {(t['set'], t['id']): t for t in json.load(open(sys.argv[2]))['tasks']} if len(sys.argv) > 2 else {}
    for s in sorted({r['set'] for r in rows}):
        rs = [r for r in rows if r['set'] == s]
        n = len(rs)
        pct = lambda k: f'{k} ({100.0 * k / n:.0f}%)'
        print(f'== {s}: {n} tasks')
        print('visible:', {k: pct(v) for k, v in collections.Counter(r['visible'] for r in rs).most_common()})
        print('checked numerically:', pct(sum(bool(r.get('checked')) for r in rs)))
        print('primary:', {k: v for k, v in sorted(collections.Counter(r['primary'] for r in rs).items())})
        print('primary level:', {k: v for k, v in sorted(collections.Counter(LEVEL.get(r['primary'], '?') for r in rs).items())})
        top = collections.Counter(max((LEVEL.get(c, '?') for c in r['classes']), key=lambda x: ORDER.find(x))
                                  for r in rs)
        print('highest level used:', dict(sorted(top.items())))
        print('class involvement:', {k: v for k, v in sorted(collections.Counter(c for r in rs for c in set(r['classes'])).items())})
        print('classes per task: mean', round(sum(len(set(r['classes'])) for r in rs) / n, 2))
        conf = [r['confidence'] for r in rs if isinstance(r.get('confidence'), (int, float))]
        print('confidence mean', round(sum(conf) / max(1, len(conf)), 2))
        if audit:
            hit = sum(1 for r in rs if audit.get((s, r['id']), {}).get('classes'))
            print('signature audit fires on', pct(hit))
        for r in rs:
            if r['visible'] != 'clear':
                print(f"  [{r['visible']}] {r['id']} {r['primary']}: {r['rule'][:140]} | missing: {r.get('missing', '')[:120]}")
        print()


if __name__ == '__main__':
    main()
