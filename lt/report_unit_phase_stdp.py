"""Create a self-contained report for the unit-phase Gaussian experiment."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'runs/unit_phase_gaussian_20261004'
DEST = ROOT / 'docs/research/2026-10-04/unit_phase_gaussian_benchmark.html'
NAMES = {
    'fixed_gemm': '고정 위상·기존 지수창 — 비용 기준',
    'split_g': '동적 위상·기존 지수창 — 분리형',
    'unit_phase_torch': '단위 위상·가우시안 — Torch',
    'gaussian_lag_torch': '시차 부호·가우시안 — Torch 대조군',
    'unit_phase_triton': '단위 위상·가우시안 — 상대위상 전용 커널',
    'unit_phase_precomputed': '단위 위상·가우시안 — 뉴런별 위상 재사용',
    'gaussian_lag_triton': '시차 부호·가우시안 — 전용 커널 대조군',
}


def table(report):
    rows = ''
    for r in sorted(report['rows'], key=lambda r: (r['tokens'], list(NAMES).index(r['implementation']))):
        f, b = r['forward'], r['forward_backward']
        vals = [str(r['tokens']), NAMES[r['implementation']],
                f'{f["median_ms"]:.3f}', f'{b["median_ms"]:.3f}',
                f'{b["extra_peak_mib"]:.1f}', f'{b["peak_allocated_mib"]:.1f}']
        rows += '<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in vals) + '</tr>'
    return ('<table><tr><th>토큰 수</th><th>구현</th><th>순전파 ms</th>'
            '<th>순전파+역전파 ms</th><th>추가 최대 MiB</th><th>전체 최대 MiB</th></tr>'
            + rows + '</table>')


def main():
    b8 = json.loads((ART / 'compiled_b8.json').read_text())
    b8['rows'] += json.loads((ART / 'precomputed_b8.json').read_text())['rows']
    b128 = json.loads((ART / 'compiled_b128.json').read_text())
    checks = json.loads((ART / 'cuda_checks.json').read_text())
    assert len(b128['rows']) == 10 and all('forward_backward' in r for r in b128['rows'])
    output_error = max(c['output_max_abs_error'] for c in checks['cuda_correctness'])
    gradient_error = max(max(c['gradient_max_abs_errors'].values()) for c in checks['cuda_correctness'])
    idx = {(r['tokens'], r['implementation']): r for r in b128['rows']}
    unit = idx[(900, 'unit_phase_triton')]['forward_backward']
    old = idx[(900, 'split_g')]['forward_backward']
    fixed = idx[(900, 'fixed_gemm')]['forward_backward']
    control = idx[(900, 'gaussian_lag_triton')]['forward_backward']
    precomputed = idx[(900, 'unit_phase_precomputed')]['forward_backward']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for ax, extent in zip(axes, (2.6, .001)):
        for direction in (-1, 1):
            x = np.linspace(direction * 1e-9, direction * extent, 700)
            ax.plot(x, np.sign(np.sin(x)) * np.exp(-x*x), color='#2563eb', linewidth=2,
                    label='Unit direction x Gaussian' if direction == 1 else None)
            ax.plot(x, np.sign(x) * np.exp(-np.abs(x)), color='#667085', linestyle='--',
                    label='Original signed exponential' if direction == 1 else None)
            ax.plot(x, np.sin(x) * np.exp(-x*x), color='#16a34a', linestyle=':',
                    label='Sine direction x Gaussian' if direction == 1 else None)
        ax.scatter([0.], [0.], color='#101828', s=30, zorder=10)
        ax.axhline(0., color='#98a2b3', linewidth=.7)
        ax.set_xlabel('Phase difference (rad)')
        ax.set_ylabel('Window coefficient L')
        ax.set_ylim(-1.08, 1.08)
        ax.grid(alpha=.2)
    axes[0].set_title('Window shapes / gain=1 / tau=1 rad squared')
    axes[0].legend(fontsize=9)
    axes[1].set_title('Near zero: limits +/-1, exact zero contributes 0')
    png = ART / 'window_shapes.png'
    fig.savefig(png, dpi=180)
    plt.close(fig)
    image_data = base64.b64encode(png.read_bytes()).decode()
    probe = ''.join('<tr>' + ''.join('<td>' + html.escape(str(r[k])) + '</td>'
                    for k in ('lag', 'A', 'gaussian', 'gaussian_phase_gradient')) + '</tr>'
                    for r in checks['window'] if abs(r['lag']) <= 1e-7)
    body = f'''
<p class="meta">2026-10-04 · 단위 위상 방향과 가우시안 감쇠 · 구현 및 비용 측정</p>
<h1>단위 위상의 내적에서 STDP 강화·약화 방향 얻기</h1>
<p class="lead">단위 위상 관계의 허수부에서 <strong>크기를 버리고 부호만 읽는 A</strong>를 구현했다.
작은 시차에서도 A는 ±1을 유지하고, 정확히 같은 위상의 기여는 0으로 둔다.
출력 및 Q·K·V·위상 기울기 검증을 통과했다. 아래 수치는 전체 모델 학습이 아닌 G 쓰기·읽기의 비용이다.</p>

<h2>구현한 식과 범위</h2>
<pre>uV = exp(i phiV), uK = exp(i phiK)
c = uV conj(uK)
A = sign(Im(c)) = sign(sin(delta)), delta = phiV - phiK
L(delta) = A exp(-delta^2 / tau), tau = 1, gain = 1
G[i,j] = mean_t V[t,i] K[t,j] L(delta[t,i,j])
Y[s,i] = sum_j Q[s,j] G[i,j]</pre>
<p>τ는 시차 제곱의 척도이며 여기서는 1 rad²다. A의 크기는 시차의 크기와 분리되어 있다.
signed real K,V 활동값의 부호는 A의 부호와 별개다.
순간 G를 한 번 만든 뒤 모든 실수 Q가 공유해서 읽으며, 재귀 사이의 G 누적은 추가하지 않았다.</p>
<p><strong>위상차가 (−π,π) 안일 때 A의 부호가 시차 부호와 일치한다.</strong>
측정 입력의 각 위상은 [−1.3,1.3]이므로 차이는 [−2.6,2.6]이다.
이 범위 밖에서는 단위 위상이 앨리어싱하므로 동일한 시간 방향 보장이 성립하지 않는다.
STDP 시차는 채널 위상차이며, 반복 횟수나 토큰 위치 차이가 아니다.</p>

<h2>0 근처의 동작</h2>
<figure><img src="data:image/png;base64,{image_data}" alt="STDP windows and a close view around zero">
<figcaption>검은 점은 정확히 0에서 정한 기여 0이다. 양쪽 극한은 ±1이며, sine의 크기를 그대로 읽는 경우와 다르다.</figcaption></figure>
<table><tr><th>시차</th><th>A</th><th>L</th><th>위상에 대한 L 기울기</th></tr>{probe}</table>
<p>표는 float64의 단일 관계다. CUDA FP32에서도 ±10⁻⁷ 시차 및 1 rad 근처의 인접 float 값에서
읽기 강도가 ±1로 유지되는 것을 검증했다. epsilon을 분모에 넣어 방향을 약화하거나,
sign의 대체 기울기를 사용하는 처리는 없다.</p>
<p>일반적인 sign의 기울기는 0이고, 가우시안 포락선의 기울기는 −2δL/τ다.
따라서 원점 근처에서 읽기 강도는 유지되지만 위상 기울기는 0에 가까워진다.
기존 지수창과의 차이이며, 실제 학습에 미치는 효과는 이 연산 검증으로 판정하지 않았다.</p>

<h2>구현 경로</h2>
<p>상대위상 커널은 단위 복소 내적과 동치인 sin(δ)를 직접 계산한다.
뉴런별 재사용 커널은 sine·cosine을 한 번 만든 뒤 채널 쌍의 내적에 재사용한다.
내적의 상쇄가 수치적으로 민감한 경우에는 같은 sin(δ) 식으로 재평가한다.
이 평가 경로 선택은 A를 완만하게 만들거나 작은 시차를 제거하는 처리가 아니다.</p>
<p>전용 커널은 채널 쌍을 타일로 처리해 G에 합산하며 토큰×채널쌍 텐서를 저장하지 않는다.
전용 역전파에서 K,V 및 두 위상의 기울기를 계산한다.
토큰 수 T, Q 수 S, 헤드당 채널 D에 대해 O((T+S)D²)이며, S=T에서도 토큰 수에 선형이다.</p>

<h2>비용 비교: 배치 8</h2>
{table(b8)}
<p>같은 가우시안 창의 Torch 구현도 비교했다. 큰 채널 쌍 중간값을 저장하는 경우 역전파 메모리가 증가했으며,
전용 커널은 그 중간값을 저장하지 않는 경로로 측정했다.</p>

<h2>비용 비교: 학습에 쓰던 배치 128</h2>
{table(b128)}
<p>900토큰의 단위 위상 상대식은 {unit['median_ms']:.3f}ms,
기존 동적 지수창 분리형은 {old['median_ms']:.3f}ms로 {unit['median_ms']/old['median_ms']:.2f}배 차이가 났다.
기존 고정 위상 비용 기준 {fixed['median_ms']:.3f}ms와는 {unit['median_ms']/fixed['median_ms']:.2f}배 차이다.</p>
<p>같은 위상차 범위에서 sign(sin δ)=sign δ이므로 시차 부호 대조군은 수학적으로 같은 창이다.
그 커널은 {control['median_ms']:.3f}ms였다. 뉴런별 위상 재사용은 {precomputed['median_ms']:.3f}ms로,
이번 구현과 입력에서는 상대식보다 빨라지지 않았다.
위상 재사용 방법 전체의 성능 한계를 증명한 측정은 아니다.</p>
<p>고정 위상 기준 및 기존 동적 지수창은 새 가우시안 모델과 창이 다르다.
비용 비교만 가능하며 학습 성능의 우열을 뜻하지 않는다.
900토큰 ARC/Sudoku 전체 모델 학습, QKV·위상 projection, FFN, 두 RMSNorm,
RoPE, optimizer, 전체 재귀의 메모리·시간은 포함하지 않았다.</p>

<h2>검증과 재현</h2>
<p>CPU float64에서 문자 그대로의 복소 내적과 상대위상 식을 대조했다.
CUDA FP32에서 전용 커널의 출력·Q/K/V/위상 기울기를 직접 가우시안 창과 대조했다.
같은 위상, 반복된 위상, 0 근처, 인접 float, 다른 Q 개수, 연속하지 않은 입력,
관계별 단위화 이후의 합산, 다른 τ·gain, 공통 carrier 회전 불변성을 확인했다.
compiled CUDA 출력 최대 오차를 포함한 전체 최대 출력 오차는 {output_error:.3g},
최대 기울기 오차는 {gradient_error:.3g}였다.</p>
<p>{html.escape(b128['protocol']['gpu'])} · PyTorch {html.escape(b128['protocol']['torch'])} ·
FP32 · TF32 off · torch.compile(default, fullgraph=True).
시간은 초기 컴파일을 제외한 CUDA event 반복 측정 세 라운드의 중앙값이다.
입력을 제외한 추가 최대 메모리와 입력 포함 전체 최대 메모리를 함께 표시했다.</p>
<pre>python -m lt.benchmark_unit_phase_stdp --checks-only \\
  --out runs/unit_phase_gaussian_20261004/cuda_checks.json
python -m lt.benchmark_unit_phase_stdp --batch 128 --lengths 81,900 --reps 3 \\
  --out runs/unit_phase_gaussian_20261004/compiled_b128.json
python -m lt.report_unit_phase_stdp</pre>
<p>구현: lt/unit_phase_stdp.py · 측정: lt/benchmark_unit_phase_stdp.py · 원본: runs/unit_phase_gaussian_20261004/</p>
'''
    style = '''body{font:16px/1.65 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#182230}
    h1{font-size:31px;line-height:1.3}h2{margin-top:34px;font-size:22px}.meta,figcaption{color:#667085;font-size:13px}
    table{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #d0d5dd;text-align:left}
    th{background:#eef4ff}pre{background:#f2f4f7;padding:16px;overflow:auto;line-height:1.6}img{width:100%;height:auto}
    .lead{font-size:18px}'''
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" '
                    'content="width=device-width,initial-scale=1"><title>단위 위상 가우시안 STDP 측정</title>'
                    '<style>' + style + '</style><body>' + body + '</body></html>')
    print(f'SAVED {DEST}')


if __name__ == '__main__':
    main()
