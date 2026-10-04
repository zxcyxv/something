"""Create a standalone HTML/PNG comparison of exact phase-STDP kernels."""
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
ART = ROOT / 'runs/dynamic_phase_benchmark_20261004'
DEST = ROOT / 'docs/research/2026-10-04/dynamic_phase_900_benchmark.html'
NAMES = {
    'fixed_gemm': '고정 위상 GEMM — 기존 비용 기준',
    'direct_g': '직접 지수창 + 공유 G',
    'split_g': '분리 지수항 + 공유 G',
    'scan_q': '정렬 + Q별 누적합',
}


def rows_table(report):
    rows = ''
    for r in report['rows']:
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
    b128 = json.loads((ART / 'compiled_b128.json').read_text())
    checks = json.loads((ART / 'cuda_checks.json').read_text())
    methods = list(NAMES)
    index = {(r['tokens'], r['implementation']): r for r in b8['rows']}
    labels = ['Fixed GEMM', 'Direct exp / G', 'Split exp / G', 'Sorted scan / Q']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(methods))
    for offset, tokens, color in [(-.19, 81, '#667085'), (.19, 900, '#2563eb')]:
        timing = [index[(tokens, m)]['forward_backward']['median_ms'] for m in methods]
        memory = [index[(tokens, m)]['forward_backward']['extra_peak_mib'] for m in methods]
        axes[0].bar(x + offset, timing, .36, label=f'{tokens} tokens', color=color)
        axes[1].bar(x + offset, memory, .36, label=f'{tokens} tokens', color=color)
        for xx, yy in zip(x + offset, timing):
            axes[0].text(xx, yy * 1.13, f'{yy:.1f}', ha='center', fontsize=8)
        for xx, yy in zip(x + offset, memory):
            axes[1].text(xx, yy * 1.13, f'{yy:.0f}', ha='center', fontsize=8)
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis='y', alpha=.25)
        ax.legend()
    axes[0].set_ylabel('Forward + backward time (ms, logarithmic)')
    axes[0].set_title('All query tokens read the same exponential STDP rule')
    axes[1].set_ylabel('Extra peak allocated memory (MiB, logarithmic)')
    axes[1].set_title('Batch 8 / 8 heads / 104 channels / FP32')
    image_path = ART / 'comparison.png'
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    image_data = base64.b64encode(image_path.read_bytes()).decode()
    out_err = max(c['output_max_abs_error'] for c in checks['cuda_correctness'])
    grad_err = max(max(c['gradient_max_abs_errors'].values()) for c in checks['cuda_correctness'])
    s = index[(900, 'split_g')]['forward_backward']
    direct = index[(900, 'direct_g')]['forward_backward']
    scan = index[(900, 'scan_q')]['forward_backward']
    split_scale = s['median_ms'] / index[(81, 'split_g')]['forward_backward']['median_ms']
    scan_scale = scan['median_ms'] / index[(81, 'scan_q')]['forward_backward']['median_ms']
    large = {(r['tokens'], r['implementation']): r for r in b128['rows']}
    full_ratio = (large[(900, 'split_g')]['forward_backward']['median_ms'] /
                  large[(900, 'fixed_gemm')]['forward_backward']['median_ms'])
    logical_pair_gib = 128 * 8 * 900 * 104 * 104 * 4 / 2**30
    content = f'''
<p class="meta">2026-10-04 · 30×30 / 900-token exact implementation comparison</p>
<h1>상태 의존 위상: 900토큰에서의 구현 비용</h1>
<p class="lead"><strong>현재 측정에서는 지수항을 분리하면서 공유 G를 유지하는 방식이 학습용 후보로 가장 적합하다.</strong>
정렬·누적합으로 모든 Q를 직접 읽는 구현은 같은 규칙을 보존하지만, 900토큰에서 비용이 크게 늘었다.</p>

<h2>비교하는 규칙과 축</h2>
<p>기존 실험의 L(δ)=sign(δ) exp(−|δ|), τ=1 rad, 정확한 0에서는 0을 유지한다.
발화 폭, smoothing, Fourier 근사, 주기별 발화 합은 추가하지 않았다.
동적 위상은 각 토큰·채널마다 공급한다. 실수 Q와 signed real K,V 활동을 그대로 사용한다.</p>
<pre>G[i,j] = mean_t V[t,i] K[t,j] L(phiV[t,i] - phiK[t,j])
Y[s,i] = sum_j Q[s,j] G[i,j]</pre>
<p>T는 쓰기에 참여하는 토큰 수, S는 Q의 개수, D는 헤드당 채널 수다.
여기서는 S=T이며 H=8, D=104, hidden=832다.
STDP 선후관계는 채널 위상차로 정의하며, 정렬/누적합의 축은 한 토큰 안의 K 채널이다.</p>

<h2>계산량 비교</h2>
<table><tr><th>구현</th><th>핵심 연산량 (배치·헤드 생략)</th><th>S=T일 때</th><th>지수 계산 수</th></tr>
<tr><td>고정 위상 GEMM</td><td>O((T+S)D²)</td><td>O(TD²)</td><td>O(D²), 토큰 공유</td></tr>
<tr><td>직접 지수창 + 공유 G</td><td>O((T+S)D²)</td><td>O(TD²)</td><td>O(TD²)</td></tr>
<tr><td>분리 지수항 + 공유 G</td><td>O((T+S)D²)</td><td>O(TD²)</td><td>O(TD)</td></tr>
<tr><td>정렬 + Q별 누적합</td><td>O(TD log D + TSD)</td><td>O(TD log D + T²D)</td><td>O(TD)</td></tr></table>
<p>분리형은 선후 마스크를 직접 비교하므로 정렬이 필요 없다.
정렬형은 K 순서를 모든 Q가 공유하지만, Q 가중 누적합 자체는 각 Q마다 달라진다.
고정 위상 행은 기존 모델의 비용 기준이며 동적 모델과 표현력이 같은 대안은 아니다.</p>
<pre>L(p-k) = [k&lt;p] exp(-p) exp(k) - [k&gt;p] exp(p) exp(-k)</pre>
<p>900/81≈11.11이므로 선형 연산량의 증가율은 11.11배, 제곱 연산량은 약 123.46배다.
실측 학습 시간 증가율은 분리 G {split_scale:.2f}배, Q별 누적합 {scan_scale:.2f}배였다.
고정 비용·커널 실행 비용·GPU 이용률 때문에 이론적 증가율과 정확히 같지는 않다.</p>

<h2>실측: 동일 배치 8</h2>
{rows_table(b8)}
<p>900토큰의 분리 G는 순전파+역전파 {s['median_ms']:.3f}ms,
Q별 누적합은 {scan['median_ms']:.3f}ms로 이번 구현에서 {scan['median_ms']/s['median_ms']:.1f}배 차이가 났다.
직접 지수창 대비 분리 G의 추가 최대 메모리는 {direct['extra_peak_mib']:.1f} → {s['extra_peak_mib']:.1f}MiB였다.
torch.compile이 순전파를 융합하므로 직접 G도 큰 토큰×채널쌍 텐서를 항상 물리적으로 만들지는 않는다.
역전파 중간값을 저장/재계산하는 방식에서는 큰 차이가 관측됐다.</p>
<figure><img src="data:image/png;base64,{image_data}" alt="Runtime and extra peak GPU memory for 81 and 900 tokens"><figcaption>
FP32 · TF32 off · compilation excluded · all-Q read · isolated current write/read kernel</figcaption></figure>

<h2>현재 학습 배치 128에서도 확인</h2>
{rows_table(b128)}
<p>900토큰에서 동적 분리 G의 학습용 write/read 시간은 기존 고정 위상 기준의 {full_ratio:.2f}배였다.
이를 전체 모델 학습 시간의 배수로 해석할 수는 없다. QKV/위상 projection, FFN, RoPE,
두 RMSNorm, optimizer 및 재귀 블록의 전체 activation checkpointing은 이 측정에서 제외했다.</p>
<p>배치 128의 토큰×채널쌍 FP32 텐서 하나는 {logical_pair_gib:.2f}GiB다.
따라서 이를 통째로 저장하는 구현은 24GiB GPU에서 불가능하다.
분리·융합된 공유 G는 전체 pair 텐서를 저장하지 않는 계산 경로로 배치 128 / 900토큰에서도 완료됐다.</p>

<h2>검증 및 측정 조건</h2>
<p>CPU float64에서 출력, Q/K/V/위상 기울기, 동일 위상 처리, 고정 위상으로의 환원을 검증했다.
CUDA FP32 compiled 출력과 역전파도 81·900토큰에서 직접 계산과 대조했다.
최대 출력 오차는 {out_err:.3g}, 최대 기울기 오차는 {grad_err:.3g}였다.
FP32 계산 순서에 따른 오차이며 bitwise 동일성이나 학습 성능을 입증한 것은 아니다.</p>
<p>GPU: {html.escape(b8['protocol']['gpu'])}, PyTorch {html.escape(b8['protocol']['torch'])}.
torch.compile(default, fullgraph=True), FP32 및 TF32 off.
컴파일/초기 설정은 시간에서 제외하고 CUDA event 반복 측정의 세 라운드 중앙값을 사용했다.
메모리는 PyTorch allocated peak이며 입력을 제외한 추가 최대치와 입력 포함 최대치를 함께 표시했다.</p>
<p>정렬 구현은 Q를 32개씩 나눈다. 역전파에서는 chunk activation checkpoint로
큰 누적합을 재계산해서 메모리를 제한한다. gradient truncation은 사용하지 않는다.
맞춤 CUDA/Triton scan 최적화는 수행하지 않았으므로, 이번 시간 차이를 모든 정렬 알고리즘의 하한으로 볼 수는 없다.</p>

<h2>판단과 재현</h2>
<p><strong>900토큰을 고려하면 우선 구현할 기준은 “정확한 동적 지수창 + 분리 지수항 + 공유 G”다.</strong>
원래 STDP 창을 유지하고 토큰 수에 대한 선형 연산량을 보존한다.
위상 불연속의 학습 특성과 실제 모델 성능은 이 구현 비용 비교의 다음 문제다.</p>
<pre>python -m lt.benchmark_dynamic_phase_stdp --batch 8 --lengths 81,900
python -m lt.benchmark_dynamic_phase_stdp --batch 128 --lengths 81,900 \\
  --variants fixed_gemm,split_g --reps 3 --out runs/dynamic_phase_benchmark_20261004/compiled_b128.json
python -m lt.benchmark_dynamic_phase_stdp --verify-cuda-only \\
  --out runs/dynamic_phase_benchmark_20261004/cuda_checks.json
python -m lt.report_dynamic_phase_benchmark</pre>
<p>측정 원본 JSON과 검증 결과는 runs/dynamic_phase_benchmark_20261004에 저장했다.</p>
'''
    style = '''body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#182230}
    h1{font-size:32px;line-height:1.3}h2{margin-top:36px;font-size:22px}.meta,figcaption{color:#667085;font-size:13px}
    table{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0}th,td{padding:10px;border-bottom:1px solid #d0d5dd;text-align:left}
    th{background:#eef4ff}pre{background:#f2f4f7;padding:16px;overflow:auto;line-height:1.6}img{width:100%;height:auto}
    .lead{font-size:18px}a{color:#175cd3}'''
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" '
                    'content="width=device-width,initial-scale=1"><title>900토큰 동적 위상 STDP 비교</title>'
                    '<style>' + style + '</style><body>' + content + '</body></html>')
    print(f'SAVED {DEST}')


if __name__ == '__main__':
    main()
