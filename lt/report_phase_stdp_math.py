"""Export the mathematical research as a self-contained local HTML report."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'runs/phase_stdp_math_20261004'
DEST = ROOT / 'docs/research/2026-10-04/phase_stdp_math.html'


def equation(formula, caption=''):
    fig = plt.figure(figsize=(10, 0.45))
    fig.text(0, 0.5, '$' + formula + '$', fontsize=16, va='center')
    stream = io.BytesIO()
    fig.savefig(stream, format='svg', bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    uri = 'data:image/svg+xml;base64,' + base64.b64encode(stream.getvalue()).decode()
    return '<figure class="equation"><img src="' + uri + '" alt="' + html.escape(formula) + '">' + (
        '<figcaption>' + caption + '</figcaption>' if caption else '') + '</figure>'


def picture(name, caption):
    data = base64.b64encode((ART / name).read_bytes()).decode()
    return '<figure><img src="data:image/png;base64,' + data + '"><figcaption>' + caption + '</figcaption></figure>'


def source(url, name):
    return '<a href="' + url + '">' + name + '</a>'


def main():
    r = json.loads((ART / 'results.json').read_text())
    eq = equation
    mamba = source('https://arxiv.org/html/2603.15569v1', 'Mamba-3 §3.2, 식 8–10')
    ckda = source('https://arxiv.org/html/2609.24797v1', 'Complex KDA §2–3')
    scarpetta = source('https://proceedings.neurips.cc/paper_files/paper/2000/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf',
                       'Scarpetta·Li·Hertz, Spike-Timing-Dependent Learning for Oscillatory Networks, 식 6·9')
    phase2010 = source('https://www.frontiersin.org/journals/synaptic-neuroscience/articles/10.3389/fnsyn.2010.00032/full',
                       'Storage of Phase-Coded Patterns via STDP, 2010')
    luz = source('https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004878',
                 'Luz·Shamir, 2016, 식 9–11')
    recall = source('https://www.gatsby.ucl.ac.uk/~dayan/papers/lkdp05.pdf', 'Lengyel 외, Matching storage and recall, 식 3')
    uncertainty = source('https://www.gatsby.ucl.ac.uk/~dayan/papers/ld2006.pdf', 'Lengyel·Dayan, Uncertainty, phase and oscillatory hippocampal recall, §2')
    tpam = source('https://pmc.ncbi.nlm.nih.gov/articles/PMC6731666/', 'Frady·Sommer, Robust computation with rhythmic spike patterns, 2019')
    transformer = source('https://arxiv.org/html/2306.09827', 'Building Blocks for a Complex-Valued Transformer Architecture, §3.2')
    training_rows = ''
    for row in r['training']:
        for w in row['windows'][:2]:
            training_rows += '<tr>' + ''.join('<td>' + cell + '</td>' for cell in [
                'URM' if row['run'].startswith('urm_') else 'v1.7 현재 하네스 재현',
                f'{w["start_exclusive"]:,} → {w["end_inclusive"]:,}',
                f'{w["all_segment_loss"]:.5f}', f'{w["terminal_loss"]:.5f}',
                f'{100*w["terminal_cell_accuracy"]:.2f}%', f'{100*w["terminal_exact_accuracy"]:.2f}%']) + '</tr>'
    metrics_rows = ''
    chosen = [(16, 0), (32, .04), (64, .04), (32, .1)]
    for modes, jitter in chosen:
        m = next(m for m in r['kernel_metrics'] if m['tau'] == 1 and m['modes'] == modes and m['jitter'] == jitter)
        metrics_rows += '<tr>' + ''.join('<td>' + cell + '</td>' for cell in [
            str(modes), str(jitter), f'{m["rmse"]:.6f}', f'{100*m["wrong_sign_fraction"]:.2f}%',
            f'{m["small_lag_values"]["0.1"]:.4f}', f'{m["peak_phase"]:.4f}']) + '</tr>'
    certificate = next(c for c in r['sign_certificates'] if c['tau'] == 1 and c['jitter'] == .1 and c['modes'] == 32)
    summary = {k: v for k, v in r['checks'].items() if 'error' in k or 'gradient' in k or 'gradcheck' in k}
    content = f'''
<p class="meta">2026-10-04 · 수학적 형태에 대한 조사와 독립 수치 검증 · 학습 성능 검증과 구분</p>
<h1>상태 의존 복소 활동에서 STDP Fast Weight를 만드는 형태</h1>
<p class="lead"><strong>가장 자연스러운 기준식은 “발화 활동에 STDP 필터를 적용한 뒤 상관을 계산한다”이다.</strong>
복소수는 활동의 크기와 발화 위치를 표현하고, 창의 푸리에 계수는 복소 외적의 계수가 된다.
순수 sine 하나는 이 구조의 한 주파수 경우다. 좁은 시차의 중요도를 표현하려면 활동 파형의 시간 해상도까지 명시해야 한다.</p>

<h2>1. 우리가 원하는 바</h2>
<p>채널을 뉴런으로 취급한다. FFN이 만든 현재 상태에서 독립적인 복소 K와 V를 만들고,
그 위상이 나타내는 채널 간 선후관계로 강화·약화를 결정한다. 일관된 관계는 재귀 동안 남고,
방향이 흔들리는 관계는 가중 변화의 합에서 상쇄되게 한다. 읽기는 실수 Q로 한다.
토큰 축의 causal attention을 재귀 시차로 옮기는 것이 연구 목표는 아니다.</p>
<table><tr><th>축/값</th><th>의미</th></tr>
<tr><td>토큰 t</td><td>현재 문맥의 관측 행. 외적을 토큰에 대해 평균한다.</td></tr>
<tr><td>채널 i, j</td><td>각 head 안의 post(V), pre(K) 뉴런. 기존 설정은 8 heads × 104 channels = 832 hidden channels이다.</td></tr>
<tr><td>carrier 위상 θ</td><td>한 공통 주기 안의 연속적인 발화 시각. STDP의 시차는 이 축에서 정의한다.</td></tr>
<tr><td>재귀 r</td><td>상태와 활동을 다시 계산하고, 필요하면 빠른 가중치 M에 쓰는 축이다.</td></tr></table>
{eq(r'z^K_{r,t,j}=W^K_R h_{r,t}+iW^K_I h_{r,t},\quad z^V_{r,t,i}=W^V_R h_{r,t}+iW^V_I h_{r,t}')}
{eq(r'z=a e^{i\phi},\quad a=|z|\geq 0,\quad y_r=M_r q_r,\quad M_r=\rho M_{r-1}+\eta G_r')}
<p>위 식의 W 표기는 각 채널 성분을 생략한 것이다. Q는 기존 채널 좌표의 실수 벡터다.
실제 row-vector 구현에서는 Q @ Mᵀ이다. 새 블록을 구현할 때는 프로젝트의 두 post-residual RMSNorm 규칙을 그대로 적용한다.</p>
<p><strong>공통 carrier의 정확한 의미:</strong> 모든 뉴런에 같은 ω를 부여한 각 상태의 활동 파형을 전제한다.
K와 V에 공통 회전을 곱하면 외적에서 그 회전은 소거된다.
상태에 따라 φ가 변한다면, 전체 신호를 시간에 연속적으로 이었을 때의 순간 주파수까지 같다는 뜻은 아니다.</p>
{eq(r'(z^V e^{i\chi})(z^K e^{i\chi})^*=z^V(z^K)^*')}
<p>이 보고서는 φ를 <em>발화 지연 좌표</em>로 정의하고 φV−φK&gt;0을 LTP로 둔다.
물리 신호 exp(iωt+iφ)의 특정 위상 통과를 발화로 정의하면 실제 시차의 부호는 반대다.
회전 방향/부호 관례를 고정해야 하지만, 이 선택은 공통 주파수 조건을 바꾸지 않는다.</p>

<h2>2. 논문에서 가져올 수 있는 수식과 제외한 부분</h2>
<table><tr><th>원문</th><th>수학적 형태</th><th>이번 목표에 대한 판단</th></tr>
<tr><td>{mamba}</td><td>h′ = exp(ΔA) R h + 입력. 누적 회전을 입력·출력 좌표에 옮긴 표현과 동치.</td>
<td>감쇠와 회전을 구분하는 참고다. 이 전이는 이전 상태의 운반을 정의하며, 채널 쌍의 STDP 창은 정의하지 않는다. 전이식은 도입 후보에서 제외한다.</td></tr>
<tr><td>{ckda}</td><td>A = (I−βkkᵀ) Diag(α), α∈[−1,1], β∈[0,2]. 반사 조합으로 회전을 표현.</td>
<td>“Complex”는 실수 행렬의 복소 고유값을 뜻한다. 복소 K,V의 위상차 창은 아니다. 직접 사용할 수식은 이번 목표에서 제외한다.</td></tr>
<tr><td>{scarpetta}</td><td>STDP 창으로 활동의 시간 상관을 적분. 창의 Fourier transform과 복소 활동 외적이 연결된다.</td>
<td>가장 직접적인 출발점이다. 창을 붙이는 위치와 활동 파형의 주파수 성분을 함께 설명한다.</td></tr>
<tr><td>{phase2010}, {luz}</td><td>주기적 활동의 STDP 효과는 창의 적분과 해당 주파수 Fourier 성분에 의해 결정된다.</td>
<td>지수형 창을 사용했다고 해서 위상차 함수 자체도 지수가 되는 것은 아니라는 점을 뒷받침한다.</td></tr>
<tr><td>{uncertainty}</td><td>발화량, 평균 위상, 위상 집중도를 분리해 표현.</td>
<td>좁은 발화 사건과 퍼진 활동을 구별하는 참고다. 이 논문의 전체 추론 동역학은 도입하지 않는다.</td></tr>
<tr><td>{recall}</td><td>그 논문의 모델에서는 저장 규칙 Ω와 읽기 상호작용 w∂Ω/∂x가 대응한다.</td>
<td>STDP 저장 규칙을 정했다고 일반적인 읽기 규칙의 Hopfield 수렴성이 따라오는 것은 아니다. 우리 Gq의 수렴 정리로 인용할 수 없다.</td></tr>
<tr><td>{tpam}, {transformer}</td><td>복소 켤레 외적/내적이 위상차를 표현.</td>
<td>위상 코드와 복소 대수의 기반이다. 그 자체로 지수형 시차 창을 만들어 주지는 않는다.</td></tr></table>
<p>위 표의 적용/제외 판단은 우리 목표에 대한 해석이다. 아래 폐형식, 근사 및 오차 경계는 이 조사에서 직접 유도하고 수치적으로 확인했다.</p>

<h2>3. sine의 문제를 활동 파형에서 해석하기</h2>
<p>활동을 순수한 사인파 계수로 읽으면, 지수형 STDP 창을 정확히 적용해도 최종 위상차 함수는 sine·cosine 조합이다.
이 경우 sine은 지수의 서툰 근사가 아니라, 활동이 한 주파수만 갖는 데서 나온 결과다.
반대로 φ를 좁은 발화 사건의 위치로 해석하려면 그 파형의 고조파를 함께 표현해야 한다.
같은 뉴런 파형의 n번째 고조파는 nω다. 뉴런마다 기본 주파수를 다르게 설정하는 것과 구분된다.</p>
{eq(r's^V_{r,t,i}(\theta)=a^V_{r,t,i}p(\theta-\phi^V_{r,t,i}),\quad s^K_{r,t,j}(\theta)=a^K_{r,t,j}p(\theta-\phi^K_{r,t,j})')}
{eq(r'G_{r,ij}=\frac{1}{T}\sum_t\int_0^{2\pi}\!\int_0^{2\pi}s^V_{r,t,i}(\theta_V)L_{\rm per}(\theta_V-\theta_K)s^K_{r,t,j}(\theta_K)\,d\theta_V\,d\theta_K')}
<p>이 식이 한 상태에서 한 주기의 모든 spike-pair 기여를 합한 기준식이다.
실제 carrier를 시간에 따라 돌리지 않고도 계산할 수 있다. G는 이 주기에서 계산된 변화량/쓰기 연산자이며,
이전 재귀 상태를 STDP의 과거 발화로 취급하지 않는다.</p>

<h2>4. 지수형 STDP 창을 원 위에서 정의하기</h2>
{eq(r'L(x)=\mathrm{sgn}(x)e^{-a|x|},\quad a=1/\tau_\phi,\quad\tau_\phi=\omega\tau_{\rm time}')}
<p>위상으로만 표현하면 주기 번호는 사라진다. τ로 나누는 것은 감쇠 척도를 바꾸는 동작이며,
잃어버린 주기 번호를 복원하지 않는다. 여기서는 같은 주기를 반복하는 활동 사이의 모든 쌍을 합하는 의미를 명시적으로 선택한다.</p>
{eq(r'L_{\rm per}(\delta)=\sum_{m\in\mathbb{Z}}L(\delta+2\pi m)')}
{eq(r'L_{\rm per}(\delta)=\frac{e^{-a\delta}-e^{-a(2\pi-\delta)}}{1-e^{-2\pi a}}=\frac{\sinh[a(\pi-\delta)]}{\sinh(a\pi)},\quad 0<\delta<\pi')}
<p>음수 시차에는 홀수 확장을 적용한다. 0에서는 중간값 0을 택한다.
양쪽 극한은 ±1이다. 반주기 π에서는 앞/뒤의 쌍이 균형을 이루어 0이 된다.
이는 단순히 가장 가까운 한 쌍에 exp(−|δ|/τ)를 적용하는 모델과는 다르다.</p>
{eq(r'L_{\rm per}(\delta)=\sum_{n=1}^{\infty}b_n\sin(n\delta),\quad b_n=\frac{2n}{\pi(n^2+a^2)}')}
<p>또한 이 창은 대칭적인 주기 지수 커널의 위상 미분이다.</p>
{eq(r'E_{\rm per}(\delta)=\sum_{m\in\mathbb{Z}}e^{-a|\delta+2\pi m|},\quad L_{\rm per}=-\frac{1}{a}\frac{dE_{\rm per}}{d\delta}')}
<p>따라서 대칭적인 위상 근접성에서 방향성을 가진 STDP로 넘어가는 수학적 연결을 만들 수 있다.
이 연결만으로 모델 성능 우위나 Hopfield 에너지의 단조 감소를 주장하지는 않는다.</p>

<h2>5. 선형어텐션 외적으로 되돌리는 표현</h2>
<p>상대 시차의 표준편차를 σ로 둔 Gaussian 발화 오차를 전제한다.
각 뉴런의 독립 발화 파형 폭은 σ/√2다. STDP 창의 계수를 임의로 학습하는 대신
창과 발화 파형에서 다음 특징 사상을 유도한다.</p>
{eq(r'\Phi_n(z)=\sqrt{b_n}\,e^{-\sigma^2n^2/4}|z|\left(\frac{z}{|z|}\right)^n')}
{eq(r'G_r=\frac{1}{T}\,\mathrm{Im}\left[\sum_{t,n}\Phi_n(z^V_{r,t})\Phi_n(z^K_{r,t})^H\right]')}
<p>채널의 발화량은 고조파마다 |z|로 동일하게 유지한다. zⁿ을 그대로 쓰면 크기도 |z|ⁿ이 되어
다른 활동 모델이 된다. Φₙ은 공통 회전 χ에 대해 exp(inχ)만큼 회전하므로, 같은 n의 외적에서 carrier가 소거된다.</p>
<p>지금의 sine 모델은 n=1만 쓰는 특수한 형태다. 각 mode는 두 실수 외적으로 계산할 수 있고,
mode와 토큰을 합한 뒤 하나의 실수 Dv×Dk 행렬을 읽는다. 별도의 M을 mode마다 저장할 필요는 없다.</p>
<p>LTP/LTD의 진폭과 감쇠를 독립적으로 두는 일반화도 같은 틀에서 가능하다.
이때는 창의 복소 Fourier 계수를 한쪽 활동 특징에 적용하고 실수부를 읽는다.</p>
{eq(r'c_n=\frac{1}{2\pi}\left[\frac{A_+}{\tau_+^{-1}+in}-\frac{A_-}{\tau_-^{-1}-in}\right]')}
{eq(r'L_\sigma(\delta)=c_0+2\mathrm{Re}\sum_{n=1}^{\infty}c_n e^{-\sigma^2n^2/2}e^{in\delta}')}
<p>대칭 진폭과 동일 감쇠에서는 이 식이 앞의 허수부 읽기로 줄어든다.
일반화한 창에는 짝수 성분과 DC 성분이 생길 수 있으므로, 아래 홀수 창의 부호 보존·상쇄 조건을 그대로 적용할 수는 없다.</p>
<p>토큰 수 T에 대한 계산량은 선형이며, 표준 외적보다 고조파 개수 N만큼 계산량이 늘어난다.
정확한 시차 커널을 채널 쌍마다 직접 계산하고 토큰에 대해 합하는 방법 역시 O(T Dv Dk)이지만,
일반적인 두 projection의 GEMM으로 바로 환원되지는 않는다. 동적 위상에서는 창을 토큰 합 뒤로 꺼낼 수 없다.</p>
{eq(r'\mathrm{rank}(G_N)\leq\min(D_V,D_K,2NT)')}
<p>한 토큰에서는 N=1,4,8,16의 행렬 rank가 각각 2,8,16,32임을 수치적으로 확인했다.
그러나 현재 head 채널 104, 토큰 81에서는 N=1의 전체 토큰 합도 full rank가 가능하다.
따라서 sine 창의 시차 표현 한계와 전체 행렬의 rank 부족을 같은 문제로 볼 수 없다.</p>
{picture('window_shapes.png', '검정: 정확한 주기 지수창. 단일 sine은 작은 시차에서 넓게 약해진다. 유한 해상도는 좁은 0 주변을 부드럽게 만든다.')}

<h2>6. 근사로 인한 부호 뒤집힘을 통제하기</h2>
<p>무감쇠 Fourier 급수를 유한하게 자르면 Gibbs 진동이 생긴다. σ가 작아도 N을 충분히 늘리지 않으면
양수 시차에서 음수 값이 나타날 수 있다. 아래 수치는 τφ=1이며, RMSE의 기준은 각각 정확한 창 또는 해당 σ의 매끄러운 창이다.</p>
<table><tr><th>N</th><th>상대 시차 σ</th><th>RMSE</th><th>양수 시차 부호 오류</th><th>L(0.1)</th><th>최대점 위상</th></tr>{metrics_rows}</table>
<p>부호 오류의 수치 검사는 0.01&lt;δ&lt;π−0.02 구간에서 −10⁻⁸보다 작은 값을 센 것이다.
이 측정만으로 부호 보존을 증명하지 않으므로 별도의 충분조건을 유도했다.</p>
{eq(r'L_\sigma(\delta)\geq \frac{a}{\sinh(a\pi)}e^{-\sigma^2/2}\sin\delta,\quad 0<\delta<\pi')}
{eq(r'\frac{|L_\sigma-L_{\sigma,N}|}{|\sin\delta|}\leq B_N=\frac{\sqrt{2/\pi}}{\sigma}\,\mathrm{erfc}\left(\frac{N\sigma}{\sqrt{2}}\right)')}
<p>첫 경계는 초기 창이 (a/sinh(aπ))sinδ 이상이고, 홀수 Gaussian convolution이 구간 (0,π)의
양의 열방정식 연산자라는 사실에서 나온다. 두 번째 경계는 |sin(nδ)|≤n|sinδ|와 Gaussian 꼬리 적분에서 나온다.</p>
<p><strong>Bₙ이 첫 식의 계수보다 작으면 모든 0&lt;δ&lt;π에서 유한 근사의 부호가 보존된다.</strong>
τφ=1, σ=0.1, N=32에서는 Bₙ={certificate['omitted_tail_relative_to_sine_bound']:.5f},
양의 하한 계수={certificate['positive_ratio_lower_bound']:.5f}다.
이 설정은 표본 검사와 별개로 부호 보존의 충분조건을 만족한다.
σ=0.04라면 같은 충분조건은 N≥72에서 만족한다. 이 수치는 최소 계산량을 증명한 값은 아니다.</p>
<p>따라서 다음 구현의 검증용 출발점은 <strong>τφ=1, σ=0.1, N=32</strong>로 구체화할 수 있다.
σ는 감쇠 τ와 다른 값이다. 발화 시각의 해상도를 나타내며, 0 주변의 부드러운 구간 폭을 정한다.</p>

<h2>7. 다른 후보를 비교하고 정리한 결과</h2>
<p>동일한 유한 양수 파형 pₘ(θ)∝cos²ᵐ(θ/2)를 모든 뉴런에 사용하면,
STDP 결과 자체가 정확히 m개의 고조파만 가진다. 계수는 아래처럼 정해진다.</p>
    {eq(r'\widetilde b_n=b_n\left[\frac{\binom{2m}{m-n}}{\binom{2m}{m}}\right]^2,\quad 1\leq n\leq m')}
<p>이 구성은 대칭이고 원 위의 거리와 함께 작아지는 양수 활동 파형을 사용한다.
활동 상관의 양수·단봉 커널을 홀수 STDP에 적용하므로 양수 시차의 부호를 보존한다.
추가 hard cutoff가 없다. 대신 m이 작으면 발화 파형이 넓어서 가까운 시차의 구별도 거칠다.</p>
<p>대칭 지수 커널을 s=cos²(δ/2)의 양수 거듭제곱 급수로 근사한 뒤 미분하는 방법도 확인했다.
유한 차수에서도 부호와 과대평가 방지는 보존되지만, 작은 시차를 구별하는 데 필요한 차수가 크게 늘어난다.</p>
{picture('sign_preserving_windows.png', '두 후보 모두 수치 검사에서 부호가 보존됐지만, 낮은 차수의 창은 여전히 넓다. 간결한 식과 시간 해상도는 별개의 평가 항목이다.')}
<p>복소 unit phasor의 chord distance에 지수 커널을 두고 미분하는 대안도 검증했다.
atan2 branch 없이 매끄러운 창을 쓸 수 있지만, 시차를 φ/ω로 보는 원래의 지수 STDP와 거리가 달라진다.
그 변경이 현재 목표에 필수적이지 않아 주 후보에서 제외했다.</p>

<h2>8. 0 시차와 누적 상쇄에서 남는 선택</h2>
<p><strong>연속적인 홀수 창은 반드시 L(0)=0이다.</strong> 고조파를 늘려도 이 조건을 없앨 수 없다.
균형 STDP를 적용한 완전히 같은 활동 파형의 순 LTP/LTD 역시 0이다.
정확한 0에서도 양의 읽기를 원한다면, LTP/LTD 진폭을 다르게 두거나 짝수 성분을 읽는 선택이 필요하다.
그 경우 +δ와 −δ의 변화량이 정확히 상쇄되는 조건도 달라진다.</p>
<p>G를 ΔM으로 쓰면, G=0은 더 이상 쓰지 않는다는 뜻이다. ρ=1이면 이전에 쌓인 M은 남는다.
G 자체를 현재 가중치로 읽으면 이 경우의 읽기 연산은 0이다. 현재 연구에서 두 의미를 구분해야 한다.</p>
<p>대칭 STDP 창을 사용하더라도 독립적인 K와 V의 G가 반대칭 행렬일 필요는 없다.
K=V로 묶었을 때에는 Gᵀ=−G가 되며 xᵀGx=0이다. 이 특수한 연산자와 고전 Hopfield의 대칭 기억 행렬은 다르다.
현재의 독립 K,V와 학습된 Q/출력 projection에 고전 Hopfield의 에너지 증명을 그대로 옮길 수 없다.</p>
<p>일관된 관계와 흔들리는 관계의 단순 수치 검증도 했다. 128회 동안 +0.2와 −0.2가 동일한 크기로 번갈아 들어오면
ρ=1에서 누적 기여가 정확히 0이었다. +0.1과 −0.5처럼 시차의 크기가 다르면 잔여가 생겼고,
같은 시차에서 활동량만 1.2와 0.8로 바꾸어도 잔여가 생겼다.</p>
{eq(r'\mathbb{E}\left[a^V a^K L_\sigma(\phi^V-\phi^K)\right]=0')}
<p>이 식이 평균적인 상쇄의 조건이다. FFN이 의미 없는 관계를 이러한 <em>가중된</em> 방향 변화로 표현해야 한다.
단순한 부호 변화만으로 자동 삭제되는 것은 아니다. 또한 ρ=1에서 새로운 변화량의 합이 0이라는 사실은
이미 존재하는 M의 잘못된 값까지 지운다는 뜻은 아니다. ρ&lt;1에서는 그 값이 감쇠하지만, 고정된 망각 척도가 추가된다.</p>
{picture('resolution_and_consistency.png', '기여 상쇄의 확인용 시뮬레이션. 위상 스케줄은 직접 공급했으며 FFN을 학습시킨 성능 실험은 아니다.')}

<h2>9. 수치 검증과 재현</h2>
<p>반복 spike-pair 합과 폐형식, Fourier 계수의 수치 적분, 직접 채널 쌍 커널과 복소 특징 외적,
공통 carrier 불변성, 역할 역전, 활동 파형의 직접 convolution, 자동미분을 각각 확인했다.</p>
<pre>{html.escape(json.dumps(summary, indent=2))}</pre>
<p>Φ(z)의 위상은 z=0에서 정의되지 않는다. 검증 코드에는 고조파의 매끄러운 확장
zⁿ/(|z|²+ε²)^((n−1)/2)를 별도로 두었다. 첫 mode는 정확히 z이고, 매우 작은 크기의 높은 mode가 약해진다.
이 선택이 작은 활동의 파형을 조금 바꾼다는 점을 기록했으며, 일반 활동의 gradcheck와 0 활동의 유한 gradient를 확인했다.</p>
<p>부호 보존의 충분조건은 정확한 실수 연산에 대한 결과다.
별도 FP32 검사에서는 20,001개 시차에서 최대 창 오차가 약 3.6×10⁻⁷,
공통 회전 전후 최대 차이가 약 6.6×10⁻⁷이었다.
회전 후 π에서 10⁻⁷ rad 떨어진 한 점에서 작은 음수 값이 나왔다.
이는 수치 정밀도의 경계이며, 구현에서는 0·π에 극도로 가까운 시차의 동률 처리를 별도로 정해야 한다.</p>
<pre>python -m lt.research_phase_stdp_math
python -m lt.report_phase_stdp_math</pre>
<p>검증 결과: runs/phase_stdp_math_20261004/results.json.
이번 검증의 범위는 수식의 일관성, 시간 해상도, 부호와 수치 성질이다.
STDP 모델의 스도쿠 성능이나 재귀 외삽 개선은 아직 이 후보로 측정하지 않았다.</p>

<h2>10. 함께 확인한 학습 결과</h2>
<p>URM SwiGLU loops=16은 요청된 6,000스텝 checkpoint 후 6,025에서 종료됐고,
v1.7 현재 하네스 재현은 6,000에서 정상 종료됐다. 연구 중 두 프로세스와 checkpoint, queue 종료 상태를 확인했다.
아래는 둘 다 6,000스텝까지만 집계했다. 전체 loss는 모든 segment 평균,
셀/완판 정확도는 실제 count가 있는 마지막 segment 16의 train 지표다.</p>
<table><tr><th>실험</th><th>구간</th><th>전체 loss</th><th>segment 16 loss</th><th>셀 정확도</th><th>완판 정확도</th></tr>{training_rows}</table>
<p>두 모델 모두 앞선 구간보다 평균 loss가 내려갔다. URM의 초기 학습 곡선이 더 좋았다.
같은 데이터·집계 하네스이지만 layer 수, 정규화와 어텐션 등 아키텍처가 달라 단일 원인의 비교는 아니다.</p>
{picture('training_monitor.png', '종료한 두 학습의 같은 구간 평균. train 정확도이며 eval 정확도와 구분한다.')}
'''
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>상태 의존 위상 STDP: 수식 조사</title><style>
body{margin:0;background:#f7f7f4;color:#202628;font:17px/1.8 system-ui,sans-serif}main{max-width:980px;margin:40px auto;padding:38px;background:white}
h1{font-size:30px;line-height:1.4}h2{margin-top:42px;font-size:23px}.lead{padding:18px 22px;background:#eef6f2;border-left:4px solid #4a826b}
.meta,figcaption{font-size:14px;color:#586566}a{color:#146860}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.65;margin:22px 0}
th,td{text-align:left;vertical-align:top;border-bottom:1px solid #d9dfdb;padding:10px}th{background:#f0f4f1}figure{margin:22px 0}figure img{max-width:100%;height:auto}
.equation{overflow-x:auto;background:#fafafa;padding:8px 14px}.equation img{max-height:110px}pre{padding:18px;background:#f4f5f4;overflow:auto;font-size:13px;line-height:1.5}
@media(max-width:700px){main{margin:0;padding:22px}body{font-size:16px}table{font-size:12px}th,td{padding:7px}}
@media print{body{background:white}main{margin:0;padding:0}figure,table{break-inside:avoid}h2{break-after:avoid}}
</style><main>''' + content + '</main></html>'
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(document)
    print(str(DEST))


if __name__ == '__main__':
    main()
