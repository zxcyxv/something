import base64, sys
R = sys.argv[1]
img = lambda n: 'data:image/png;base64,' + base64.b64encode(open(f'{R}/fig/{n}.png', 'rb').read()).decode()
FIGS = [
 ('f1_extrap', '학습 지평 너머의 붕괴', '2천~1만 스텝의 평가마다 held-out 4,096행을 128블록까지 반복했다. 회색 띠가 학습한 8블록이다. 지평 안은 1만 스텝에 99.95%지만, 16블록 83%, 32블록 16%, 128블록 1%로 무너진다. 붕괴 지점은 평가마다 앞뒤로 흔들린다.'),
 ('f2_lightcone', '모델은 정확히 9칸 광원뿔을 본다', '출력 로짓의 분산 중 반경 r 창(2r+1칸)이 설명하는 몫(ANOVA 보정 R²). 반경 3에서 0.49였던 것이 반경 4에서 0.99로 뛴다. 4스텝 뒤의 칸은 정확히 반경 4 안의 입력으로 정해지므로, 모델은 첫 블록부터 필요한 만큼만 모은다. 외삽에서는 R²(반경 4)가 0.995에서 0.835로 내려가고, 반경 5~6이 조금 더 설명한다. 광원뿔 밖이나 창 안 잡음에 대한 의존이 생긴다는 뜻이다. 오른쪽: 512칸 진리표(문맥 평균 로짓의 부호)는 2~12블록에서 전부 맞고 128블록에서 87%다.'),
 ('f4_table', '진리표를 그대로 그린 것', '512개 창을 16×32 격자로 놓았다. 세로는 왼쪽 4칸(−4~−1), 가로는 오른쪽 5칸(0~+4)의 비트 패턴이다. 가운데는 8블록에서 문맥을 평균한 모델 로짓(빨강 = 1)이고 틀린 칸이 없다. 오른쪽 128블록에서는 66개 창이 틀린다(×). 틀린 창은 특정 열, 즉 오른쪽 5칸의 특정 패턴에 몰려 있다.'),
 ('f3_walsh', 'Walsh 스펙트럼: 모듈러 덧셈의 Fourier 분석에 대응하는 것', '불 함수의 Fourier 기저인 Walsh 기저(9비트 창 위 512개 부분집합)로 진리표를 분해했다. 정답 함수는 맨 왼쪽 칸(−4)에 의존하지 않아 0이 아닌 계수가 정확히 256개다. 모델은 8블록에서 이 256개 계수를 거의 그대로 재현한다(cos 0.991, 지지 밖 에너지 0.1%). 128블록에서는 cos가 0.752로 떨어진다. 오른쪽 막대에서 보듯 붕괴는 3·4차 상호작용 에너지가 줄고 2차가 커지는 방향으로 일어난다. 외삽에서 고차 XOR형 구조가 먼저 녹는다.'),
 ('f5_probe', '단계별 시뮬레이션은 보이지 않는다', '각 칸의 은닉 상태(32차원)에서 선형 탐침으로 t스텝 뒤의 행 x_t[i+d]를 읽은 정확도. 비트가 1로 치우쳐 있어 기저율은 55~60%다. 자기 입력(t 0, d 0)과 정답(t 4, d 0)은 2블록부터 100%이고, 중간 단계(t 2, 3)는 81~85%에 그친다. 3칸 규칙을 4번 합성하는 과정이 선형으로 드러나지 않는다. 512칸 표를 통째로 조회하는 쪽에 가깝다. 128블록에서는 정답 읽기도 86%로 떨어진다.'),
 ('f6_neurons', 'FFN 유닛 256개: 어디를 읽고 몇 차인가', '2블록 FFN 유닛마다 활성의 문맥 평균을 512개 창에 대해 구하고 Walsh 에너지를 나눴다. 왼쪽: 유닛 에너지 중 각 칸이 들어간 부분집합의 몫. 거의 모든 유닛이 가운데(−1~+1)를 중심으로 읽고, 정답이 무시하는 −4는 어떤 유닛도 5% 넘게 쓰지 않는다. 오른쪽: 유닛 하나하나는 1~2차가 주력이다(평균 2.5차). 출력의 3~5차 구조는 단일 유닛이 아니라 유닛들의 조합에서 나온다. 모듈러 덧셈처럼 몇 개 주파수로 깔끔하게 군집하지는 않는다.'),
 ('f7_units', '활성 폭이 가장 큰 유닛 8개', '진리표와 같은 배치. 세로 줄무늬가 많다. 즉 많은 유닛이 오른쪽 5칸 패턴을 검출하고 왼쪽에는 둔감하다. 제목의 오프셋은 유닛이 가장 많이 쓰는 세 칸이다.'),
 ('f9_progress', '학습 과정: 표는 750스텝에 완성되고, 그 뒤 외삽은 오르다 내려간다', '250스텝마다 남긴 재학습 체크포인트 41개(1번 런과 비트 단위로 같지는 않다). 512칸 진리표는 750스텝에 완성된다(Walsh cos 0.994). 그 뒤 학습은 두 가지를 한다. 2블록 정답률을 0%에서 99%로 올려 더 빨리 답하게 되고, 가중치 노름을 19.6에서 15.0으로 압축한다. 32블록 정답률은 3,250~4,750스텝에 약 40%로 정점을 찍고, 2블록 정답률이 포화된 뒤로는 10~20%로 내려간다. 32블록의 진리표 코사인도 0.97에서 0.89로 떨어진다. 지평 안이 포화된 뒤 외삽이 나빠지는 것은 스도쿠 v1.71에서 본 양상과 같다.'),
 ('f8_kernel', '읽기 커널: 1블록과 그 뒤는 다른 일을 한다', '칸 i의 query와 칸 j의 key의 평균 내적(Hebb 읽기 가중치)을 거리 j−i별로 그렸다. 1블록은 왼쪽 1~2칸을 음으로, 오른쪽 1~2칸을 양으로 읽어 이웃을 모은다. 2블록부터는 커널이 다른 모양(자기 −, +1 +, +3 −)으로 바뀌고 128블록까지 거의 그대로다. 외삽 붕괴는 읽기 위치가 변해서가 아니라, 같은 커널로 읽는 내용(상태와 W)이 표류해서 생긴다.'),
]
sections = '\n'.join(f'''<figure id="{k}"><figcaption><h3>{t}</h3><p>{c}</p></figcaption><div class="plate"><img src="{img(k)}" alt="{t}"></div></figure>''' for k, t, c in FIGS)
html = f'''<title>Rule 110 해부</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: single reading column, figures break out wider on white plates; header carries a live Rule 110 spacetime strip */
:root {{
  --ink: #17202a; --muted: #56616d; --paper: #f6f8f9; --plate: #ffffff; --rule: #d5dde2; --accent: #0f7a75;
  --f-body: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; --f-mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --ink: #e3e9ed; --muted: #9aa7b1; --paper: #11171c; --plate: #ffffff; --rule: #2b353d; --accent: #4fc3bb; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --ink: #e3e9ed; --muted: #9aa7b1; --paper: #11171c; --plate: #ffffff; --rule: #2b353d; --accent: #4fc3bb; color-scheme: dark; }}
body {{ background: var(--paper); color: var(--ink); font-family: var(--f-body); font-size: 15.5px; line-height: 1.7;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 0 64px; }}
.wrap {{ max-width: 1120px; margin: 0 auto; display: grid; gap: 40px; }}
header {{ display: grid; gap: 14px; padding-top: 28px; }}
canvas {{ width: 100%; height: 72px; display: block; border-radius: 4px; }}
.eyebrow {{ font-family: var(--f-mono); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font-size: clamp(28px, 4vw, 40px); line-height: 1.2; margin: 0; text-wrap: balance; }}
.lede {{ max-width: 68ch; color: var(--muted); margin: 0; }}
.findings {{ max-width: 76ch; display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; }}
.findings li {{ padding-left: 18px; position: relative; }}
.findings li::before {{ content: ""; position: absolute; left: 0; top: .72em; width: 8px; height: 8px; background: var(--accent); }}
.findings b {{ font-weight: 600; }}
.setup {{ font-family: var(--f-mono); font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); padding-block: 10px; display: flex; flex-wrap: wrap; gap: 6px 22px; }}
figure {{ margin: 0; display: grid; gap: 12px; }}
figcaption {{ max-width: 76ch; }}
h3 {{ margin: 0 0 4px; font-size: 18px; text-wrap: balance; }}
figcaption p {{ margin: 0; color: var(--muted); }}
.plate {{ background: var(--plate); border: 1px solid var(--rule); border-radius: 6px; padding: 10px; overflow-x: auto; }}
.plate img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
h2 {{ font-size: 21px; margin: 0; }}
.caveats {{ max-width: 76ch; color: var(--muted); display: grid; gap: 6px; }}
.caveats p {{ margin: 0; }}
</style>
<div class="wrap">
<header>
  <canvas id="ca" aria-label="Rule 110 시공간 그림" role="img"></canvas>
  <div class="eyebrow">LinearTuring · B-EMA (η 0.125) · 은닉 32 · 2026-10-15</div>
  <h1>Rule 110을 푸는 모델은 진리표를 외웠다</h1>
  <p class="lede">현재 모델(EMA 형태 B)을 은닉 32, 헤드 1로 줄여 36칸 행의 4스텝 뒤를 맞히게 학습한 뒤, 모듈러 덧셈 grokking 연구의 Fourier 분석을 불 함수용 Walsh 기저로 옮겨 해부했다. 대상은 1만 스텝 체크포인트다.</p>
  <ul class="findings">
    <li><b>정확한 9칸 조회표.</b> 출력은 첫 블록부터 반경 4 광원뿔로 99% 설명되고, 512개 창의 진리표가 2~12블록에서 전부 맞는다. Walsh 계수도 정답의 256개 지지 집합을 재현한다(cos 0.991).</li>
    <li><b>정답이 무시하는 칸은 모델도 무시한다.</b> 4스텝 Rule 110은 맨 왼쪽 칸에 의존하지 않는데, 이 칸을 5% 넘게 쓰는 FFN 유닛이 없다.</li>
    <li><b>3칸 규칙을 4번 돌리는 시뮬레이션은 보이지 않는다.</b> 중간 단계의 행이 은닉 상태에서 선형으로 읽히지 않는다.</li>
    <li><b>학습 중 외삽은 오르다 내려간다.</b> 표는 750스텝에 완성되고, 32블록 정답률은 4천 스텝 근처 약 40%가 정점이다. 그 뒤 학습이 빠른 풀이와 가중치 압축으로 가면서 10~20%로 떨어진다.</li>
    <li><b>외삽 붕괴는 표의 침식이다.</b> 읽기 커널은 2블록 뒤로 고정인데, 3·4차 상호작용이 녹고 2차가 커지며 128블록에서 66개 창이 틀린다.</li>
  </ul>
  <div class="setup"><span>36칸, 경계 0</span><span>목표 = 4스텝 뒤 행</span><span>8블록 1세그먼트, 매 블록 손실</span><span>AdamATan2 1e-3, wd 0.1</span><span>1만 스텝, 파라미터 28,900</span><span>평가: 32,768행 × 가운데 20칸</span></div>
</header>
{sections}
<section class="caveats">
  <h2>읽을 때 주의할 점</h2>
  <p>진리표와 Walsh 계수는 창 밖 문맥을 평균한 로짓에서 구했다. 문맥에 따른 흔들림은 R²로만 잡힌다.</p>
  <p>선형 탐침의 기저율은 55~60%다. 비선형으로는 중간 단계가 읽힐 수 있으므로 "단계별 시뮬레이션이 없다"는 선형 수준의 결론이다.</p>
  <p>시드 하나다. 정적 분석은 1번 런의 1만 스텝 체크포인트, 학습 과정 그림은 같은 설정의 재학습에서 나왔다.</p>
</section>
</div>
<script>
(function () {{
  const c = document.getElementById('ca'); const ctx = c.getContext('2d');
  function draw() {{
    const w = c.clientWidth, h = c.clientHeight, s = 4, dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.ceil(w / s), rows = Math.ceil(h / s); let row = new Array(n).fill(0); row[n - 1] = 1;
    const ink = getComputedStyle(document.body).color;
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = ink; ctx.globalAlpha = 0.85;
    const rule = [0, 1, 1, 1, 0, 1, 1, 0];
    for (let y = 0; y < rows; y++) {{
      for (let x = 0; x < n; x++) if (row[x]) ctx.fillRect(x * s, y * s, s - 0.6, s - 0.6);
      row = row.map((v, i) => rule[4 * (row[i - 1] || 0) + 2 * v + (row[i + 1] || 0)]);
    }}
  }}
  draw(); addEventListener('resize', draw);
}})();
</script>
'''
open(f'{R}/rule110_report.html', 'w').write(html); print(len(html))
