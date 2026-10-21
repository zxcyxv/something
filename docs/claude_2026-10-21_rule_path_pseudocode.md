# 규칙 시냅스(rule_path) B 런 수도코드와 인수인계 (2026-10-21)

다른 AI에게 넘기는 문서다. 코드: `lt/model.py`(`injection`, `forward`, `block_step`, `rule_path_step`, `bands_step`),
`lt/data.py`(`ArcContextDataset`, `ctx_pack`), `lt/train.py`. 설정: `configs/miniarc_pe_rule.json`(B), `configs/miniarc_pe.json`(A).

```python
# =====================================================================================================================
# [인수인계] 배경
# - 프로젝트: LT(재귀 블록 + STDP/bands 기억)로 ARC를 푼다. 목표는 URM식 "과제별 퍼즐 임베딩"보다 나은 편향을 찾는 것.
#   규칙: 손으로 만든 사전(고정 기저, 손 코드)은 금지. 색별 분해, C·X·Wᵀ 같은 ARC 형식 구조는 허용.
# - 이론적 동기(10-20): 시범 (입력 -> 출력)을 "델타 쓰기 + 소프트 문턱(희소 LTD)"으로 연산자에 쓰면, 쓰기의 고정점은
#   시범을 맞추는 연산자 중 원소가 가장 적은 것(lasso)이다. 희소 해 = 짧은 기술(MDL/단순성 사전). 단 "짧다"는 문턱이
#   걸리는 좌표에서 재므로, 그 좌표(구조 코드)가 과제 전체에서 학습되어 ARC 법칙이 그 좌표에서 희소해야 한다.
# - rule_path는 이것을 LT 안에 "손 배선 없이" 넣은 것이다.
# - 비교 구도: A = bands + 퍼즐 임베딩(앞 토큰). B = A + rule_path(+ 역할·쌍 임베딩) + R0를 퍼즐 임베딩에서 만듦.
#   둘 다 같은 에피소드(시범들 + 질의 입력을 문맥으로)를 받는다. A도 문맥을 본다(bands 메모리로).
# =====================================================================================================================

# ---------------------------------------------------------------- 데이터 (Mini-ARC, data/miniarc_pairs.npz)
# 149과제, 격자 <= 5x5. 과제마다 시범 쌍 2-8개는 학습, 시험 쌍 1개는 평가(학습에서 제외) = 전환적(URM과 같음).
# 증강판 v = 0..100: v=0 원본, v>=1은 (과제, v)마다 고정된 D4 변환 + 색 1..9 순열(색 0 고정).
# 퍼즐 ID = 과제 * 101 + v. 평가는 v = 0.
def make_episode(task, rng):
    v = rng.integers(101)
    pairs = trainable_pairs(task)                    # 시범 쌍(시험 쌍 제외)
    query = rng.choice(pairs); demos = shuffle(pairs - query)[:8]
    k, perm = fixed_version(task, v)                 # 증강
    grids = [transform(g, k, perm) for g in demos + [query]]
    # 격자마다 6x6 슬롯(5x5 + EOS 행·열). 슬롯 순서: (시범입력, 시범출력) x 8, 질의입력, 질의출력
    # 패킹: 실제 칸(+EOS)만 한 줄로. 질의 출력 슬롯은 6x6 전부 넣고 입력은 "빈칸" 토큰(배경색 0과 다름).
    # 토큰마다: color(빈칸/색0..9/EOS), row(= 슬롯번호*6 + 격자 안 행), col, valid
    # 레이블: 질의 출력 슬롯의 색과 EOS만 손실에 들어감(나머지는 무시)
    return tokens, labels, puzzle_id = task * 101 + v

# ---------------------------------------------------------------- 첫 벡터 (LT.injection)
d, H, Dh = 512, 8, 64
def injection(tokens, puzzle_id):
    x = color_emb[tokens.color]                                  # [T, 512] 원래 있던 입력
    slot = tokens.row // 6
    role = demo_in / demo_out / query_in / query_out  (slot에서 정해짐)
    pair = slot // 2   (질의는 마지막 번호)
    x += role_emb[role] + pair_emb[pair]                         # [B 전용] 형식 정보. 값은 학습됨. A에는 없음(교란 요인)
    pe = puzzle_table[puzzle_id]                                 # [512] URM 퍼즐 임베딩, SignSGD로 학습(lr 1e-2, wd 0.1)
    return concat([pe], x)                                       # 앞 토큰 1개. 앞 토큰의 RoPE 각도 0, valid 1

# ---------------------------------------------------------------- 사전 시냅스 R0 (B 전용, rule_prior_pe)
def prior_synapse(pe):
    e = P @ pe                                                    # P: [32, 512] 학습
    return (M @ e).reshape(H, 2*Dh, Dh)                          # M: [8*128*64, 32] 0으로 시작. R0: 헤드마다 128x64

# ---------------------------------------------------------------- 바깥 루프 (lt.train + LT.forward)
# 레인(배치 64)마다 에피소드 하나를 8세그먼트 동안 붙잡는다. 옵티마이저 스텝 1번 = 세그먼트 1개.
# 세그먼트 = 블록 16개(같은 가중치를 16번 재귀 적용). 세그먼트 사이에는 h와 기억 상태를 detach해서 넘김.
# grad_blocks = 1: 각 세그먼트의 마지막 블록만 기울기를 받는다(앞 15블록은 no_grad). HRM식 1-step 기울기.
# 손실: 매 세그먼트, 질의 출력 칸의 교차엔트로피. AdamW lr 1e-4(예열 2000), wd 0.1. EMA 0.999(퍼즐 임베딩 포함).
def segment(h, mem_state, R_minus_R0, inj, R0):
    for i in range(16):
        with grad_enabled(i == 15):
            h, mem_state, R_minus_R0 = block(h, inj, mem_state, R_minus_R0, R0)
    logits = classifier(h[1:])                                   # 앞 토큰 제외
    return h, mem_state, R_minus_R0, logits

# ---------------------------------------------------------------- 블록 (LT.block_step)
def block(h, inj, mem_state, R_minus_R0, R0):
    u = h + scale * inj
    q, k, v = heads(Wq u), heads(Wk u), heads(Wv u)
    read, mem_state = bands_memory(q, k, v, mem_state)          # 기존 경로(A에도 있음). 아래 설명
    rp, R_minus_R0 = rule_path(u, R_minus_R0, R0)                # [B 전용] 규칙 시냅스
    h = rms_norm(u + out_proj(read) + rp)
    h = rms_norm(h + W_down(silu(W_g h) * (W_a h)))              # SwiGLU FFN
    return h, mem_state, R_minus_R0

# bands_memory (요약): 키·값의 복소 극점 흔적(주기 2, 6, 20, 64 블록인 대역 4개)을 블록마다 갱신하고, STDP식 시간차 창으로
# 짝지은 헤브 합 s = vᵀ k_mix + v_mixᵀ k~ 를 만들어 W <- (1-eta) W + eta s 로 누적, read = q~ (vᵀ k~ + W)ᵀ.
# q~, k~ 는 단위화 + 학습된 2D RoPE(각도 = theta_row*row + theta_col*col). 데이터에 선형이라 희소성은 못 만든다.
# [인수인계] 시간차 창(어느 pre->post 쌍을 짝지을지)은 bands에만 있고, 희소화(문턱)는 rule_path에만 있다. 둘이 연결 안 됨.

# ---------------------------------------------------------------- 규칙 시냅스 (LT.rule_path_step), 헤드마다, 레인마다
def rule_path(u, R_minus_R0, R0):                                # u: [T, 512], 상태 R - R0: [8, 128, 64]
    phi = theta_row * row + theta_col * col                      # 토큰마다 RoPE 각도(앞 토큰 0)
    psi = (cos phi, sin phi)                                     # [T, 64] 위치 위상
    # (1) 목표 A = rownorm(P V): 선형 어텐션(softmax 없음, 마스크 없음, 레인 전체)
    Q  = rope(unit(u @ W_rq.T), phi)                             # [T, 64]
    K  = rope(unit(u @ W_rk.T), phi) * valid                     # [T, 64]
    V  = concat(u @ W_rv.T, psi) * valid                         # [T, 128]  (내용 ; 위치)
    A  = rownorm(Q @ (K.T @ V))                                  # = rownorm(P V), P = Q Kᵀ. aₜ = 토큰 t가 가리키는 곳의 (내용, 위치)
    # (2) 시냅스 키와 쓰기 게이트
    Kh = unit(u @ W_key.T)                                       # [T, 64] 시냅스 키 κ. RoPE 없음
    w  = sigmoid(u @ W_write.T + b_write) * valid                # [T] (헤드마다). 초기 편향 -2
    # (3) 델타 쓰기
    R  = R0 + R_minus_R0                                         # [128, 64]
    E  = A - Kh @ R.T                                            # 오차 [T, 128]
    eta = sigmoid(eta_raw)                                       # 헤드마다, 초기 0.5
    R  = R + eta * (E.T @ diag(w) @ Kh) / (1 + sum(w))           # 게이트 가중 "오차 x 키" 외적의 평균
    # (4) 희소 LTD: R - R0 에 원소별 소프트 문턱
    theta = softplus(theta_raw)                                  # 헤드마다 스칼라, 초기 1e-3 (softplus는 양수화용일 뿐)
    D  = R - R0
    D  = sign(D) * max(abs(D) - theta, 0)
    R  = R0 + D
    # (5) 읽기 -> 잔차
    read = concat_heads(Kh @ R.T)                                # [T, 8*128 = 1024]
    rp   = read @ W_out.T                                        # [T, 512], W_out 0으로 시작
    return rp, D                                                 # D(= R - R0)는 다음 블록, 다음 세그먼트로 이어짐
# 의도: 시범 출력 칸은 색이 보이므로 어텐션으로 자기 짝 입력의 출처를 찾아 aₜ = (출처 내용, 출처 위치).
#       델타 쓰기가 "κ(답 없이도 알 수 있는 설명) -> a(답을 써서 찾은 포인터)"를 R에 쓴다.
#       질의 출력 칸은 빈칸이라 어텐션으로 출처를 못 찾으므로 Kh R ᵀ 로 읽어 출처를 얻는다.
# [인수인계] 이 의도는 강제되지 않는다: 짝(시범 입력 -> 출력)은 명시되지 않고 A의 어텐션으로만 간접적으로 생기며,
#   κ에 답 색이 섞이지 말라는 제약도 없고, 질의 칸도 게이트만큼 R에 쓴다.

# =====================================================================================================================
# [인수인계] 열린 질문 (이론)
# 1. 희소 LTD가 의미를 가지려면: (i) 과제마다 고정된 회귀 문제(키, 목표), (ii) 문턱이 걸리는 좌표에서 ARC 법칙이 희소하고
#    엉터리 대응은 조밀할 것 = 구조 코드, (iii) 시범이 희소 해는 정하고 조밀 해는 못 정하는 구간, (iv) 질의를 같은 코드로 읽음.
#    (ii)를 만드는 학습 신호가 문제다: grad_blocks = 1이면 키·값 투영(W_key, W_rv)에 "문턱 고정점을 통과한" 기울기가
#    마지막 블록 한 걸음분만 간다. 쓰기 반복을 펼쳐 역전파하는 쪽이 이론상 맞다.
# 2. 토큰별(활성화) 희소성은 반대 방향: 키가 희소 = 시범 하나가 좌표 몇 개만 건드림 = 칸 좌표처럼 빈 곳을 못 채움.
#    필요한 건 조밀한 키 + 희소한 연산자. 후보: 원소별 학습 축소 함수 + 쓰기 반복 펼침(LISTA, Gregor & LeCun 2010).
# 3. STDP 창(쌍 짓기)과 문턱(희소화)이 다른 경로에 있다. 합치는 안: bands의 W 누적에 델타 오차 + 소프트 문턱을 건다.
#    패킹에서 시범 입력-출력 시간차가 격자 크기마다 달라지는 문제는 확인 필요.
# 4. 갱신을 1 + Σw로 나누므로 게이트 편향의 절대 수준은 거의 무의미하다(모든 w를 같이 올려도 가중 평균은 그대로).
#    선택성은 게이트 가중치에서만 나온다. theta는 블록마다 걸리므로 한 블록 쓰기의 원소 크기와 비교해 정해야 한다.
# 5. 공정 비교: A에는 역할·쌍 임베딩이 없고 파라미터 수도 다르다.
# =====================================================================================================================
```
