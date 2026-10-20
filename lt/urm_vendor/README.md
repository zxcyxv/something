# URM 실행용 최소 사본

원본은 [UbiquantAI/URM](https://github.com/UbiquantAI/URM)의 commit
`c14e55f5f9227873617015cf60a239126b55adcd`다.

| 이 폴더 | 원본 경로 |
|---|---|
| common.py | models/common.py |
| layers.py | models/layers.py |
| sparse_embedding.py | models/sparse_embedding.py |
| urm.py | models/urm/urm.py |

변경은 package-relative import와 FlashAttention 미설치 시 PyTorch SDPA fallback이다.
URM 블록의 수식은 유지했다. 공통 하네스 연결, 내부 no_grad 제거, 2-layer/8-iteration,
ACT 비활성화, ConvSwiGLU에서 SwiGLU로 바꾸는 실험은 별도 `lt/urm_full_bptt.py`에 있다.

원본 파일 해시는 [IMPORT_PROVENANCE.json](../../docs/research/2026-10-04/reference/urm/IMPORT_PROVENANCE.json),
실행용 사본 해시는 [archive_manifest.json](../../docs/research/2026-10-04/archive_manifest.json)에 있다.
원본 `pretrain.py`, URM YAML, Sudoku 실행 설정도 같은 reference 폴더에 보관했다.
복제본에 들어 있던 ARC 데이터와 관계없는 모델/도구는 제외했다.

필요한 Python 패키지는 torch, pydantic, einops다. FlashAttention은 선택 사항이다.
