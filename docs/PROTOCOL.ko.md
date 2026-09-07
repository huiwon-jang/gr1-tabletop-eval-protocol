# GR-1 평가 리셋 프로토콜 (v4, 2026-09-05 확정)

GR-1 tabletop 평가에서 리셋 직후 물체가 **쓰러지는 중 / 낑김 / 테이블 소실** 상태로 시작되던
에피소드를 제거한다. 측정 결과 전체 에피소드의 16%+가 이런 상태였고, 그런 에피소드의 실평가
성공률은 CLEAN의 절반(24.1% vs 49.0%)이라 GR-1 성적의 ~4pp가 정책이 아닌 리셋 품질 몫이었다.

두 축으로 고친다 — 어느 쪽도 평가 중 상태 검사를 하지 않으므로 **결정론이 유지**된다
(같은 seed → 어느 노드에서든 같은 에피소드; 노드 간 692 pose 비트 동일 실측):

1. **`GR1_RESET_SETTLE_S=5.0`** — reset 끝에 분기 없는 순수 물리 5초 정착. 정책의 첫 관측이
   정지 장면이 된다. 팔 텔레포트 후 컨트롤러 캐시를 강제 갱신하므로 settle 중 손이 안 흐른다.
2. **`GR1_EPISODE_SEED_MANIFEST=<json>`** — 5초를 줘도 깨져 있는 슬롯(130/1,250)은 사전
   검증된 같은-lane 예비 seed로 치환. 판정은 오프라인에서 1회 동결(JSON이 유일한 진실),
   wrapper는 조회만 한다. 치환 시 `[seed-manifest]` 로그가 남아 사후 감사 가능.

**프로토콜은 모델 무관이다** — settle은 GR-1 tabletop env 안, 치환은 rollout wrapper 안에 있어서
백본이 무엇이든(wan2.2든 아니든) 이 레포의 GR-1 평가 경로를 타면 똑같이 적용된다. wan22용이 아닌
자기 sbatch 래퍼를 쓰는 경우엔 아래 두 env만 export하면 끝이고, 쳐내는 규칙 자체는 이 문서 아래
"교체 규칙 v4"와 `build_gr1_seed_manifest.py`에, 결과 치환표는 `manifest/gr1_episode_seed_manifest.json`에
모델과 무관하게 동결돼 있다.

## 기본 동작

**`scripts/run_gr1_eval.sh`가 두 값을 기본으로 켠다** — 표준 제출은 아무것도 바꿀 필요 없다.

```bash
# 표준 제출 (프로토콜 자동 적용)
MODEL_OUTPUT_DIR=<ckpt-dir> sbatch --array=0-23 \
    scripts/run_gr1_eval.sh

# 수정 전(legacy) 동작이 필요할 때만
GR1_RESET_SETTLE_S=0 GR1_EPISODE_SEED_MANIFEST=off MODEL_OUTPUT_DIR=... sbatch ...
```

⚠ **2026-09-05 이전 GR-1 결과는 구 프로토콜 측정치다.** 새 결과와 한 표에서 섞어 비교하지 말 것.
전환 효과 자체는 aug_sfc `_rq` A/B 쌍(같은 ckpt를 두 프로토콜로 평가)으로 읽는다.

## 교체 규칙 v4 (5.0s 정착 시점 판정 — episode가 실제 시작하는 상태)

교체: ① 집을 물체·넣을 곳 테이블 소실(낙하 >25cm) ② 넣을 곳 전도 >30°
③ 역할 무관 들림 +3cm↑(로봇/타 물체에 얹힘 = 낑김) ④ 병·캔류 집을 물체(wine, bottled water,
can, milk) 전도 >30° 또는 낙하 중(각속도 >0.5 rad/s).

교체하지 않음: 테이블 위 단순 밀림(17cm 밀린 basket도 3/3 성공) · 소스 용기 전도/이탈(집을
물체가 남으면 과제 가능) · 병·캔 외 물체(cup·과일·야채)의 전도.

예비 seed 기준은 더 엄격: 아무 임계 물체도 안 움직였고(strict) **+ 규칙 v4 통과**여야 배정.
현재 130/130이 같은 env lane의 strict-clean 예비(offset 10–34)이며 배정 후 재검증 통과.

## 임계값 변경 / 재측정

평가 시점 재유도는 설계상 금지 — 규칙을 바꾸려면 매니페스트를 재생성한다:

```bash
# 1) (물체/task가 바뀐 경우만) 측정 스윕: baseline offsets -1..9 + reserve 10..34
sbatch --array=0-23 tools/probe_gr1_reset_stability.py   # + EP_LIST/OUT_TAG

# 2) 임계값은 build_gr1_seed_manifest.py 상수 수정 후:
python tools/build_gr1_seed_manifest.py \
  --baseline <...>/_gr1_reset_stability_probe_v2 <...>/_gr1_reset_stability_probe_v2m1 \
  --reserve  <...>/_gr1_reset_stability_reserve_v3 \
  --out manifest/gr1_episode_seed_manifest.json
```

측정 원본(`_gr1_reset_stability_probe_v2*`, `_reserve_v3`)은 unified storage의
`output/` 아래에 보존돼 있다. 근거 수치·갤러리·규칙별 대표 영상은 진단/프로토콜 아티팩트 참고.
