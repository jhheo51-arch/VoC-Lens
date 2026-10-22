"""상담 통화 전사본(call_youtube_check.py 결과)에서 발화별 각성·화자 라벨링 페이지(HTML 한 장)를 만든다.

음향 각성 규칙과 화자 분리·역할 판정을 사람 라벨로 확인하기 위한 페이지다. 발화마다 해당 구간의 음성을 재생하고
  1. 누가 말했나 (상담원 / 고객 / 해설 / 섞임·불분명). 롤플레이 영상에는 통화 밖 해설 목소리가 섞여 있다
  2. 목소리의 각성 (차분 / 보통 / 흥분, 판단 불가)
를 고른다. 모델 출력(각성 점수, 추정 역할)은 보여 주지 않는다(라벨이 모델에 끌려가지 않게).
진행 상황은 브라우저에 저장되고 "결과 내려받기"로 [{"unit_id", "speaker", "arousal", "memo"}] JSON을 받는다.
unit_id는 상담 파이프라인과 같은 "call:<영상 ID>#<발화 순서>"라서 결과와 바로 맞춰 볼 수 있다.

    python scripts/make_arousal_labeling_page.py                      # data/external/call_youtube/*.json 전부
    python scripts/make_arousal_labeling_page.py --out data/labeling/call_arousal.html
    # 전체 대신 무작위 50발화만 (앞뒤 발화는 문맥으로 함께 보여 준다)
    python scripts/make_arousal_labeling_page.py --sample 50 --out data/labeling/call_arousal_50.html

전사 원문이 들어 있고 음성은 data/external/call_youtube/의 WAV를 상대 경로로 재생하므로 data/ 아래에 만들고 저장소에 올리지 않는다.
"""

import argparse
import html
import json
import os
import random
from pathlib import Path

from voclens.config import REPO_ROOT

SOURCE = REPO_ROOT / "data" / "external" / "call_youtube"


def items_from(paths: list[Path], out: Path) -> list[dict]:
    items = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        wav = Path(data["audio"])
        src = Path(os.path.relpath(wav, out.parent)).as_posix()
        turns = data["turns"]
        for i, t in enumerate(turns):
            items.append({"unit_id": f"call:{data['call_id']}#{i}", "call": data["call_id"], "src": src,
                          "start": round(t["start"], 2), "end": round(t["end"], 2), "text": t["text"],
                          "before": turns[i - 1]["text"] if i else "", "after": turns[i + 1]["text"] if i + 1 < len(turns) else ""})
    return items


def sample_items(items: list[dict], n: int | None, seed: int) -> list[dict]:
    """n을 주면 무작위로 n개만 고르고, 통화와 발화 순서대로 돌려놓는다."""
    if not n or n >= len(items):
        return items
    keep = set(map(id, random.Random(seed).sample(items, n)))
    return [it for it in items if id(it) in keep]


TEMPLATE = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f4f6f9; --card:#fff; --ink:#131a24; --muted:#6b7585; --rule:#dce2ea; --blue:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1218; --card:#151c26; --ink:#edf1f6; --muted:#8b96a6; --rule:#263040; --blue:#3987e5; } }
* { box-sizing: border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--ink); font:15px/1.6 "Malgun Gothic", system-ui, sans-serif; word-break:keep-all; }
main { max-width:860px; margin:0 auto; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; justify-content:space-between; margin-bottom:12px; }
h1 { font-size:18px; margin:0; }
.bar { height:6px; background:var(--rule); border-radius:3px; overflow:hidden; margin-bottom:16px; }
.bar > i { display:block; height:100%; background:var(--blue); width:0; }
.card { background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:16px; margin-bottom:12px; }
.small { font-size:13px; color:var(--muted); }
.ctx { color:var(--muted); font-size:14px; margin:2px 0; }
.clause { font-size:18px; font-weight:600; margin:6px 0; }
fieldset { border:1px solid var(--rule); border-radius:8px; margin:0 0 10px; padding:8px 10px; }
legend { font-size:13px; color:var(--muted); padding:0 4px; }
fieldset label { display:inline-flex; align-items:center; gap:4px; margin:3px 14px 3px 0; cursor:pointer; }
textarea { width:100%; min-height:48px; border:1px solid var(--rule); border-radius:8px; background:var(--card); color:var(--ink); padding:8px; font:inherit; }
nav { display:flex; flex-wrap:wrap; gap:8px; }
button { font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--rule); background:var(--card); color:var(--ink); cursor:pointer; }
button.primary { background:var(--blue); border-color:var(--blue); color:#fff; }
.warn { color:#d0453f; font-size:13px; min-height:1.4em; }
</style>
<main>
<header><h1>__TITLE__</h1><span class="small" id="pos"></span></header>
<div class="bar"><i id="prog"></i></div>
<div class="card">
  <div class="small" id="where"></div>
  <p class="ctx" id="before"></p>
  <p class="clause" id="clause"></p>
  <p class="ctx" id="after"></p>
  <nav><button id="play" class="primary">▶ 이 발화 듣기 (Space)</button><button id="playctx">앞뒤 포함 듣기</button></nav>
  <div class="small" style="margin-top:6px">글자는 STT 결과라 틀릴 수 있습니다. 라벨은 소리를 듣고 붙여 주세요.</div>
</div>
<div class="card" id="choices">
  <fieldset><legend>누가 말했나</legend>
    <label><input type="radio" name="speaker" value="agent"> 상담원</label>
    <label><input type="radio" name="speaker" value="customer"> 고객</label>
    <label><input type="radio" name="speaker" value="narrator"> 해설(통화 밖 목소리)</label>
    <label><input type="radio" name="speaker" value="mixed"> 섞임·불분명</label></fieldset>
  <fieldset><legend>목소리의 각성 (내용이 아니라 목소리만)</legend>
    <label><input type="radio" name="arousal" value="calm"> 차분 (1)</label>
    <label><input type="radio" name="arousal" value="normal"> 보통 (2)</label>
    <label><input type="radio" name="arousal" value="excited"> 흥분 (3)</label>
    <label><input type="radio" name="arousal" value="unknown"> 판단 불가 (4)</label></fieldset>
  <textarea id="memo" placeholder="메모 (애매한 이유 등, 선택)"></textarea>
</div>
<div class="warn" id="warn"></div>
<nav>
  <button id="prev">← 이전</button><button id="next" class="primary">다음 →</button>
  <button id="download">결과 내려받기</button>
  <button id="reset">처음부터 다시</button>
</nav>
<p class="small">기준: 같은 통화 안에서 그 사람의 평소 목소리보다 크고 높고 빠르고 날이 서 있으면 흥분, 평소 같으면 보통, 평소보다 가라앉아 있으면 차분.
말의 내용(욕설·불만 표현)이 아니라 목소리만 봅니다. 진행 상황은 이 브라우저에 저장됩니다. 단축키: ← → 이동, Space 듣기, 1~4 각성.
음성은 data/external/call_youtube/의 WAV를 재생하므로 이 파일을 data/labeling/에서 그대로 여세요.</p>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "voclens-arousal-" + ITEMS.length + "-" + ITEMS[0].unit_id;
let state = {}; let i = 0;
try { const s = JSON.parse(localStorage.getItem(KEY) || "{}"); state = s.state || {}; i = s.i || 0; } catch (e) {}
const audio = new Audio(); let stopAt = null;
audio.addEventListener('timeupdate', () => { if (stopAt !== null && audio.currentTime >= stopAt) { audio.pause(); stopAt = null; } });
function play(a, b) {
  const it = ITEMS[i];
  if (!audio.src.endsWith(it.src)) audio.src = it.src;
  const go = () => { audio.currentTime = Math.max(0, a); stopAt = b; audio.play(); };
  if (audio.readyState >= 1) go(); else audio.addEventListener('loadedmetadata', go, {once: true});
}
const pick = n => (document.querySelector(`input[name=${n}]:checked`) || {}).value || null;
function save() { try { localStorage.setItem(KEY, JSON.stringify({state, i})); } catch (e) {} }
function collect() { state[ITEMS[i].unit_id] = {speaker: pick('speaker'), arousal: pick('arousal'), memo: document.getElementById('memo').value}; }
function render() {
  const it = ITEMS[i], s = state[it.unit_id] || {};
  document.getElementById('where').textContent = `${it.call} · ${it.start.toFixed(1)}~${it.end.toFixed(1)}초`;
  document.getElementById('before').textContent = it.before ? "앞: " + it.before : "";
  document.getElementById('clause').textContent = it.text;
  document.getElementById('after').textContent = it.after ? "뒤: " + it.after : "";
  document.querySelectorAll('#choices input').forEach(b => b.checked = (b.name === 'speaker' ? s.speaker : s.arousal) === b.value);
  document.getElementById('memo').value = s.memo || "";
  const done = ITEMS.filter(x => (state[x.unit_id] || {}).arousal).length;
  document.getElementById('pos').textContent = `${i + 1} / ${ITEMS.length} · 완료 ${done}`;
  document.getElementById('prog').style.width = (done / ITEMS.length * 100) + "%";
  document.getElementById('warn').textContent = "";
}
function go(d) {
  collect();
  const s = state[ITEMS[i].unit_id];
  if (d > 0 && (!s.arousal || !s.speaker)) { document.getElementById('warn').textContent = "화자와 각성을 모두 고르세요."; return; }
  audio.pause(); i = Math.min(ITEMS.length - 1, Math.max(0, i + d)); save(); render();
}
document.getElementById('play').onclick = () => play(ITEMS[i].start, ITEMS[i].end);
document.getElementById('playctx').onclick = () => play(ITEMS[i].start - 3, ITEMS[i].end + 2);
document.getElementById('prev').onclick = () => go(-1);
document.getElementById('next').onclick = () => go(1);
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowRight') go(1);
  if (e.key === 'ArrowLeft') go(-1);
  if (e.key === ' ') { e.preventDefault(); play(ITEMS[i].start, ITEMS[i].end); }
  const v = {"1": "calm", "2": "normal", "3": "excited", "4": "unknown"}[e.key];
  if (v) document.querySelector(`input[name=arousal][value=${v}]`).checked = true;
});
document.getElementById('download').onclick = () => {
  collect(); save();
  const out = ITEMS.filter(x => (state[x.unit_id] || {}).arousal).map(x => ({unit_id: x.unit_id, ...state[x.unit_id]}));
  const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  const a = Object.assign(document.createElement('a'), {href: url, download: "arousal_labels.json"}); a.click();
  URL.revokeObjectURL(url);
};
document.getElementById('reset').onclick = () => {
  if (!confirm("고른 라벨과 메모를 모두 지우고 1번부터 다시 할까요? 되돌릴 수 없습니다.")) return;
  state = {}; i = 0;
  try { localStorage.removeItem(KEY); } catch (e) {}
  render();
};
render();
</script>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transcripts", nargs="*", help="전사본 JSON (기본: data/external/call_youtube/*.json)")
    ap.add_argument("--title", default="상담 발화 각성 라벨링")
    ap.add_argument("--out", default="data/labeling/call_arousal.html")
    ap.add_argument("--sample", type=int, help="무작위로 n발화만")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    paths = [Path(p) for p in args.transcripts] or sorted(SOURCE.glob("*.json"))
    out = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    items = sample_items(items_from(paths, out), args.sample, args.seed)
    if not items:
        raise SystemExit("전사본이 없습니다. 먼저 scripts/call_youtube_check.py를 돌리세요")
    out.parent.mkdir(parents=True, exist_ok=True)
    page = (TEMPLATE.replace("__TITLE__", html.escape(args.title))
            .replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")))
    out.write_text(page, encoding="utf-8", newline="\n")
    print(f"통화 {len(paths)}개, 발화 {len(items)}개 → {out}")


if __name__ == "__main__":
    main()
