"""작업 JSON(voclens eval prepare / label prepare 결과)에서 브라우저로 여는 라벨링 페이지(HTML 한 장)를 만든다.

Label Studio 없이 라벨링할 때 쓴다. 진행 상황은 브라우저에 저장되고, 끝나면 "결과 내려받기"로
[{"unit_id", "labels", "hold", "memo"}] JSON을 받는다. 이 JSON은 voclens eval gold / label agreement에 그대로 넣는다.

    python scripts/make_labeling_page.py data/labeling/eval_tasks.json --n 50 --out data/labeling/human_check.html
    # 이미 검수한 절은 빼고 30절 더 (다른 seed)
    python scripts/make_labeling_page.py data/labeling/eval_tasks.json --n 30 --seed 3030         --exclude data/labeling/human_labels_20.json --out data/labeling/human_check_30.html

원문이 들어 있으므로 data/ 아래에 만들고 저장소에 올리지 않는다.
"""

import argparse
import html
import json
import random
from pathlib import Path

from voclens.text.taxonomy import Taxonomy

# 유형을 함께 붙이지 않는 라벨. 분류 불가는 기준서 v0.3부터 "이유 없는 불만"으로 불만에 센다(칭찬·해당 없음만 불만 아님)
NON_COMPLAINT = ["칭찬", "해당 없음", "분류 불가"]


def labels_for(domain: str) -> dict[str, list[str]]:
    """대분류 → 소분류 (해당 도메인 + 공통)."""
    tx = Taxonomy.load()
    groups: dict[str, list[str]] = {}
    for name, sub in tx.subcategories.items():
        domains = sub.get("domains")
        if not domains or domain in domains:
            groups.setdefault(sub["category"], []).append(name)
    return groups


def build(tasks: list[dict], groups: dict[str, list[str]], title: str, preload: dict[str, dict] | None = None) -> str:
    items = [{"unit_id": t["data"]["unit_id"], "text": t["data"]["text"], "review": t["data"]["review"]} for t in tasks]
    choices = '<fieldset><legend>불만 아님</legend>' + "".join(
        f'<label><input type="checkbox" value="{n}"> {n}</label>' for n in NON_COMPLAINT[:2]) + "</fieldset>"
    choices += ('<fieldset><legend>불만인데 이유가 없음</legend><label><input type="checkbox" value="분류 불가"> 분류 불가</label>'
                '<span class="small">"다신 안 살게요", "실망이에요"처럼 이 절에 이유가 없거나 어느 유형에도 안 맞는 불만</span></fieldset>')
    order = ["제품·품질", "배송·이행", "가격·혜택", "응대·서비스", "사용성"]
    groups = {c: groups[c] for c in sorted(groups, key=lambda c: order.index(c) if c in order else len(order))}
    choices += "".join(
        f'<fieldset><legend>{html.escape(cat)}</legend>'
        + "".join(f'<label><input type="checkbox" value="{html.escape(n)}"> {html.escape(n)}</label>' for n in names)
        + "</fieldset>"
        for cat, names in groups.items()
    )
    preload = {k: v for k, v in (preload or {}).items() if k in {i["unit_id"] for i in items}}
    return (TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__CHOICES__", choices)
            .replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__PRELOAD__", json.dumps(preload, ensure_ascii=False).replace("</", "<\\/")))


def excluded_ids(paths: list[str]) -> set[str]:
    """제외할 절 ID. 라벨 JSON([{unit_id, ...}]), 작업 JSON([{data: {unit_id}}]), ID 목록(["...", ...]) 모두 받는다."""
    ids: set[str] = set()
    for path in paths:
        for r in json.loads(Path(path).read_text(encoding="utf-8")):
            ids.add(r if isinstance(r, str) else r["data"]["unit_id"] if "data" in r else r["unit_id"])
    return ids


def pick_tasks(tasks: list[dict], n: int | None, seed: int, exclude: set[str] | None = None) -> list[dict]:
    """exclude에 있는 절을 먼저 빼고, n을 주면 남은 것에서 무작위로 n개 고른다."""
    tasks = [t for t in tasks if t["data"]["unit_id"] not in (exclude or set())]
    return random.Random(seed).sample(tasks, min(n, len(tasks))) if n else tasks


def needs_review(labels: list[str]) -> bool:
    """유형 없는 라벨(칭찬·해당 없음·분류 불가)과 불만 유형이 함께 있거나, 라벨이 3개 이상이면 다시 본다."""
    non = [l for l in labels if l in NON_COMPLAINT]
    return len(labels) >= 3 or bool(non and len(non) < len(labels)) or len(non) > 1


TEMPLATE = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f4f6f9; --card:#fff; --ink:#131a24; --muted:#6b7585; --rule:#dce2ea; --blue:#2a78d6; --hl:#fff3c4; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1218; --card:#151c26; --ink:#edf1f6; --muted:#8b96a6; --rule:#263040; --blue:#3987e5; --hl:#4a3f12; } }
* { box-sizing: border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--ink); font:15px/1.6 "Malgun Gothic", system-ui, sans-serif; word-break:keep-all; }
main { max-width:860px; margin:0 auto; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; justify-content:space-between; margin-bottom:12px; }
h1 { font-size:18px; margin:0; }
.bar { height:6px; background:var(--rule); border-radius:3px; overflow:hidden; margin-bottom:16px; }
.bar > i { display:block; height:100%; background:var(--blue); width:0; }
.card { background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:16px; margin-bottom:12px; }
.small { font-size:13px; color:var(--muted); }
.clause { font-size:18px; font-weight:600; margin:6px 0 0; }
.review mark { background:var(--hl); color:inherit; padding:0 2px; border-radius:3px; }
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
  <div class="small">리뷰 전체 (참고용, 노란 부분이 라벨을 붙일 절)</div>
  <div class="review" id="review"></div>
  <div class="small" style="margin-top:12px">라벨을 붙일 절 (이 절 안에서 말한 것만 봅니다. 이유가 다른 절에만 있으면 분류 불가)</div>
  <p class="clause" id="clause"></p>
</div>
<div class="card" id="choices">__CHOICES__
  <label><input type="checkbox" id="hold"> 보류 (확신 없음)</label>
  <textarea id="memo" placeholder="메모 (애매한 이유 등, 선택)"></textarea>
</div>
<div class="warn" id="warn"></div>
<nav>
  <button id="prev">← 이전</button><button id="next" class="primary">다음 →</button>
  <button id="download">결과 내려받기</button>
  <button id="reset">처음부터 다시</button>
</nav>
<p class="small">진행 상황은 이 브라우저에 저장됩니다(앱 안 미리보기 창에서는 저장되지 않을 수 있으니 창을 닫기 전에 결과를 내려받으세요). 단축키: ← → 이동. 기준은 labeling/guidelines.md와 label_definitions_product.md를 따릅니다.</p>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "voclens-label-" + ITEMS.length + "-" + ITEMS[0].unit_id;
const PRELOAD = __PRELOAD__;
let state = Object.assign({}, PRELOAD); let i = 0;
try { const s = JSON.parse(localStorage.getItem(KEY) || "{}"); Object.assign(state, s.state || {}); i = s.i || 0; } catch (e) {}
const boxes = [...document.querySelectorAll('#choices fieldset input')];
const NON = ["칭찬", "해당 없음", "분류 불가"];
// 실수 방지: 유형 없는 라벨(칭찬·해당 없음·분류 불가) 하나를 고르면 나머지가 모두 풀리고, 불만 유형을 고르면 유형 없는 라벨이 풀린다
boxes.forEach(b => b.addEventListener('change', () => {
  if (!b.checked) return;
  if (NON.includes(b.value)) boxes.forEach(o => { if (o !== b) o.checked = false; });
  else boxes.forEach(o => { if (NON.includes(o.value)) o.checked = false; });
  const n = boxes.filter(o => o.checked).length;
  document.getElementById('warn').textContent = n >= 3 ? `라벨이 ${n}개입니다. 보통 1개, 많아야 2개예요. 맞는지 확인해 주세요.` : "";
}));
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function save() { try { localStorage.setItem(KEY, JSON.stringify({state, i})); } catch (e) {} }
function collect() {
  const it = ITEMS[i];
  state[it.unit_id] = { labels: boxes.filter(b => b.checked).map(b => b.value),
    hold: document.getElementById('hold').checked, memo: document.getElementById('memo').value };
}
function render() {
  const it = ITEMS[i], s = state[it.unit_id] || {labels: [], hold: false, memo: ""};
  const at = it.review.indexOf(it.text);
  document.getElementById('review').innerHTML = at < 0 ? esc(it.review)
    : esc(it.review.slice(0, at)) + "<mark>" + esc(it.text) + "</mark>" + esc(it.review.slice(at + it.text.length));
  document.getElementById('clause').textContent = it.text;
  boxes.forEach(b => b.checked = s.labels.includes(b.value));
  document.getElementById('hold').checked = s.hold;
  document.getElementById('memo').value = s.memo;
  const done = ITEMS.filter(x => (state[x.unit_id] || {}).labels?.length).length;
  document.getElementById('pos').textContent = `${i + 1} / ${ITEMS.length} · 완료 ${done}`;
  document.getElementById('prog').style.width = (done / ITEMS.length * 100) + "%";
  document.getElementById('warn').textContent = "";
}
function go(d) {
  collect();
  if (d > 0 && !state[ITEMS[i].unit_id].labels.length) { document.getElementById('warn').textContent = "라벨을 하나 이상 고르세요."; return; }
  i = Math.min(ITEMS.length - 1, Math.max(0, i + d)); save(); render();
}
document.getElementById('prev').onclick = () => go(-1);
document.getElementById('next').onclick = () => go(1);
document.addEventListener('keydown', e => { if (e.target.tagName === 'TEXTAREA') return; if (e.key === 'ArrowRight') go(1); if (e.key === 'ArrowLeft') go(-1); });
document.getElementById('download').onclick = () => {
  collect(); save();
  const out = ITEMS.filter(x => (state[x.unit_id] || {}).labels?.length).map(x => ({unit_id: x.unit_id, ...state[x.unit_id]}));
  const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  const a = Object.assign(document.createElement('a'), {href: url, download: "human_labels.json"}); a.click();
  URL.revokeObjectURL(url);
};
document.getElementById('reset').onclick = () => {
  if (!confirm("고른 라벨과 메모를 모두 지우고 1번부터 다시 할까요? 되돌릴 수 없습니다.")) return;
  state = Object.assign({}, PRELOAD); i = 0;
  try { localStorage.removeItem(KEY); } catch (e) {}
  render();
};
render();
</script>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tasks", help="작업 JSON (Label Studio 형식)")
    ap.add_argument("--n", type=int, help="무작위로 n개만 (사람 검수용)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--domain", default="product")
    ap.add_argument("--title", default="VoC Lens 라벨링")
    ap.add_argument("--out", default="data/labeling/labeling_page.html")
    ap.add_argument("--preload", help="이전 라벨 JSON([{unit_id, labels, hold, memo}]). 체크된 상태로 시작한다")
    ap.add_argument("--exclude", nargs="*", default=[], help="뺄 절 ID가 든 JSON (이미 검수한 사람 라벨 등). 무작위 추출 전에 뺀다")
    ap.add_argument("--only-flagged", action="store_true", help="--preload 중 다시 볼 항목(불만 아님과 유형이 섞임, 3개 이상)만")
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = pick_tasks(tasks, args.n, args.seed, excluded_ids(args.exclude))
    preload = {}
    if args.preload:
        preload = {r["unit_id"]: {"labels": r["labels"], "hold": bool(r.get("hold")), "memo": r.get("memo", "")}
                   for r in json.loads(Path(args.preload).read_text(encoding="utf-8"))}
        if args.only_flagged:
            tasks = [t for t in tasks if t["data"]["unit_id"] in preload and needs_review(preload[t["data"]["unit_id"]]["labels"])]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(tasks, labels_for(args.domain), args.title, preload), encoding="utf-8")
    print(f"{len(tasks)}개 → {out}")


if __name__ == "__main__":
    main()
