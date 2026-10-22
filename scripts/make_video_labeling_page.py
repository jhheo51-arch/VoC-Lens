"""영상 크리에이터 문장에 사람이 감성과 내용 구간 정답을 붙이는 라벨링 페이지(HTML 한 장)를 만든다.

scripts/video_bundle_cache.py로 data/external/youtube/ 에 받아 둔 번들을 파이프라인과 같은 방식으로 문장으로 나누고
(상투 문구 제외), 문장마다 시각과 본문만 보여 준다. 모델 예측은 넣지 않는다 (라벨러가 영향을 받지 않도록).

- 감성: 부정 / 중립 / 긍정 / 판단 불가
- 구간: 도입·언박싱·사용·장단점·총평·본문. 비워 두면 "앞 문장과 같은 구간"으로 본다 (구간이 바뀌는 문장에만 고르면 된다)

진행 상황은 브라우저에 저장되고, "결과 내려받기"로
[{"unit_id", "video_id", "split", "start_sec", "sentiment", "segment", "segment_explicit", "memo"}] JSON을 받는다.
unit_id는 파이프라인의 unit_id(video:<id>#<순번>)와 같아서 예측과 바로 맞춰 볼 수 있다.

    python scripts/make_video_labeling_page.py --dev dStfwAbQB0A,-D9zBKMBkeU,ULoEZw889Lc --eval crOa9gsD9tE,hNwemeKKNGs,AWmjVAoC_zE
    # 평가용 영상의 크리에이터 문장에서 무작위 50문장, 감성만 (구간 경계는 이어진 문장이 있어야 해서 표본으로는 재지 않는다)
    python scripts/make_video_labeling_page.py --eval crOa9gsD9tE,hNwemeKKNGs,AWmjVAoC_zE --sample 50 --out data/labeling/video_speech_50.html

원문이 들어 있으므로 data/labeling/ 아래에 만들고 저장소에 올리지 않는다 (data/ 는 .gitignore).
"""

import argparse
import html
import json
import random
import re
from pathlib import Path

from voclens.config import REPO_ROOT, load_yaml
from voclens.ingest.youtube import load_bundle
from voclens.pipelines.video import main_speaker, split_timed
from voclens.text.preprocess import preprocess

CACHE = REPO_ROOT / "data" / "external" / "youtube"
SENTIMENTS = ["부정", "중립", "긍정", "판단 불가"]


def video_items(video_id: str, split: str, vcfg: dict) -> dict:
    bundle = load_bundle(CACHE / f"{video_id}.json")
    boilerplate = re.compile("|".join(vcfg["boilerplate"])) if vcfg.get("boilerplate") else None
    spans = split_timed(bundle, boilerplate)
    creator = main_speaker(spans)
    texts = [preprocess(s.text).text for s in spans]
    rows = [
        {"unit_id": f"video:{video_id}#{i}", "t": s.start, "text": texts[i], "guest": s.speaker != creator,
         "before": texts[i - 1] if i else "", "after": texts[i + 1] if i + 1 < len(texts) else ""}
        for i, s in enumerate(spans)
    ]
    return {"video_id": video_id, "split": split, "title": bundle.meta.get("title") or video_id,
            "duration": bundle.meta.get("duration"), "rows": rows}


def sample_rows(videos: list[dict], n: int, seed: int) -> list[dict]:
    """크리에이터 문장에서 무작위로 n개를 고르고, 영상별 시간 순서로 돌려놓는다. 고른 문장이 없는 영상은 뺀다."""
    pool = [(vi, ri) for vi, v in enumerate(videos) for ri, r in enumerate(v["rows"]) if not r["guest"]]
    keep = set(random.Random(seed).sample(pool, min(n, len(pool))))
    out = []
    for vi, v in enumerate(videos):
        rows = [r for ri, r in enumerate(v["rows"]) if (vi, ri) in keep]
        if rows:
            out.append({**v, "rows": rows})
    return out


def build(videos: list[dict], segment_labels: list[str], title: str, key: str = "voclens-video-labels-v1") -> str:
    """segment_labels를 비우면 구간 선택 없이 감성만 받는 페이지가 된다(표본 모드)."""
    data = json.dumps({"videos": videos, "sentiments": SENTIMENTS, "segments": segment_labels, "key": key}, ensure_ascii=False)
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data.replace("</", "<\\/"))


TEMPLATE = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f4f6f9; --card:#fff; --ink:#131a24; --muted:#6b7585; --rule:#dce2ea; --blue:#2a78d6; --hl:#fff3c4; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1218; --card:#151c26; --ink:#edf1f6; --muted:#8b96a6; --rule:#263040; --blue:#3987e5; --hl:#4a3f12; } }
* { box-sizing: border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--ink); font:15px/1.6 "Malgun Gothic", system-ui, sans-serif; word-break:keep-all; }
main { max-width:960px; margin:0 auto; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; justify-content:space-between; margin-bottom:12px; }
h1 { font-size:18px; margin:0; }
.small { font-size:13px; color:var(--muted); }
select, button, input { font:inherit; padding:6px 10px; border-radius:8px; border:1px solid var(--rule); background:var(--card); color:var(--ink); }
button.primary { background:var(--blue); border-color:var(--blue); color:#fff; }
.row { display:grid; grid-template-columns:56px 1fr; gap:4px 12px; background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:10px 12px; margin-bottom:8px; }
.row.start { border-left:4px solid var(--blue); }
.row .t { font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums; }
.row .text { font-size:16px; }
.row .ctl { grid-column:2; display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; }
.row label { display:inline-flex; align-items:center; gap:3px; cursor:pointer; font-size:14px; }
.row .seg-now { font-size:13px; color:var(--muted); }
.guest { font-size:12px; background:var(--hl); padding:0 6px; border-radius:4px; margin-left:6px; }
.row input.memo { flex:1 1 160px; min-width:120px; padding:4px 8px; font-size:13px; }
</style>
<main>
<header>
  <h1>__TITLE__</h1>
  <div><select id="video"></select> <span id="progress" class="small"></span></div>
</header>
<p class="small" id="intro">문장마다 감성을 고르고, 구간이 바뀌는 문장에서만 구간 이름을 고르세요 (비우면 앞 문장과 같은 구간). 영상 링크를 열어 해당 시각을 들어 봐도 됩니다.
진행 상황은 이 브라우저에 저장됩니다. 끝나면 결과를 내려받으세요.</p>
<p><a id="link" target="_blank" rel="noopener"></a></p>
<div id="rows"></div>
<p><button class="primary" id="dl">결과 내려받기 (전체 영상)</button></p>
</main>
<script>
const D = __DATA__;
const KEY = D.key;
if (!D.segments.length) document.getElementById("intro").textContent = "평가용 영상에서 무작위로 뽑은 크리에이터 문장입니다. 문장마다 감성만 고르세요(앞뒤 문장은 참고용). 시각을 누르면 그 부분을 들을 수 있습니다. 진행 상황은 이 브라우저에 저장됩니다. 끝나면 결과를 내려받으세요.";
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} };
const sel = document.getElementById("video");
D.videos.forEach((v, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `[${v.split}] ${v.title}`; sel.appendChild(o); });
const fmt = s => { s = Math.floor(s); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
function resolved(v) {
  let cur = "";
  return v.rows.map(r => { const s = (store[r.unit_id] || {}).segment; if (s) cur = s; return cur; });
}
function progress() {
  const v = D.videos[sel.value];
  const done = v.rows.filter(r => (store[r.unit_id] || {}).sentiment).length;
  document.getElementById("progress").textContent = `감성 ${done} / ${v.rows.length}`;
}
function render() {
  const v = D.videos[sel.value];
  const a = document.getElementById("link");
  a.href = `https://www.youtube.com/watch?v=${v.video_id}`; a.textContent = `영상 열기 (${v.video_id}, ${fmt(v.duration || 0)})`;
  const box = document.getElementById("rows"); box.innerHTML = "";
  const segs = resolved(v);
  v.rows.forEach((r, i) => {
    const st = store[r.unit_id] || {};
    const div = document.createElement("div"); div.className = "row" + (st.segment ? " start" : "");
    const t = document.createElement("a"); t.className = "t"; t.textContent = fmt(r.t);
    t.href = `https://www.youtube.com/watch?v=${v.video_id}&t=${Math.floor(r.t)}s`; t.target = "_blank"; t.rel = "noopener";
    const tx = document.createElement("div"); tx.className = "text"; tx.textContent = r.text;
    if (!D.segments.length && (r.before || r.after)) {
      const c = document.createElement("div"); c.className = "small"; c.textContent = `앞: ${r.before || "(없음)"} / 뒤: ${r.after || "(없음)"}`; tx.appendChild(c);
    }
    if (r.guest) { const g = document.createElement("span"); g.className = "guest"; g.textContent = "다른 화자"; tx.appendChild(g); }
    const ctl = document.createElement("div"); ctl.className = "ctl";
    D.sentiments.forEach(s => {
      const l = document.createElement("label"); const inp = document.createElement("input");
      inp.type = "radio"; inp.name = r.unit_id; inp.value = s; inp.checked = st.sentiment === s;
      inp.onchange = () => { store[r.unit_id] = { ...(store[r.unit_id] || {}), sentiment: s }; save(); progress(); };
      l.append(inp, " " + s); ctl.appendChild(l);
    });
    const ss = document.createElement("select");
    ss.innerHTML = `<option value="">(앞과 같은 구간)</option>` + D.segments.map(s => `<option${st.segment === s ? " selected" : ""}>${s}</option>`).join("");
    ss.onchange = () => { store[r.unit_id] = { ...(store[r.unit_id] || {}), segment: ss.value }; save(); render(); };
    const now = document.createElement("span"); now.className = "seg-now"; now.textContent = segs[i] ? `구간: ${segs[i]}` : "구간: 미정";
    const memo = document.createElement("input"); memo.className = "memo"; memo.placeholder = "메모"; memo.value = st.memo || "";
    memo.onchange = () => { store[r.unit_id] = { ...(store[r.unit_id] || {}), memo: memo.value }; save(); };
    if (D.segments.length) ctl.append(ss, now);
    ctl.append(memo);
    div.append(t, tx, ctl); box.appendChild(div);
  });
  progress();
}
sel.onchange = render;
document.getElementById("dl").onclick = () => {
  const out = [];
  D.videos.forEach(v => {
    const segs = resolved(v);
    v.rows.forEach((r, i) => {
      const st = store[r.unit_id] || {};
      out.push({ unit_id: r.unit_id, video_id: v.video_id, split: v.split, start_sec: r.t, sentiment: st.sentiment || null,
                 segment: segs[i] || null, segment_explicit: !!st.segment, memo: st.memo || "" });
    });
  });
  const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "video_speech_labels.json"; a.click();
};
render();
</script>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=lambda s: s.split(","), default=[], help="개발용 영상 ID (쉼표로 구분)")
    ap.add_argument("--eval", type=lambda s: s.split(","), default=[], help="평가용 영상 ID (쉼표로 구분)")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "labeling" / "video_speech.html"))
    ap.add_argument("--sample", type=int, help="크리에이터 문장에서 무작위로 n개만, 감성만 라벨링")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    vcfg = load_yaml("video.yaml")
    videos = [video_items(v, "dev", vcfg) for v in args.dev] + [video_items(v, "eval", vcfg) for v in args.eval]
    labels = [*vcfg["segment_labels"], vcfg["default_segment_label"]]
    title, key = "영상 문장 감성·구간 라벨링", "voclens-video-labels-v1"
    if args.sample:
        videos, labels = sample_rows(videos, args.sample, args.seed), []
        title, key = f"영상 문장 감성 라벨링 ({args.sample}문장)", f"voclens-video-sample-{args.sample}-{args.seed}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(videos, labels, title, key), encoding="utf-8", newline="\n")
    print(f"영상 {len(videos)}편, 문장 {sum(len(v['rows']) for v in videos)}개 → {out}")
    print("원문이 들어 있으니 저장소에 올리지 마세요 (data/ 는 .gitignore).")


if __name__ == "__main__":
    main()
