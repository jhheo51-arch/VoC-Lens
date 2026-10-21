"""직접 라벨 합치기(scripts/train_complaint.py)와 사람 검수 페이지의 제외 옵션(scripts/make_labeling_page.py)."""

import importlib.util
import json

import pandas as pd
import pytest

from voclens.config import REPO_ROOT


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def page():
    return load_script("make_labeling_page")


def tasks(n):
    return [{"data": {"unit_id": f"review:ns{i}#0", "text": "t", "review": "r"}} for i in range(n)]


def test_pick_tasks_excludes_before_sampling(page):
    ts = tasks(50)
    done = {f"review:ns{i}#0" for i in range(20)}
    picked = page.pick_tasks(ts, 30, seed=3030, exclude=done)
    ids = [t["data"]["unit_id"] for t in picked]
    assert len(ids) == 30 and not set(ids) & done
    assert page.pick_tasks(ts, 30, seed=3030, exclude=done) == picked          # 같은 seed면 같은 절


def test_pick_tasks_without_exclude_keeps_old_sample(page):
    import random

    ts = tasks(40)
    assert page.pick_tasks(ts, 10, seed=2110) == random.Random(2110).sample(ts, 10)   # 기존 20절 페이지와 같은 추출
    assert page.pick_tasks(ts, None, seed=1) == ts


def test_excluded_ids_reads_label_task_and_id_files(page, tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps([{"unit_id": "a", "labels": ["칭찬"]}]), encoding="utf-8")
    (tmp_path / "tasks.json").write_text(json.dumps([{"data": {"unit_id": "b"}}]), encoding="utf-8")
    (tmp_path / "ids.json").write_text(json.dumps(["c"]), encoding="utf-8")
    assert page.excluded_ids([str(tmp_path / f) for f in ("labels.json", "tasks.json", "ids.json")]) == {"a", "b", "c"}


def test_direct_examples():
    train = load_script("train_complaint")
    gold = pd.DataFrame([
        {"unit_id": "u1", "labels": ["성능"], "hold": False},
        {"unit_id": "u2", "labels": ["칭찬"], "hold": False},
        {"unit_id": "u3", "labels": ["오배송·누락", "포장·파손"], "hold": False},
        {"unit_id": "u4", "labels": ["효과 없음"], "hold": True},                 # 보류는 뺀다
        {"unit_id": "u5", "labels": ["없는 유형"], "hold": False},
        {"unit_id": "u6", "labels": ["성능"], "hold": False},                     # 절을 못 찾음
    ])
    inputs = {f"u{i}": f"text{i}" for i in range(1, 6)}
    classes = ["불만 없음", "성능", "오배송·누락", "포장·파손", "효과 없음"]
    out, skipped = train.direct_examples(gold, inputs, classes)
    assert out == [("text1", ["성능"]), ("text2", []), ("text3", ["오배송·누락"]), ("text3", ["포장·파손"])]
    assert skipped == {"없는 유형": 1, "절을 찾지 못함": 1}
