"""라벨링 도구 테스트: 후보 추출 → Label Studio 작업 → 내보내기 읽기 → 자기 일치도."""

import json

import numpy as np
import pandas as pd
import pytest

from voclens import labeling
from voclens.cli import main
from voclens.config import REPO_ROOT, load_settings

SAMPLE = REPO_ROOT / "data" / "samples" / "delivery_reviews_synthetic.csv"


@pytest.fixture(scope="module")
def table():
    return labeling.clause_table(SAMPLE, load_settings())


def test_clause_table(table):
    assert {"unit_id", "review_id", "text", "review", "labels", "taxonomy_conf"} <= set(table.columns)
    assert table["review_id"].nunique() == 180


def test_proxy_uncertainty_orders_confusing_first(table):
    scored = labeling.score_uncertainty(table)
    picked = labeling.select_for_labeling(scored, n=60)
    assert picked["margin"].is_monotonic_increasing
    assert picked["text"].str.replace(r"\s+", "", regex=True).is_unique          # 같은 문장은 한 번만
    assert picked.groupby("review_id").size().max() <= 2
    assert set(picked["kind"]) - {"certain"}                                        # 확실한 절은 나중에


def test_embedding_uncertainty_with_fake_encoder(table):
    def encode(texts):
        out = np.zeros((len(texts), 512))
        for r, t in enumerate(texts):
            for ch in t:
                out[r, hash(ch) % 512] += 1
        return out

    scored = labeling.score_uncertainty(table.head(30), encode=encode)
    assert scored["basis"].eq("embedding").all() and scored["margin"].between(0, 2).all()


def test_label_studio_tasks_match_config(table):
    tasks = labeling.to_label_studio(labeling.select_for_labeling(labeling.score_uncertainty(table), n=5))
    config = (REPO_ROOT / "labeling" / "label_studio_config.xml").read_text(encoding="utf-8")
    for var in ("text", "review", "zero_shot", "margin"):
        assert f"${var}" in config and var in tasks[0]["data"]


def _export(rows):
    return [
        {
            "data": {"unit_id": uid, "text": f"문장 {uid}"},
            "annotations": [{"result": [
                {"from_name": "complaint", "type": "choices", "value": {"choices": labels}},
                *([{"from_name": "hold", "type": "choices", "value": {"choices": ["보류"]}}] if hold else []),
                {"from_name": "memo", "type": "textarea", "value": {"text": ["메모"]}},
            ]}],
        }
        for uid, labels, hold in rows
    ] + [{"data": {"unit_id": "unlabeled"}, "annotations": []}]


def test_read_export_and_csv_without_text(tmp_path):
    path = tmp_path / "export.json"
    path.write_text(json.dumps(_export([("u1", ["배달 지연"], False), ("u2", ["맛", "온도·식감"], True)]), ensure_ascii=False), encoding="utf-8")
    df = labeling.read_label_studio_export(path)
    assert list(df["unit_id"]) == ["u1", "u2"] and df.loc[1, "hold"]
    out = labeling.labels_to_csv(df, tmp_path / "labels.csv")
    csv = pd.read_csv(out)
    assert "text" not in csv.columns and csv.loc[1, "labels"] == "맛|온도·식감"


def test_cohen_kappa_known_values():
    assert labeling.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert labeling.cohen_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    assert labeling.cohen_kappa([], []) is None


def test_agreement(tmp_path):
    a = labeling.read_label_studio_export(_write(tmp_path / "a.json", [("u1", ["배달 지연"], False), ("u2", ["맛"], False), ("u3", ["칭찬"], False), ("u4", ["가격"], True)]))
    b = labeling.read_label_studio_export(_write(tmp_path / "b.json", [("u1", ["배달 지연"], False), ("u2", ["온도·식감"], False), ("u3", ["칭찬"], False), ("u4", ["가격"], False)]))
    result = labeling.agreement(a, b)
    assert result["n_common"] == 3                      # 보류한 u4는 뺀다
    assert result["exact_match"] == pytest.approx(2 / 3, abs=1e-3)
    assert result["per_label"]["배달 지연"] == 1.0
    assert result["macro_kappa"] < 1.0


def _write(path, rows):
    path.write_text(json.dumps(_export(rows), ensure_ascii=False), encoding="utf-8")
    return path


def test_cli_label_flow(tmp_path, capsys):
    tasks = tmp_path / "tasks.json"
    assert main(["label", "prepare", str(SAMPLE), "--n", "10", "--out", str(tasks)]) == 0
    assert len(json.loads(tasks.read_text(encoding="utf-8"))) == 10
    export = _write(tmp_path / "e.json", [("u1", ["배달 지연"], False), ("u2", ["맛"], False)])
    assert main(["label", "import", str(export), "--out", str(tmp_path / "l.csv")]) == 0
    assert main(["label", "agreement", str(export), str(export)]) == 0
    assert "macro κ 1.0" in capsys.readouterr().out
