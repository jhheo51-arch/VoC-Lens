from voclens.config import load_dotenv


def test_load_dotenv_sets_missing_keys_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# 주석\nVOCLENS_T_A=abc\nVOCLENS_T_B="quoted"\nVOCLENS_T_EMPTY=\nVOCLENS_T_KEEP=new\n', encoding="utf-8")
    for key in ("VOCLENS_T_A", "VOCLENS_T_B", "VOCLENS_T_EMPTY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VOCLENS_T_KEEP", "old")
    load_dotenv(env)
    import os

    assert os.environ["VOCLENS_T_A"] == "abc"
    assert os.environ["VOCLENS_T_B"] == "quoted"
    assert "VOCLENS_T_EMPTY" not in os.environ   # 빈 값은 건너뜀
    assert os.environ["VOCLENS_T_KEEP"] == "old"  # 이미 있는 환경변수는 덮어쓰지 않음
    for key in ("VOCLENS_T_A", "VOCLENS_T_B"):
        monkeypatch.delenv(key)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "없음.env")
