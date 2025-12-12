# this_file: tests/eledubby/test_cast.py

from __future__ import annotations

from pathlib import Path

import pytest


def test_cast_missing_function_then_fails() -> None:
    import eledubby.eledubby as mod

    assert hasattr(mod, "cast"), "Expected `eledubby.eledubby.cast` to exist"


def test_parse_voices_list_when_csv_then_parses() -> None:
    import eledubby.eledubby as mod

    parse = mod._parse_voice_specs_from_voices_list
    assert parse(" a,b , c ") == [("a", None), ("b", None), ("c", None)]


def test_parse_voices_path_when_semicolon_then_parses(tmp_path: Path) -> None:
    import eledubby.eledubby as mod

    voices_path = tmp_path / "voices.txt"
    voices_path.write_text(
        "\n".join(
            [
                "",
                "# comment",
                "voice_1",
                "voice_2; key_2",
            ]
        ),
        encoding="utf-8",
    )

    parse = mod._parse_voice_specs_from_voices_path
    assert parse(voices_path) == [("voice_1", None), ("voice_2", "key_2")]


def test_default_cast_output_dir_when_output_none_then_basename_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eledubby.eledubby as mod

    monkeypatch.chdir(tmp_path)
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    text_path = text_dir / "hello.txt"
    text_path.write_text("hello", encoding="utf-8")

    default_dir = mod._default_cast_output_dir
    assert default_dir(text_path, None) == tmp_path / "hello"


def test_cast_when_voices_list_then_writes_mp3s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eledubby.eledubby as mod

    class FakeClient:
        init_keys: list[str | None] = []

        def __init__(self, api_key: str | None = None, max_retries: int = 3):
            self.api_key = api_key
            self.max_retries = max_retries
            FakeClient.init_keys.append(api_key)

        def text_to_speech(
            self,
            *,
            text: str,
            voice_id: str,
            output_path: str,
            _output_format: str = "mp3_44100_128",
        ) -> str:
            Path(output_path).write_bytes(
                b"MP3:" + voice_id.encode("utf-8") + b":" + text.encode("utf-8")
            )
            return output_path

        def list_voices(self) -> list[dict[str, str]]:
            return [{"id": "v1", "name": "V1", "category": "unknown"}]

    monkeypatch.setattr(mod, "ElevenLabsClient", FakeClient)
    monkeypatch.chdir(tmp_path)

    text_path = tmp_path / "t.txt"
    text_path.write_text("hi", encoding="utf-8")

    mod.cast(text_path=text_path, voices_list="v1,v2", api_key="k", output=None)

    out_dir = tmp_path / "t"
    assert (out_dir / "v1.mp3").read_bytes().startswith(b"MP3:v1:")
    assert (out_dir / "v2.mp3").read_bytes().startswith(b"MP3:v2:")
    assert FakeClient.init_keys == ["k"]


def test_cast_when_voices_path_with_per_voice_keys_then_uses_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eledubby.eledubby as mod

    class FakeClient:
        init_keys: list[str | None] = []

        def __init__(self, api_key: str | None = None, max_retries: int = 3):
            self.api_key = api_key
            self.max_retries = max_retries
            FakeClient.init_keys.append(api_key)

        def text_to_speech(
            self,
            *,
            text: str,
            voice_id: str,
            output_path: str,
            _output_format: str = "mp3_44100_128",
        ) -> str:
            _ = (text, voice_id, _output_format)
            Path(output_path).write_bytes(b"OK")
            return output_path

        def list_voices(self) -> list[dict[str, str]]:
            return []

    monkeypatch.setattr(mod, "ElevenLabsClient", FakeClient)
    monkeypatch.chdir(tmp_path)

    text_path = tmp_path / "t.txt"
    text_path.write_text("hi", encoding="utf-8")

    voices_path = tmp_path / "voices.txt"
    voices_path.write_text("v1;k1\nv2; k2\n", encoding="utf-8")

    mod.cast(text_path=text_path, voices_path=voices_path, output=None)

    assert (tmp_path / "t" / "v1.mp3").exists()
    assert (tmp_path / "t" / "v2.mp3").exists()
    assert set(FakeClient.init_keys) == {"k1", "k2"}
