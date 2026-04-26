"""Tests for Feishu media download, upload, and extension resolution.

Run with: python3 -m pytest tests/test_feishu_media.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyagent.platforms.feishu import (
    _guess_extension_from_content_type,
    _guess_extension_from_filename,
    _resolve_extension,
    _CT_EXT_OVERRIDES,
)


def test_guess_extension_from_content_type():
    """Content-Type to extension mapping."""
    assert _guess_extension_from_content_type("image/png") == ".png"
    assert _guess_extension_from_content_type("image/jpeg") == ".jpg"
    assert _guess_extension_from_content_type("image/jpg") == ".jpg"
    assert _guess_extension_from_content_type("image/gif") == ".gif"
    assert _guess_extension_from_content_type("image/webp") == ".webp"
    assert _guess_extension_from_content_type("audio/mpeg") == ".mp3"
    assert _guess_extension_from_content_type("audio/mp3") == ".mp3"
    assert _guess_extension_from_content_type("video/mp4") == ".mp4"
    assert _guess_extension_from_content_type("application/pdf") == ".pdf"
    assert _guess_extension_from_content_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document") == ".docx"


def test_guess_extension_from_content_type_with_charset():
    """Content-Type with charset suffix should be handled."""
    assert _guess_extension_from_content_type("text/plain; charset=utf-8") == ".txt"
    assert _guess_extension_from_content_type("image/jpeg; charset=utf-8") == ".jpg"


def test_guess_extension_from_content_type_unknown():
    """Unknown Content-Type should return empty string."""
    assert _guess_extension_from_content_type("application/x-unknown") == ""
    assert _guess_extension_from_content_type("") == ""
    assert _guess_extension_from_content_type(None) == ""


def test_guess_extension_from_filename():
    """Filename to extension extraction."""
    assert _guess_extension_from_filename("photo.jpg") == ".jpg"
    assert _guess_extension_from_filename("document.PDF") == ".pdf"
    assert _guess_extension_from_filename("archive.tar.gz") == ".gz"
    assert _guess_extension_from_filename("no_extension") == ""
    assert _guess_extension_from_filename("") == ""
    assert _guess_extension_from_filename(None) == ""


def test_resolve_extension_priority():
    """Filename extension takes priority over Content-Type."""
    assert _resolve_extension("image/png", "photo.jpg") == ".jpg"
    assert _resolve_extension("application/pdf", "doc.txt") == ".txt"


def test_resolve_extension_fallback_to_content_type():
    """When no filename extension, fall back to Content-Type."""
    assert _resolve_extension("image/png", None) == ".png"
    assert _resolve_extension("video/mp4", "") == ".mp4"


def test_resolve_extension_default():
    """When neither filename nor Content-Type provides extension, use default."""
    assert _resolve_extension(None, None, default=".bin") == ".bin"
    assert _resolve_extension("application/x-unknown", "noext", default=".dat") == ".dat"


def test_ct_ext_overrides_coverage():
    """All override mappings should be lowercase and start with dot."""
    for ct, ext in _CT_EXT_OVERRIDES.items():
        assert ct == ct.lower(), f"Content-Type '{ct}' should be lowercase"
        assert ext.startswith("."), f"Extension '{ext}' should start with dot"


if __name__ == "__main__":
    import traceback

    tests = [
        test_guess_extension_from_content_type,
        test_guess_extension_from_content_type_with_charset,
        test_guess_extension_from_content_type_unknown,
        test_guess_extension_from_filename,
        test_resolve_extension_priority,
        test_resolve_extension_fallback_to_content_type,
        test_resolve_extension_default,
        test_ct_ext_overrides_coverage,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {test.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
