"""Standalone tests for tyagent Feishu message building and parsing.

Run with: python3 -m pytest tests/test_feishu_message_building.py -v
Or directly: python3 tests/test_feishu_message_building.py
"""

import json
import sys
from pathlib import Path

# Add parent to path so we can import tyagent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyagent.platforms.feishu import (
    _build_markdown_post_rows,
    _build_outbound_payload,
    _convert_tables_to_code_blocks,
    _extract_post_text,
    _MARKDOWN_HINT_RE,
    _MARKDOWN_TABLE_RE,
)


def test_table_detection():
    """Table content should be detected and forced to text type."""
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert _MARKDOWN_TABLE_RE.search(table), "Simple table should match"

    table_with_bold = "| a | b |\n|---|---|\n| 1 | 2 |\n\n**bold**"
    assert _MARKDOWN_TABLE_RE.search(table_with_bold), "Table+bold should match table re"


def test_no_false_positive_on_bold():
    """Bold text without table should NOT match table re."""
    bold = "**bold** text\nmore text"
    assert not _MARKDOWN_TABLE_RE.search(bold), "Bold only should not match table re"


def test_outbound_payload_table_forces_text():
    """Table content should use post type with table converted to code block."""
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    msg_type, payload = _build_outbound_payload(table)
    assert msg_type == "post", f"Table should use post, got {msg_type}"
    data = json.loads(payload)
    assert "zh_cn" in data
    # The table should be wrapped in a code fence inside the post content
    content_str = json.dumps(data, ensure_ascii=False)
    assert "```" in content_str, "Table should be wrapped in code fence"
    assert "| a | b |" in content_str


def test_outbound_payload_table_with_bold_forces_text():
    """Table + bold should use post type with table in code block."""
    content = "| a | b |\n|---|---|\n| 1 | 2 |\n\n**bold**"
    msg_type, payload = _build_outbound_payload(content)
    assert msg_type == "post", f"Table+bold should use post, got {msg_type}"
    data = json.loads(payload)
    assert "zh_cn" in data
    content_str = json.dumps(data, ensure_ascii=False)
    assert "```" in content_str, "Table should be wrapped in code fence"
    assert "**bold**" in content_str, "Bold markdown should still be present"


def test_outbound_payload_bold_uses_post():
    """Bold without table should use post type."""
    bold = "**bold** text"
    msg_type, payload = _build_outbound_payload(bold)
    assert msg_type == "post", f"Bold should use post, got {msg_type}"
    data = json.loads(payload)
    assert "zh_cn" in data


def test_outbound_payload_plain_text():
    """Plain text without markdown should use text type."""
    plain = "Hello world"
    msg_type, payload = _build_outbound_payload(plain)
    assert msg_type == "text", f"Plain text should use text, got {msg_type}"


def test_convert_tables_to_code_blocks_basic():
    """Simple table should be wrapped in a code fence."""
    text = "| C1 | C2 |\n|---|---|\n| A | B |"
    result = _convert_tables_to_code_blocks(text)
    assert result.startswith("```\n"), "Should start with code fence"
    assert result.endswith("\n```"), "Should end with code fence"
    assert "| C1 | C2 |" in result


def test_convert_tables_to_code_blocks_with_surrounding_text():
    """Table embedded in surrounding text."""
    text = "开头文字\n\n| C1 | C2 |\n|---|---|\n| A | B |\n\n结尾文字"
    result = _convert_tables_to_code_blocks(text)
    assert "开头文字" in result
    assert "```\n| C1 | C2 |" in result, "Table should be code-fenced"
    assert "结尾文字" in result


def test_convert_tables_to_code_blocks_multiple_tables():
    """Multiple tables should each be wrapped."""
    text = "| T1 |\n|---|\n| A |\n\n| T2 |\n|---|\n| B |"
    result = _convert_tables_to_code_blocks(text)
    assert result.count("```") == 4, "Two tables should create 2x ``` pairs, got %d" % result.count("```")


def test_convert_tables_to_code_blocks_table_with_bold():
    """Bold markdown outside the table should remain unchanged."""
    text = "**粗体**\n\n| C1 | C2 |\n|---|---|\n| A | B |\n\n更多文字"
    result = _convert_tables_to_code_blocks(text)
    assert "**粗体**" in result
    assert "```\n| C1 | C2 |" in result


def test_convert_tables_to_code_blocks_no_table():
    """Text without a table should be unchanged."""
    text = "普通文字\n\n**粗体**\n- 列表"
    result = _convert_tables_to_code_blocks(text)
    assert result == text, "No-table text should be unchanged"


def test_convert_tables_to_code_blocks_trailing_newline():
    """Table at end of text without trailing newline."""
    text = "前文\n\n| C1 | C2 |\n|---|---|\n| A | B |"
    result = _convert_tables_to_code_blocks(text)
    assert "前文" in result
    assert "```\n| C1 | C2 |" in result


def test_fence_boundary_4_backticks():
    """4-backtick fence should not be closed by inner 3-backtick line."""
    content = "text before\n````python\ninner ```\n````\ntext after"
    rows = _build_markdown_post_rows(content)
    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}: {rows}"
    # First row: prose before code block
    assert "text before" in rows[0][0]["text"]
    # Second row: code block (including inner ```)
    assert "inner ```" in rows[1][0]["text"]
    # Third row: prose after code block
    assert "text after" in rows[2][0]["text"]


def test_fence_boundary_tilde():
    """Tilde fence (~~~) should be recognized."""
    content = "text\n~~~bash\necho hello\n~~~\nmore text"
    rows = _build_markdown_post_rows(content)
    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
    assert "text" in rows[0][0]["text"]
    assert "echo hello" in rows[1][0]["text"]
    assert "more text" in rows[2][0]["text"]


def test_fence_boundary_mixed_fences():
    """Backtick and tilde fences in same content."""
    content = "```python\nprint(1)\n```\n\n~~~bash\necho 2\n~~~"
    rows = _build_markdown_post_rows(content)
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    assert "print(1)" in rows[0][0]["text"]
    assert "echo 2" in rows[1][0]["text"]


def test_fence_boundary_nested_4_in_6():
    """4-backtick line inside 6-backtick fence is content, not a close."""
    content = "``````\n````\ncode\n````\n``````"
    rows = _build_markdown_post_rows(content)
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    assert "code" in rows[0][0]["text"]


def test_extract_post_text_basic():
    """Extract text from Feishu post JSON with all supported tags."""
    post_json = {
        "post": {
            "zh_cn": {
                "content": [
                    [{"tag": "text", "text": "Hello "}],
                    [{"tag": "text", "text": "world"}],
                    [{"tag": "img", "image_key": "img_123"}],
                    [{"tag": "media", "file_key": "media_123"}],
                    [{"tag": "file", "file_key": "file_123"}],
                    [{"tag": "audio", "file_key": "audio_123"}],
                    [{"tag": "video", "file_key": "video_123"}],
                ]
            }
        }
    }
    result = _extract_post_text(post_json)
    assert "Hello" in result
    assert "world" in result
    assert "[Image]" in result
    assert "[Media]" in result
    assert "[File]" in result
    assert "[Audio]" in result
    assert "[Video]" in result


def test_markdown_hint_re_patterns():
    """Verify _MARKDOWN_HINT_RE matches expected patterns."""
    assert _MARKDOWN_HINT_RE.search("# Heading")
    assert _MARKDOWN_HINT_RE.search("- list item")
    assert _MARKDOWN_HINT_RE.search("* list item")
    assert _MARKDOWN_HINT_RE.search("1. numbered")
    assert _MARKDOWN_HINT_RE.search("```code```")
    assert _MARKDOWN_HINT_RE.search("`inline code`")
    assert _MARKDOWN_HINT_RE.search("**bold**")
    assert _MARKDOWN_HINT_RE.search("~~strike~~")
    assert _MARKDOWN_HINT_RE.search("<u>underline</u>")
    assert _MARKDOWN_HINT_RE.search("*italic*")
    assert _MARKDOWN_HINT_RE.search("[link](http://x.com)")
    assert _MARKDOWN_HINT_RE.search("> quote")
    assert _MARKDOWN_HINT_RE.search("---")


def test_markdown_hint_re_no_match_plain():
    """Plain text should not match _MARKDOWN_HINT_RE."""
    assert not _MARKDOWN_HINT_RE.search("Just plain text")
    assert not _MARKDOWN_HINT_RE.search("Hello world 123")


if __name__ == "__main__":
    import traceback

    tests = [
        test_table_detection,
        test_no_false_positive_on_bold,
        test_outbound_payload_table_forces_text,
        test_outbound_payload_table_with_bold_forces_text,
        test_outbound_payload_bold_uses_post,
        test_outbound_payload_plain_text,
        test_convert_tables_to_code_blocks_basic,
        test_convert_tables_to_code_blocks_with_surrounding_text,
        test_convert_tables_to_code_blocks_multiple_tables,
        test_convert_tables_to_code_blocks_table_with_bold,
        test_convert_tables_to_code_blocks_no_table,
        test_convert_tables_to_code_blocks_trailing_newline,
        test_fence_boundary_4_backticks,
        test_fence_boundary_tilde,
        test_fence_boundary_mixed_fences,
        test_fence_boundary_nested_4_in_6,
        test_extract_post_text_basic,
        test_markdown_hint_re_patterns,
        test_markdown_hint_re_no_match_plain,
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
