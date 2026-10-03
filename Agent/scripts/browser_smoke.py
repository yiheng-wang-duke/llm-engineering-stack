"""Optional: pip install playwright && python -m playwright install chromium."""

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    screenshots = root / "docs" / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="agent-ui-") as data:
        env = {
            **os.environ,
            "AW_MODE": "demo",
            "AW_DATA_DIR": data,
            "AW_API_TOKEN": "",
            "AW_EMBEDDING_BACKEND": "bm25",
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "workbench", "--port", str(port)],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(100):
                try:
                    if httpx.get(base + "/healthz").is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("Temporary UI server did not start")
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={"width": 1440, "height": 1000}, device_scale_factor=1
                )
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base)
                expect(page.locator("#mode-badge")).to_have_text("离线演示 · 无模型")
                expect(page.locator("#welcome")).to_be_visible()
                page.screenshot(path=str(screenshots / "desktop.png"), full_page=True)
                page.locator('[data-view="knowledge"]').click()
                page.locator("#load-example").click()
                expect(page.locator("#doc-count")).to_have_text("1")
                page.locator("#search-query").fill("项目架构")
                page.locator("#search-form button").click()
                expect(page.locator("#search-results .source-card").first).to_be_visible()
                page.locator("#document-list button").first.click()
                expect(page.locator("#detail-dialog")).to_be_visible()
                page.locator("#close-dialog").click()
                page.locator('[data-view="memory"]').click()
                page.locator("#memory-content").fill("我喜欢中文和 Python 示例。")
                page.locator("#memory-form button").click()
                expect(page.locator("#memory-count")).to_have_text("1")
                page.locator("#new-chat").click()
                page.locator("#prompt").fill("请介绍项目架构")
                page.locator("#send-button").click()
                expect(page.locator("#run-status")).to_have_text("完成")
                expect(page.locator(".message.assistant .message-body")).to_contain_text("离线演示")
                expect(page.locator(".message.assistant .message-body")).to_contain_text("Python")
                expect(page.locator("#sources .source-card").first).to_be_visible()
                page.screenshot(path=str(screenshots / "conversation.png"), full_page=True)
                with page.expect_download() as download:
                    page.locator("#export-chat").click()
                assert download.value.suggested_filename.endswith(".json")
                page.reload()
                page.locator(".session-select").first.click()
                expect(page.locator(".message")).to_have_count(2)
                page.locator("#new-chat").click()
                page.set_viewport_size({"width": 390, "height": 844})
                expect(page.locator("#menu-button")).to_be_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                page.locator("#toggle-trace").click()
                expect(page.locator("#close-trace")).to_be_visible()
                page.locator("#close-trace").click()
                expect(page.locator("#trace-panel")).not_to_be_visible()
                page.screenshot(path=str(screenshots / "mobile.png"), full_page=True)
                page.locator("#menu-button").click()
                page.locator('[data-view="knowledge"]').click()
                expect(page.locator("#view-knowledge")).to_be_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                browser.close()
                assert not errors, errors
            print(
                "PASS: browser upload/search/memory/chat/citations/export/history/mobile; no JS errors"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
