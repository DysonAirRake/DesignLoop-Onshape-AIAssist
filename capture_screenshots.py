"""
Capture report screenshots — DesignLoop full first-run walkthrough.
Saves PNGs to report_screenshots/. Run after clearing prior partial run.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

BASE    = "http://localhost:5000"
OUT_DIR = Path(__file__).parent / "report_screenshots"
OUT_DIR.mkdir(exist_ok=True)

VIEWPORT = {"width": 400, "height": 820}
n = 0

async def shot(page, label):
    global n
    n += 1
    fname = OUT_DIR / f"{n:02d}_{label}.png"
    await page.screenshot(path=str(fname))
    print(f"  saved: {fname.name}")


async def run():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, slow_mo=80)
        ctx     = await browser.new_context(viewport=VIEWPORT)
        page    = await ctx.new_page()

        # ── Load and clear all cached state ──────────────────────────────────
        await page.goto(BASE, wait_until="domcontentloaded")
        await page.evaluate("""() => {
            localStorage.removeItem('designloop_onboard_v1');
            localStorage.removeItem('designloop_history_tip_shown');
            localStorage.removeItem('designloop_lastdoc_v1');
            localStorage.removeItem('designloop_doclist_v1');
        }""")
        await page.reload(wait_until="domcontentloaded")
        await page.wait_for_timeout(1800)

        # ── 01. Initial cold state ────────────────────────────────────────────
        await shot(page, "initial_state")
        print("  [01] Initial cold state")

        # ── 02. Document picker ───────────────────────────────────────────────
        await page.evaluate("showDocumentPicker()")
        try:
            await page.wait_for_selector(".doc-picker-item", timeout=15000)
        except Exception:
            pass
        await page.wait_for_timeout(800)
        await shot(page, "document_picker")
        print("  [02] Document picker")

        items = await page.query_selector_all(".doc-picker-item")
        if not items:
            print("  ERROR: no picker items")
            await browser.close()
            return
        for i, item in enumerate(items[:6]):
            el = await item.query_selector(".doc-picker-name")
            if el:
                print(f"    {i+1}. {await el.inner_text()}")

        # ── 03. Click first doc ───────────────────────────────────────────────
        await items[0].click()
        await page.wait_for_timeout(2500)
        await page.evaluate("window.scrollTo(0,0)")
        await page.wait_for_timeout(300)
        await shot(page, "connected_load_views_tooltip")
        print("  [03] Connected — Load Views tooltip")

        # ── 04. Tooltip close-up ─────────────────────────────────────────────
        await page.evaluate("""
            document.getElementById('loadBtn').closest('div[style]')
              .scrollIntoView({block:'center'})
        """)
        await page.wait_for_timeout(400)
        await shot(page, "load_views_tooltip_closeup")
        print("  [04] Load Views tooltip close-up")

        # ── 05. Click Load or Refresh Views (whichever is visible) ───────────
        # If views are server-cached the UI shows Refresh instead of Load
        load_visible = await page.is_visible("#loadBtn")
        if load_visible:
            await page.click("#loadBtn")
        else:
            refresh_visible = await page.is_visible("#refreshBtn")
            if refresh_visible:
                await page.click("#refreshBtn")
            else:
                # Neither visible — trigger via JS
                await page.evaluate("loadViews(false)")
        await page.wait_for_timeout(1200)
        await shot(page, "views_loading")
        print("  [05] Views loading")

        try:
            await page.wait_for_selector(".view-tab", timeout=25000)
        except Exception:
            pass
        await page.wait_for_timeout(1200)

        # ── 06. Views loaded + Analyse tooltip ───────────────────────────────
        await page.evaluate("window.scrollTo(0,0)")
        await page.wait_for_timeout(300)
        await shot(page, "views_loaded_analyse_tooltip")
        print("  [06] Views loaded — Analyse tooltip")

        # ── 07. View tabs close-up ────────────────────────────────────────────
        await page.evaluate("document.getElementById('viewTabs').scrollIntoView({block:'center'})")
        await page.wait_for_timeout(400)
        await shot(page, "view_tabs_and_display")
        print("  [07] View tabs and display")

        # ── 08. Type description ──────────────────────────────────────────────
        await page.evaluate("document.getElementById('descInput').scrollIntoView({block:'center'})")
        await page.wait_for_timeout(300)
        desc = await page.query_selector("#descInput")
        await desc.click()
        await desc.type(
            "A headphone holder made from bent sheet metal with a horizontal arm "
            "that hooks over a desk edge and a curved cradle that holds the headphones.",
            delay=14
        )
        await page.wait_for_timeout(500)
        await shot(page, "description_typed")
        print("  [08] Description typed — auto-expanding textarea")

        # ── 09. Analyse button + tooltip ──────────────────────────────────────
        await page.evaluate("document.getElementById('analyseBtn').scrollIntoView({block:'center'})")
        await page.wait_for_timeout(300)
        await shot(page, "analyse_button_with_tooltip")
        print("  [09] Analyse button with tooltip")

        # ── 10. Click Analyse — loading state ─────────────────────────────────
        await page.click("#analyseBtn")
        await page.wait_for_timeout(1400)
        await shot(page, "analysis_loading")
        print("  [10] Analysis loading")

        # Wait for response
        try:
            await page.wait_for_function(
                "document.getElementById('aiText').textContent.length > 80",
                timeout=55000
            )
        except Exception:
            pass
        await page.wait_for_timeout(800)

        # ── 11. AI analysis result ────────────────────────────────────────────
        await page.evaluate("window.scrollTo(0,0)")
        await page.wait_for_timeout(300)
        await shot(page, "ai_analysis_result")
        print("  [11] AI analysis result (panel 2)")

        # ── 12. Scroll to read more ───────────────────────────────────────────
        await page.evaluate("document.getElementById('aiText').scrollIntoView({block:'start'})")
        await page.wait_for_timeout(300)
        await shot(page, "ai_analysis_scrolled")
        print("  [12] AI analysis scrolled")

        # ── 13. Continue to Generation ────────────────────────────────────────
        # The Continue button is at the bottom of panel2
        await page.evaluate("""
            document.querySelector('#panel2 .btn-nav').scrollIntoView({block:'center'})
        """)
        await page.wait_for_timeout(400)
        await page.evaluate("document.querySelector('#panel2 .btn-nav').click()")
        await page.wait_for_timeout(900)
        await page.evaluate("window.scrollTo(0,0)")
        await page.wait_for_timeout(400)
        await shot(page, "tab3_overview")
        print("  [13] Tab 3 — overview with Finish banner + Generate tooltip")

        # ── 14. Finish Iteration banner (tab 3) ───────────────────────────────
        await page.evaluate("document.getElementById('finishPromptBanner').scrollIntoView({block:'center'})")
        await page.wait_for_timeout(300)
        await shot(page, "tab3_finish_iteration_banner")
        print("  [14] Finish Iteration banner (tab 3)")

        # ── 15. Generate Concept tooltip ──────────────────────────────────────
        await page.evaluate("document.getElementById('generateBtn').scrollIntoView({block:'center'})")
        await page.wait_for_timeout(300)
        await shot(page, "generate_concept_tooltip")
        print("  [15] Generate Concept tooltip")

        # ── 16. Type modification direction ───────────────────────────────────
        nxt = await page.query_selector("#nextInput")
        await nxt.click()
        await nxt.type(
            "Add a cable management hook underneath the cradle arm to keep the headphone cable tidy.",
            delay=15
        )
        await page.wait_for_timeout(400)
        await shot(page, "modification_typed")
        print("  [16] Modification typed")

        # ── 17. Click Generate — loading ──────────────────────────────────────
        await page.click("#generateBtn")
        await page.wait_for_timeout(1600)
        await shot(page, "image_generation_loading")
        print("  [17] Image generation loading")

        # Wait for image
        try:
            await page.wait_for_function(
                "!!document.querySelector('#conceptImgWrap img')",
                timeout=120000
            )
        except Exception:
            try:
                await page.wait_for_function(
                    "!document.getElementById('generateBtn').disabled",
                    timeout=120000
                )
            except Exception:
                pass
        await page.wait_for_timeout(1000)

        # ── 18. Concept image generated ───────────────────────────────────────
        await page.evaluate("window.scrollTo(0,0)")
        await page.wait_for_timeout(300)
        await shot(page, "concept_image_generated_full")
        print("  [18] Concept image generated — full view")

        # ── 19. Gallery card ──────────────────────────────────────────────────
        await page.evaluate("document.getElementById('conceptGalleryCard').scrollIntoView({block:'start'})")
        await page.wait_for_timeout(300)
        await shot(page, "concept_image_gallery")
        print("  [19] Concept gallery")

        # ── 20. Post-generate banner ───────────────────────────────────────────
        post = await page.query_selector("#postGenBanner")
        if post and await post.is_visible():
            await page.evaluate("window.scrollTo(0,0)")
            await page.wait_for_timeout(300)
            await shot(page, "post_generate_guidance_banner")
            print("  [20] Post-generate guidance banner")

        # ── 21. Finish Iteration ───────────────────────────────────────────────
        await page.evaluate("window.scrollTo(0,0)")
        await page.wait_for_timeout(300)
        await page.click("#newIterBtn")
        await page.wait_for_timeout(2200)
        await page.evaluate("window.scrollTo(0,0)")
        await page.wait_for_timeout(400)
        await shot(page, "iteration_finished_iteration2")
        print("  [21] Iteration finished — Iteration 2 counter")

        # ── 22. History tooltip (one-time) ────────────────────────────────────
        await page.wait_for_timeout(1000)
        hist_tip = await page.query_selector("#historyTip")
        if hist_tip and await hist_tip.is_visible():
            await shot(page, "history_button_tutorial_tip")
            print("  [22] History button one-time tutorial tip")

        # ── 23. Open History modal ─────────────────────────────────────────────
        hist_btn = await page.query_selector("#historyBtn")
        if hist_btn and await hist_btn.is_visible():
            await hist_btn.click()
            await page.wait_for_timeout(1800)
            await shot(page, "history_modal")
            print("  [23] History modal — iteration log")
            close = await page.query_selector("#historyModal button")
            if close:
                await close.click()
            await page.wait_for_timeout(400)

        # ── 24. View Report link ───────────────────────────────────────────────
        report = await page.query_selector("#reportLink")
        if report and await report.is_visible():
            await shot(page, "header_with_history_and_report_buttons")
            print("  [24] Header — History + View Report buttons")

        print(f"\n{'='*54}")
        print(f"  {n} screenshots saved to:")
        print(f"  {OUT_DIR}")
        print(f"{'='*54}")
        print("Browser open 20s for review…")
        await page.wait_for_timeout(20000)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
