import asyncio
from playwright.async_api import async_playwright

async def run(playwright):
    browser = await playwright.chromium.launch()
    # Set a large enough viewport for a good screenshot
    context = await browser.new_context(viewport={'width': 1600, 'height': 900})
    page = await context.new_page()
    
    print("Navigating to http://localhost:8080/")
    await page.goto("http://localhost:8080/")
    
    # Wait for the network to be idle and elements to load
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(2) # Extra wait for canvas rendering just in case
    
    print("Taking screenshot 1 (Default: Test 1, Multi-Camera)...")
    await page.screenshot(path="results/gui_screenshot_1.png", full_page=True)
    
    print("Selecting Test 3 (90% capacity)...")
    await page.select_option("#testSelect", "3")
    await page.click("#btnRunPipeline")
    await asyncio.sleep(8) # Wait for pipeline to run and render
    
    print("Taking screenshot 2 (Test 3)...")
    await page.screenshot(path="results/gui_screenshot_2.png", full_page=True)
    
    await browser.close()
    
async def main():
    async with async_playwright() as playwright:
        await run(playwright)

if __name__ == '__main__':
    asyncio.run(main())
