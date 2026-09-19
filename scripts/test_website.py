"""Local-only browser checks for the generated dashboard."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    output=Path('artifacts/website_qa'); output.mkdir(parents=True,exist_ok=True)
    checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[]; page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8000/website/',wait_until='networkidle')
        assert page.locator('#status').inner_text()==''
        assert page.locator('#comparison-body tr').count()==3
        assert '86.14%' in page.locator('#summary').inner_text()
        assert '0.54' in page.locator('#ablation-result').inner_text()
        assert page.locator('#repository-status a').get_attribute('href')=='https://github.com/yuannnnn1/StarNet'
        page.screenshot(path=str(output/'desktop.png'),full_page=True)
        checks.append('Desktop loads saved metrics and supplied repository link')
        page.locator('#metric').select_option('loss')
        assert 'training_loss.png' in page.locator('#training-chart').get_attribute('src')
        page.locator('#metric').select_option('accuracy')
        for model in ['baseline','ablation','starnet']:
            page.locator('#confusion-model').select_option(model)
            assert f'confusion_matrix_{model}.png' in page.locator('#confusion-chart').get_attribute('src')
            page.locator('#prediction-model').select_option(model)
            for outcome in ['correct','incorrect','all']:
                page.locator('#prediction-filter').select_option(outcome)
                assert page.locator('.example').count()==(12 if outcome=='all' else 6)
        checks.append('Metric switch, all confusion models and all prediction model/outcome combinations work')
        for width,height in [(390,844),(320,740)]:
            page.set_viewport_size({'width':width,'height':height})
            page.goto('http://127.0.0.1:8000/website/',wait_until='networkidle')
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.locator('#predictions').scroll_into_view_if_needed()
            page.screenshot(path=str(output/f'mobile-{width}-predictions.png'))
            page.locator('#overview').scroll_into_view_if_needed()
            page.screenshot(path=str(output/f'mobile-{width}.png'))
            checks.append(f'{width}px mobile viewport has no page overflow; overview/prediction screenshots captured')
        page.set_viewport_size({'width':1440,'height':1000})
        for image in page.locator('img').all(): image.scroll_into_view_if_needed()
        assert page.locator('img').evaluate_all('(images) => images.every(img => img.complete && img.naturalWidth > 0)')
        page.locator('#overview').scroll_into_view_if_needed()
        page.screenshot(path=str(output/'desktop.png'),full_page=True)
        page.screenshot(path=str(output/'desktop-overview.png'))
        page.locator('#confusion').scroll_into_view_if_needed()
        page.screenshot(path=str(output/'desktop-confusion.png'))
        assert not errors,errors
        checks.append('All displayed image assets decoded; no JavaScript page errors')
        browser.close()
    Path('artifacts/results/website_ui_checks.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
    print('\n'.join(checks))


if __name__=='__main__': main()
