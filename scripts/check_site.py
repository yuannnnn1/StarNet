"""Check all local website resources and JSON-backed assets over HTTP."""
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls=[]
    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src') and value:
                self.urls.append(value)


def main():
    base='http://127.0.0.1:8000/website/'
    parser=Links()
    with urlopen(base) as response: parser.feed(response.read().decode())
    checked=[]
    urls=parser.urls+['../artifacts/results/dashboard.json','../docs/repository.json']
    dashboard=json.loads(Path('artifacts/results/dashboard.json').read_text())
    for examples in dashboard['examples'].values():
        urls.extend('../artifacts/figures/'+example['image'] for example in examples)
    for name in ['baseline','starnet','ablation']:
        urls.append(f'../artifacts/figures/confusion_matrix_{name}.png')
    urls.append('../artifacts/figures/training_loss.png')
    for target in sorted(set(urls)):
        if target.startswith('#'): continue
        url=urljoin(base,target)
        if urlparse(url).netloc!='127.0.0.1:8000': continue
        with urlopen(url) as response:
            assert response.status==200 and len(response.read())>0,url
        checked.append(url)
    result={'passed':True,'checked_urls':checked,'scope':'HTTP resources; interactive and responsive UI checked separately'}
    Path('artifacts/results/website_http_checks.json').write_text(json.dumps(result,indent=2))
    print(f'All {len(checked)} local resources returned nonempty HTTP 200 responses')


if __name__=='__main__': main()
