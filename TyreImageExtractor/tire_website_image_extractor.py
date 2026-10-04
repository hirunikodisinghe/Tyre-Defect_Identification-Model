import argparse
import asyncio
import csv
import hashlib
import io
import re
import time
import threading
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qs
from urllib.robotparser import RobotFileParser

SITES = {
    'bkt': ('BKT', ['https://commercial.bkt-tires.com/industrial-construction/',
        'https://commercial.bkt-tires.com/pattern/maglift-plus/']),
    'michelin': ('Michelin', ['https://business.michelinman.com/tires',
        'https://business.michelinman.com/tires/michelin-x-tra-power-l5']),
    'continental': ('Continental', ['https://www.continental-tires.com/us/en/products/earthmoving/',
        'https://www.continental-tires.com/products/earthmoving/tires/em-master-e4-l4/']),
    'bridgestone': ('Bridgestone', ['https://commercial.bridgestone.com/en-us/off-the-road',
        'https://commercial.bridgestone.com/en-us/tires/otr/vut']),
    'goodyear': ('Goodyear', ['https://www.goodyearotr.com/tires',
        'https://www.goodyearotr.com/tires/tire-details?tireId=837']),
    'yokohama_atg': ('Yokohama ATG (Alliance, Galaxy, Primex)', ['https://www.yokohama-atg.com/products',
        'https://www.yokohama-atg.com/en-in/product-detail/hulk']),
    'titan': ('Titan', ['https://www.titan-intl.com/en/tires/construction',
        'https://www.titan-intl.com/Tires/ULTIMATE']),
    'mitas': ('Mitas', ['https://www.mitas-tires.com/en-us/products/material-handling-tires',
        'https://www.mitas-tires.com/en/products/material-handling-tires/forklifts/forklifts-tyres/fl-08']),
    'maxam': ('MAXAM', ['https://maxamtire.com/tires/industry-industrial-forklift-tires/',
        'https://maxamtire.com/tire/ms202/']),
    'ceat': ('CEAT Specialty', ['https://www.ceatspecialty.com/products/construction-industrial-tires',
        'https://www.ceatspecialty.com/forklift-tires']),
    'double_coin': ('Double Coin', ['https://www.doublecointires.com/tires/off-the-road/',
        'https://www.doublecointires.com/tire/rem-3/']),
}
STARTS = {b: seeds[0] for b, (_, seeds) in SITES.items()}
STOP_PATH = re.compile(r'/(?:contact(?:-us)?|blogs?|news|press-release|privacy|cookies?|'
                       r'about(?:-us)?|careers|warranty|downloads?|dealers?)(?:/|$)', re.I)
INDUSTRIAL = re.compile(r'industrial|construction|forklift|material.handling|earthmov|'
                        r'off.the.road|\botr\b|mining|quarry|loader|grader|telehandler|'
                        r'compactor|container.handler|skid.steer|port.handling', re.I)
AGENT = 'TyreImageReviewBot/1.0'
FIELDS = ['brand', 'product_name', 'construction', 'product_url', 'image_url',
          'local_filename', 'width', 'height', 'phash', 'sha256',
          'review_status', 'usage_rights']
TYRE = re.compile(r'\b(tyres?|tires?|pneumatic|solid|forklift|radial)\b', re.I)
EXCLUDE = re.compile(r'\b(tweel|retread(?:ed|ing)?|regroov(?:e|ed|ing)|rubber tracks?)\b', re.I)
DECORATION = re.compile(r'logo|favicon|icon|sprite|banner|placeholder|spinner|cookie|social', re.I)


def clean_url(url):
    p = urlsplit(url)
    known = {urlsplit(s).hostname for _, seeds in SITES.values() for s in seeds}
    scheme = 'https' if p.hostname in known else p.scheme
    return urlunsplit((scheme, p.netloc.lower(), re.sub('/+', '/', p.path), p.query, ''))


def brand_hosts(brand):
    hosts = {urlsplit(s).hostname for s in SITES[brand][1]}
    return hosts | {h[4:] if h.startswith('www.') else 'www.' + h for h in hosts}


def page_kind(url, brand):
    """Classify paths, never search for keywords in the hostname."""
    p = urlsplit(url)
    if p.scheme != 'https' or p.hostname not in brand_hosts(brand):
        return None
    path = p.path.lower().rstrip('/')
    if STOP_PATH.search(path) or re.search(r'\.(pdf|zip|jpe?g|png|webp|svg)$', path):
        return None
    if brand == 'bkt':
        if re.fullmatch(r'/pattern/[^/]+', path):
            return 'product'
        if path == '/industrial-construction':
            return 'listing'
        sector = parse_qs(p.query).get('sector', [])
        if path in ('/products', '/advanced-search') and sector == ['industrial']:
            return 'listing'
        return None
    if brand == 'michelin':
        if re.fullmatch(r'/tires/michelin-[^/]+', path):
            return 'product'
        if path == '/tires':
            return 'listing'
    elif brand == 'continental':
        if re.fullmatch(r'(?:/[a-z]{2}/en)?/products/(earthmoving|material-handling)/tires/[^/]+', path):
            return 'product'
        if re.fullmatch(r'(?:/[a-z]{2}/en)?/products/(earthmoving|material-handling)(?:/[^/]+)?', path):
            return 'listing'
    elif brand == 'bridgestone':
        if re.fullmatch(r'/en-us/tires/otr/[^/]+', path):
            return 'product'
        if path in ('/en-us/off-the-road', '/en-us/tires/otr'):
            return 'listing'
    elif brand == 'goodyear':
        if path == '/tires/tire-details' and parse_qs(p.query).get('tireId'):
            return 'product'
        if path == '/tires':
            return 'listing'
    elif brand == 'yokohama_atg':
        if re.fullmatch(r'(?:/en-[a-z]{2})?/product-detail/[^/]+', path):
            return 'product'
        if re.fullmatch(r'(?:/en-[a-z]{2})?/products(?:/[^/]+)?', path):
            return 'listing'
    elif brand == 'titan':
        if re.fullmatch(r'(?:/en)?/tires/(construction|mining)', path):
            return 'listing'
        if re.fullmatch(r'(?:/en)?/tires/[^/]+', path) and path.split('/')[-1] not in (
                'agriculture', 'forestry', 'atv-utv', 'consumer'):
            return 'product'
    elif brand == 'mitas':
        prefix = r'/(?:en|en-us)/products/(?:material-handling-tires|construction-tires)'
        if re.fullmatch(prefix + r'(?:/[^/]+){2,}', path):
            if path.endswith(('-tyres', '-tires')):
                return 'listing'
            return 'product'
        if re.fullmatch(prefix + r'(?:/[^/]+)?', path):
            return 'listing'
    elif brand == 'maxam':
        if re.fullmatch(r'/tire/[^/]+', path):
            return 'product'
        if re.fullmatch(r'/tires(?:/industry-(?:industrial-forklift|construction|off-the-road|solid-otr|mining|underground-mining)-tires)?', path):
            return 'listing'
    elif brand == 'ceat':
        if re.fullmatch(r'(?:/[a-z]{2})?/products/[^/]+/[^/]+', path):
            return 'product'
        if re.fullmatch(r'(?:/[a-z]{2})?/products/(construction-industrial|port|mining)-tires', path) or path == '/forklift-tires':
            return 'listing'
    elif brand == 'double_coin':
        if re.fullmatch(r'/tire/[^/]+', path):
            return 'product'
        if path == '/tires/off-the-road':
            return 'listing'
    return None


def product_evidence(data, url):
    if data.get('product_name'):
        return True
    title = data['title'].lower().strip()
    if re.search(r'^(products?|tires?|tyres?|features|forklifts?|construction|material handling)(?:\s|$)', title):
        return False
    model = re.sub(r'[^a-z0-9]', '', urlsplit(url).path.rstrip('/').split('/')[-1].lower())
    heading = re.sub(r'[^a-z0-9]', '', title)
    return bool(model) and (model in heading or (len(heading) >= 3 and heading in model))


def next_links(links, current, brand, hosts):
    # Do not wander into unrelated 'recommended products' from product pages.
    if page_kind(current, brand) != 'listing':
        return []
    accepted = {}
    for value in links:
        url = clean_url(urljoin(current, value))
        kind = page_kind(url, brand)
        if urlsplit(url).hostname in hosts and kind:
            accepted[url] = kind
    return sorted(accepted, key=lambda u: accepted[u] != 'product')


def image_matches(title, image):
    # Structured product images and page-specific OG images are candidates;
    # ordinary images must mention the actual model, not just 'tyre'.
    if image.get('primary'):
        return True
    words = re.findall(r'[a-z0-9]+', title.lower())
    words = [w for w in words if w not in {'bkt', 'michelin', 'continental', 'bridgestone',
        'goodyear', 'yokohama', 'galaxy', 'alliance', 'primex', 'titan', 'mitas', 'maxam',
        'ceat', 'double', 'coin', 'tire', 'tires', 'tyre', 'tyres', 'commercial', 'industrial', 'construction'}]
    evidence = set(re.findall(r'[a-z0-9]+', (image.get('alt', '') + ' ' +
                        urlsplit(image.get('src') or '').path).lower()))
    return bool(words) and all(w in evidence for w in words)


PAGE_SCRIPT = r'''() => {
  const root = document.querySelector('main') || document.body;
  const copy = root.cloneNode(true);
  copy.querySelectorAll('nav,header,footer,aside,script,style,[role="navigation"]').forEach(n=>n.remove());
  let title = (root.querySelector('h1')?.textContent || document.title.split('|')[0]).trim();
  if (/^(features|specifications|products|tires|tyres)\b/i.test(title)) title = document.title.split('|')[0].trim();
  const images = [];
  const products = [];
  function addImage(value) {
    if (typeof value === 'string') images.push({src:value,srcset:'',alt:title,primary:true});
    else if (Array.isArray(value)) value.forEach(addImage);
    else if (value && typeof value === 'object') addImage(value.url || value.contentUrl);
  }
  function walk(value) {
    if (!value || typeof value !== 'object') return;
    if (Array.isArray(value)) {value.forEach(walk);return;}
    if ([].concat(value['@type'] || []).some(t=>/^(?:https?:\/\/schema.org\/)?Product$/.test(t))) products.push(value);
    Object.values(value).forEach(v=>{if(v && typeof v==='object') walk(v);});
  }
  document.querySelectorAll('script[type="application/ld+json"]').forEach(s=>{
    try {walk(JSON.parse(s.textContent));} catch(e) {}
  });
  const normal = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g,'');
  const matching = products.filter(p=>p.name && normal(title).includes(normal(p.name)));
  const product = matching.length===1 ? matching[0] : (products.length===1 ? products[0] : null);
  if (product) {title = product.name || title; addImage(product.image);}
  const og = document.querySelector('meta[property="og:image"]')?.content;
  if (og) images.push({src:og,srcset:'',alt:title,primary:true});
  copy.querySelectorAll('img').forEach(i=>images.push({
    src:i.getAttribute('data-src') || i.getAttribute('data-lazy-src') || i.getAttribute('src'),
    srcset:i.getAttribute('srcset') || i.getAttribute('data-srcset') || '',
    alt:i.getAttribute('alt') || '',primary:false}));
  copy.querySelectorAll('[class*="related"],[class*="recommend"],.newsletter').forEach(n=>n.remove());
  const description = product?.description || document.querySelector('meta[name="description"]')?.content || '';
  return {title,product_name:product?.name || '',description,
    text:copy.textContent.slice(0,12000),
    links:[...copy.querySelectorAll('a[href]')].map(a=>a.getAttribute('href')),images};
}'''


def construction(text):
    solid = bool(re.search(r'\bsolid\b|press[ -]on|cushion tyre|cushion tire', text, re.I))
    pneumatic = bool(re.search(r'\bpneumatic\b', text, re.I))
    if solid and not pneumatic:
        return 'solid-review-required'
    if pneumatic and not solid:
        return 'pneumatic-review-required'
    return 'unknown-review-required'


def eligible(product_text):
    # Deliberately does not exclude 'airless': solid tyres may use that word.
    return bool(TYRE.search(product_text)) and not bool(EXCLUDE.search(product_text))


def self_test():
    cases = [('Solid forklift tyre', True), ('Pneumatic industrial tyre', True),
             ('Solid airless forklift tyre', True), ('Retreaded tyre', False),
             ('Regrooved tyre', False), ('Tweel airless tyre', False)]
    for value, expected in cases:
        assert eligible(value) == expected, value
    assert construction('Solid pneumatic-shaped tyre') == 'unknown-review-required'
    assert construction('Solid airless tyre') == 'solid-review-required'
    assert construction('Pneumatic tyre') == 'pneumatic-review-required'
    assert construction('Industrial tyre') == 'unknown-review-required'
    host = 'commercial.bkt-tires.com'
    base = 'https://' + host
    cases = [('/', None), ('/contact-us/', None), ('/bkt-blog/', None),
             ('/agriculture/', None), ('/otr/', None), ('/products/', None),
             ('/products/?sector=agriculture', None),
             ('/products/?sector=industrial&machinery=forklift', 'listing'),
             ('/advanced-search/?sector=industrial', 'listing'),
             ('/industrial-construction/', 'listing'),
             ('/pattern/maglift-plus/', 'product')]
    for path, expected in cases:
        assert page_kind(base + path, 'bkt') == expected, path
    links = ['/contact-us/', '/products/?sector=industrial', '/pattern/maglift-plus/',
             'https://example.com/pattern/foo', '/pattern/maglift-plus/']
    result = next_links(links, base + '/industrial-construction/', 'bkt', {host})
    assert result == [base + '/pattern/maglift-plus/', base + '/products/?sector=industrial']
    assert next_links(links, base + '/pattern/maglift-plus/', 'bkt', {host}) == []
    assert clean_url('http://' + host + '/industrial-construction/#top') == base + '/industrial-construction/'
    assert image_matches('MAGLIFT PLUS', {'src':'/images/maglift-plus.jpg','alt':''})
    assert not image_matches('MAGLIFT PLUS', {'src':'/images/tractor.jpg','alt':'Agriculture tyre'})
    count = 26
    assert len(SITES) == 11
    count += 1
    for brand, (_, seeds) in SITES.items():
        for seed in seeds:
            assert page_kind(seed, brand) in ('listing', 'product'), (brand, seed)
            count += 1
        base = 'https://' + urlsplit(seeds[0]).netloc
        for bad in ['/contact-us/', '/blog/', '/news/', '/privacy/', '/catalogue.pdf']:
            assert page_kind(base + bad, brand) is None, (brand, bad)
            count += 1
        assert page_kind('https://unrelated.example/tire/example', brand) is None
        count += 1
    assert page_kind('https://www.goodyearotr.com/tires/tire-details', 'goodyear') is None
    assert page_kind('https://www.ceatspecialty.com/us/products/agriculture-tires/liftpro', 'ceat') == 'product'
    assert page_kind('https://maxamtire.com/tires/industry-rubber-tracks/', 'maxam') is None
    assert not product_evidence({'title':'Forklift tyres'}, 'https://www.mitas-tires.com/en/products/material-handling-tires/forklifts/forklifts-tyres')
    assert product_evidence({'title':'REM-3'}, 'https://www.doublecointires.com/tire/rem-3/')
    assert product_evidence({'title':'RT-3B', 'product_name':'RT-3B'}, 'https://www.goodyearotr.com/tires/tire-details?tireId=837')
    count += 6
    count += regression_tests()
    print(f'PASS: {count} offline checks across all 11 manufacturers.')
    print('Live download availability is not tested by --self-test.')


def regression_tests():
    """Offline responses exercise the actual Access class; no dependencies needed."""
    import contextlib
    from types import SimpleNamespace

    class Response:
        def __init__(self, status, text='', headers=None):
            self.status_code, self.text, self.headers = status, text, headers or {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class Session:
        def __init__(self, responses):
            self.responses, self.calls = deque(responses), []
        def get(self, url, **kwargs):
            self.calls.append(url)
            response = self.responses.popleft()
            if isinstance(response, Exception):
                raise response
            return response

    count = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for status, expected in [(404, True), (410, True), (401, False),
                                 (403, False), (429, False), (500, False), (503, False)]:
            session = Session([Response(status)])
            access = Access(session, 0)
            assert access.allowed('https://test.example/image.jpg') is expected
            assert access.allowed('https://test.example/second.jpg') is expected
            assert len(session.calls) == 1  # Host policy cached.
            count += 3
        for responses, expected in [
            ([Response(200, 'User-agent: *\nDisallow: /private')], False),
            ([Response(200, '<html>Sign in</html>')], False),
            ([OSError('offline')], False),
            ([Response(301, headers={'Location': '/rules.txt'}),
              Response(200, 'User-agent: *\nDisallow:')], True),
            ([Response(302, headers={'Location': 'https://cdn.example/robots.txt'}),
              Response(403)], False),
            ([Response(302, headers={'Location': '/robots.txt'}) for _ in range(6)], False),
        ]:
            access = Access(Session(responses), 0)
            assert access.allowed('https://test.example/private/image.jpg') is expected
            count += 1
    assert industrial_product({'title': 'Solid forklift tyre', 'description': ''})
    assert industrial_product({'title': 'Example', 'description': 'Pneumatic loader tyre'})
    assert not industrial_product({'title': 'Agilis', 'description': 'Light truck tyre', 'text': 'industrial forklift'})
    assert not industrial_product({'title': 'Road tyre', 'description': '', 'text': 'industrial forklift'})
    args = SimpleNamespace(start_url=None, max_pages=2)
    order = []
    for state in crawl_schedule(list(SITES), args):
        order.append(state['brand'])
        state['seen'].add(state['queue'].popleft())
    assert order == list(SITES) * 2
    return count + 5


def safe_url(url):
    """Do not print query tokens/API keys in diagnostics."""
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path, '', ''))


def industrial_product(data):
    """Prefer model-specific evidence over whole-page menus and related items."""
    text = ' '.join(str(data.get(k, '')) for k in ('title', 'description'))
    # These road-tyre families were seen slipping through generic page text.
    if re.search(r'\bagilis\b|\bpassenger\b|\blight[ -]truck\b', text, re.I):
        return False
    return bool(INDUSTRIAL.search(text)) and not bool(EXCLUDE.search(text))


def crawl_schedule(brands, args):
    """Rotate one page per brand; each brand retains its own queue and budget."""
    states = []
    for brand in brands:
        seeds = [clean_url(s) for s in (args.start_url or SITES[brand][1])]
        if any(page_kind(s, brand) is None for s in seeds):
            raise ValueError('Start URL must be a supported product or listing URL.')
        states.append(dict(brand=brand, queue=deque(seeds), seen=set(),
                           scheduled=set(seeds), products=0, saved=0))
    while any(s['queue'] and len(s['seen']) < args.max_pages for s in states):
        for state in states:
            if state['queue'] and len(state['seen']) < args.max_pages:
                yield state


class Access:
    """Respect robots rules; missing files (404/410) are not a prohibition.

    Keep 401/403, server failures and network failures blocked. Follow at most
    five robots redirects; never disable TLS verification or access controls.
    """
    def __init__(self, session, delay, session_factory=None):
        self.session, self.delay = session, delay
        self.cache, self.last = {}, {}
        self.lock = threading.RLock()
        self.blocked = set()
        self.origin_locks = {}
        self.session_factory = session_factory
        self.local = threading.local()

    def http(self):
        # requests sessions contain mutable cookie state: do not share them
        # across concurrent browser callback threads.
        if self.session_factory is None:
            return self.session
        if not hasattr(self.local, 'session'):
            self.local.session = self.session_factory()
            self.local.session.headers['User-Agent'] = AGENT
        return self.local.session

    def report_block(self, url, reason):
        if (url, reason) not in self.blocked:
            self.blocked.add((url, reason))
            print(f'  BLOCKED: {reason}: {safe_url(url)}', flush=True)

    def pause(self, origin, seconds):
        time.sleep(max(0, self.last.get(origin, 0) + seconds - time.monotonic()))
        self.last[origin] = time.monotonic()

    def allowed(self, url, pace=True):
        p = urlsplit(url)
        origin = f'{p.scheme}://{p.netloc}'
        # A slow analytics host must not hold up every manufacturer/CDN.
        with self.lock:
            host_lock = self.origin_locks.setdefault(origin, threading.RLock())
        with host_lock:
            return self._allowed(url, pace)

    def _allowed(self, url, pace=True):
        p = urlsplit(url)
        if p.scheme not in ('http', 'https'):
            return False
        origin = f'{p.scheme}://{p.netloc}'
        if origin not in self.cache:
            self.pause(origin, self.delay)
            try:
                robot_url = origin + '/robots.txt'
                self.cache[origin] = None
                for hop in range(6):
                    with self.http().get(robot_url, timeout=(8, 12), allow_redirects=False) as r:
                        if r.status_code in (301, 302, 303, 307, 308):
                            target = urljoin(robot_url, r.headers.get('Location', ''))
                            if hop == 5 or not r.headers.get('Location') or urlsplit(target).scheme not in ('http', 'https'):
                                break
                            robot_url = target
                            self.pause(origin, self.delay)
                            continue
                        if r.status_code in (404, 410):
                            robot = RobotFileParser()
                            robot.parse(['User-agent: *', 'Disallow:'])
                            self.cache[origin] = robot
                            print(f'  No robots.txt ({r.status_code}): {origin}; checking individual requests normally.')
                        elif r.status_code == 200:
                            # A login/challenge HTML page is not a robots file.
                            if re.search(r'<(?:!doctype\s+html|html|body)\b', r.text, re.I):
                                print(f'  Skipping host: robots response is HTML: {origin}')
                            else:
                                robot = RobotFileParser()
                                robot.parse(r.text.splitlines())
                                self.cache[origin] = robot
                        else:
                            print(f'  Skipping host: robots.txt returned {r.status_code}: {origin}')
                        break
            except Exception as exc:
                print(f'  Skipping host: robots.txt unavailable: {origin}: {exc}')
                self.cache[origin] = None
        robot = self.cache[origin]
        if robot is None or not robot.can_fetch(AGENT, url):
            self.report_block(url, 'robots policy unavailable or disallows this URL')
            return False
        # Apply our extra delay to pages, APIs and downloads, not each static
        # script/CSS file. Explicit robots pacing still applies to all requests.
        wait = max(self.delay if pace else 0, robot.crawl_delay(AGENT) or 0)
        rate = robot.request_rate(AGENT)
        if rate and rate.requests:
            wait = max(wait, rate.seconds / rate.requests)
        self.pause(origin, wait)
        return True

    def download(self, url):
        for _ in range(6):
            if not self.allowed(url):
                print(f'  PRODUCT IMAGE BLOCKED: {safe_url(url)}', flush=True)
                return None
            with self.http().get(url, timeout=30, stream=True, allow_redirects=False) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers['Location'])
                    continue
                r.raise_for_status()
                if not r.headers.get('Content-Type', '').lower().startswith('image/'):
                    return None
                data = bytearray()
                for chunk in r.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > 30 * 1024 * 1024:
                        return None
                return bytes(data)
        return None


def optional_tracking(url):
    host = (urlsplit(url).hostname or '').lower()
    return any(host == h or host.endswith('.' + h) for h in (
        'googletagmanager.com', 'google-analytics.com', 'doubleclick.net',
        'cloudflareinsights.com', 'kameleoon.eu'))


async def load_product_page(page, url, brand, args):
    """Wait for useful content, not for every third-party script to finish."""
    from playwright.async_api import TimeoutError as BrowserTimeout
    try:
        response = await page.goto(url, wait_until='commit', timeout=args.page_timeout * 1000)
    except BrowserTimeout:
        # Only inspect a real document at a permitted URL, never chrome-error.
        if page_kind(page.url, brand) is None:
            raise RuntimeError('Navigation timed out before a supported page was available.')
        response = None
        print('  Navigation timed out; checking whether usable content arrived.')
    if response and response.status >= 400:
        raise RuntimeError(f'HTTP {response.status}; page skipped.')
    deadline = time.monotonic() + args.content_timeout
    last_data = None
    while time.monotonic() < deadline:
        await page.wait_for_timeout(500)
        if page_kind(page.url, brand) is None:
            continue
        try:
            data = await page.evaluate(PAGE_SCRIPT)
            last_data = data
            kind = page_kind(page.url, brand)
            ready = (bool(next_links(data['links'], page.url, brand, brand_hosts(brand)))
                     if kind == 'listing' else product_evidence(data, page.url) and bool(data['images']))
            if ready:
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await page.wait_for_timeout(args.render_wait * 1000)
                return await page.evaluate(PAGE_SCRIPT)
        except Exception:
            if page.is_closed():
                raise
            # Client-side navigation can briefly destroy the execution context.
            continue
    if last_data and page_kind(page.url, brand):
        print('  Content wait expired; inspecting available page content.')
        return last_data
    raise RuntimeError('No usable product/listing content within content timeout.')


async def run(args):
    import requests
    import imagehash
    from PIL import Image, ImageOps
    from playwright.async_api import async_playwright

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'tire_metadata.csv'
    hashes, urls, digests = [], set(), set()
    if manifest.exists():
        with manifest.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError('Output has an older CSV format. Choose a new --output folder.')
            for row in reader:
                if (output / row['local_filename']).exists():
                    urls.add(row['image_url'])
                    digests.add(row['sha256'])
                    hashes.append(imagehash.hex_to_hash(row['phash']))
    existing = len(digests)
    print(f'Existing unique images: {existing}; total target: {args.target_images}')
    if existing >= args.target_images:
        print('Target already reached in this output folder. No downloads needed.')
        return
    session = requests.Session()
    session.headers['User-Agent'] = AGENT
    access = Access(session, max(1.5, args.delay), requests.Session)
    brands = list(SITES) if args.brand == 'all' else [args.brand]
    saved = 0
    summary = {b: {'brand': b, 'pages_attempted': 0,
                  'product_pages_loaded': 0, 'new_images': 0} for b in brands}
    with manifest.open('a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
            f.flush()
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=not args.show_browser)
            context = await browser.new_context(user_agent=AGENT, service_workers='block')

            async def route_request(route):
                request = route.request
                # Inspect page URLs before browser navigation/redirects. Images
                # are downloaded separately through Access.download.
                if request.resource_type in ('image', 'media', 'font') or optional_tracking(request.url):
                    await route.abort()
                elif await asyncio.to_thread(access.allowed, request.url,
                                             request.resource_type not in ('script', 'stylesheet')):
                    await route.continue_()
                else:
                    await route.abort()

            await context.route('**/*', route_request)
            for state in crawl_schedule(brands, args):
                if existing + saved >= args.target_images:
                    break
                brand = state['brand']
                hosts = brand_hosts(brand)
                queue, seen, scheduled = state['queue'], state['seen'], state['scheduled']
                page_budget = len(seen) + 1
                while queue and len(seen) < page_budget:
                    url = queue.popleft()
                    if url in seen:
                        continue
                    seen.add(url)
                    print(f'[{brand}] Page {len(seen)}/{args.max_pages}: {url}')
                    # A failed navigation cannot interrupt the next page.
                    page = await context.new_page()
                    try:
                        data = await load_product_page(page, url, brand, args)
                        if urlsplit(page.url).hostname not in hosts or page_kind(page.url, brand) is None:
                            print('  Redirect ended outside the selected product/listing scope; skipped.')
                            continue
                        links = next_links(data['links'], page.url, brand, hosts)
                        products, listings = [], []
                        for link in links:
                            if link not in scheduled:
                                scheduled.add(link)
                                (products if page_kind(link, brand) == 'product' else listings).append(link)
                        queue.extendleft(reversed(products))
                        queue.extend(listings)
                        if page_kind(page.url, brand) == 'listing':
                            print(f'  Listing: queued {len(products)} product pages and {len(listings)} listings; no images downloaded here.')
                            if not links:
                                print('  No supported product links found. Check BLOCKED messages; the listing may require blocked scripts/API requests or a changed layout.')
                            continue
                        state['products'] += 1
                        if not product_evidence(data, page.url):
                            print('  No clear individual product identity in page content. Review page/layout; skipped.')
                            continue
                        if not industrial_product(data):
                            print('  No industrial application evidence in product title/description; skipped (whole-page menu text is not used).')
                            continue
                        before = saved
                        print(f'  Product: {data["title"]}; {len(data["images"])} image candidates')
                        for item in data['images']:
                            if existing + saved >= args.target_images:
                                break
                            # Use heading + image description, never the entire
                            # page/navigation. Still a heuristic, not verification.
                            text = data['title'] + ' ' + item['alt']
                            model_slug = urlsplit(page.url).path.rstrip('/').split('/')[-1]
                            if EXCLUDE.search(text) or not (image_matches(data['title'], item) or image_matches(model_slug, item)):
                                continue
                            src = item['src'] or ''
                            choices = []
                            for entry in item['srcset'].split(','):
                                parts = entry.strip().split()
                                if len(parts) == 2 and re.fullmatch(r'[0-9.]+[wx]', parts[1]):
                                    choices.append((float(parts[1][:-1]), parts[0]))
                            if choices:
                                src = max(choices)[1]
                            image_url = clean_url(urljoin(page.url, src))
                            if not src or image_url in urls or DECORATION.search(image_url + ' ' + item['alt']):
                                continue
                            try:
                                raw = await asyncio.to_thread(access.download, image_url)
                                if not raw:
                                    print('  Image unavailable/non-image/over size limit; skipped.')
                                    continue
                                digest = hashlib.sha256(raw).hexdigest()
                                if digest in digests:
                                    print('  Exact duplicate; skipped.')
                                    continue
                                with Image.open(io.BytesIO(raw)) as im:
                                    if min(im.size) < args.min_size or max(im.size) / min(im.size) > 5:
                                        print(f'  Image dimensions {im.size} fail minimum {args.min_size} or aspect-ratio filter.')
                                        continue
                                    im = ImageOps.exif_transpose(im).convert('RGBA')
                                    rgb = Image.new('RGB', im.size, 'white')
                                    rgb.paste(im, mask=im.getchannel('A'))
                                    phash = imagehash.phash(rgb)
                                    if args.phash_distance >= 0 and any(phash - previous <= args.phash_distance for previous in hashes):
                                        print('  Visually similar image; skipped (review deduplication if distinct models look alike).')
                                        continue
                                    folder = output / brand
                                    folder.mkdir(exist_ok=True)
                                    path = folder / (digest + '.jpg')
                                    rgb.save(path, quality=95)
                                    width, height = rgb.size
                                writer.writerow(dict(zip(FIELDS, [brand, data['title'], construction(text + ' ' + data.get('description', '')),
                                    page.url, image_url, str(path.relative_to(output)), width, height,
                                    str(phash), digest, 'pending_manual_review', 'not_verified'])))
                                f.flush()
                                hashes.append(phash)
                                digests.add(digest)
                                urls.add(image_url)
                                saved += 1
                                state['saved'] += 1
                                print(f'  Total {existing + saved}/{args.target_images}: {path.name[:18]}... ({width}x{height})')
                            except Exception as exc:
                                print(f'  Image skipped: {exc}')
                        print(f'  Product page completed: {saved - before} new images.')
                    except Exception as exc:
                        print(f'  Page skipped: {exc}')
                    finally:
                        await page.close()
                summary[brand].update(pages_attempted=len(seen),
                    product_pages_loaded=state['products'], new_images=state['saved'])
                with (output / 'run_summary.csv').open('w', encoding='utf-8-sig', newline='') as report:
                    report_writer = csv.DictWriter(report, fieldnames=list(summary[brand]))
                    report_writer.writeheader()
                    report_writer.writerows(summary.values())
            await browser.close()
    session.close()
    print(f'Finished: {saved} new images. Review folder: {output}')
    print(f'Total unique images recorded: {existing + saved}/{args.target_images}')
    if existing + saved < args.target_images:
        print('Target not reached: supported queues or per-brand page budgets exhausted. See run_summary.csv and skip messages.')
    print('Review every image and its usage rights before adding it to training.')
    print(f'Access diagnostics: {len(access.blocked)} distinct blocked URLs. No access restrictions were bypassed.')
    print('Per-brand results: run_summary.csv (latest run); images/metadata persist across runs.')
    if not saved:
        print('No images accepted. Check skip messages; use --start-url with a permitted product page.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brand', choices=list(SITES) + ['all'], default='all')
    parser.add_argument('--list-brands', action='store_true', help='List all manufacturers and starting pages')
    parser.add_argument('--output', default='Review_Downloads')
    parser.add_argument('--max-pages', type=int, default=10)
    parser.add_argument('--page-timeout', type=int, default=60, help='Seconds for initial navigation (5..180)')
    parser.add_argument('--content-timeout', type=int, default=30, help='Seconds to wait for useful page content (5..120)')
    parser.add_argument('--target-images', type=int, default=500,
                        help='Total unique images in output folder, including previous runs')
    parser.add_argument('--phash-disctance', type=int, default=0,
                        help='Visual dedup distance 0..64; -1 uses exact file duplicates only')
    parser.add_argument('--min-size', type=int, default=600)
    parser.add_argument('--delay', type=float, default=1.5)
    parser.add_argument('--render-wait', type=float, default=5, help='Seconds for permitted page scripts to render')
    parser.add_argument('--start-url', action='append', help='Specific permitted product URL; repeat for several')
    parser.add_argument('--show-browser', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    print('Tyre extractor v4 - independent host checks and content-based loading')
    if args.list_brands:
        for brand, (name, seeds) in SITES.items():
            print(f'{brand:16} {name}')
            for seed in seeds:
                print(f'  {seed}')
        return
    if args.self_test:
        self_test()
        return
    if args.max_pages < 1 or args.min_size < 1 or args.target_images < 1 or not 0 <= args.render_wait <= 30 or not -1 <= args.phash_distance <= 64:
        parser.error('Page limit, image size and target must be positive; render-wait 0..30; phash-distance -1..64.')
    if args.start_url and args.brand == 'all':
        parser.error('Use a single --brand with --start-url.')
    if not 5 <= args.page_timeout <= 180 or not 5 <= args.content_timeout <= 120:
        parser.error('page-timeout must be 5..180; content-timeout must be 5..120.')
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('Stopped. Saved images and flushed CSV records are preserved.')

if __name__ == '__main__':
    main()
