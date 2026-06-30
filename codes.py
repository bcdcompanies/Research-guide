import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import requests
import time
import xml.etree.ElementTree as ET
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import datetime
import hashlib
import json
import socket
from pathlib import Path

# App Configuration
st.set_page_config(page_title="Research Navigator", layout="wide")

PDF_VIEWER_HEIGHT = 600


def _safe_pdf_url(url):
    """Return url only if it is a non-empty HTTPS link, else None."""
    if url and isinstance(url, str) and url.lower().startswith("https://"):
        return url
    return None


def _retry_get(url, *, params=None, headers=None, timeout=15, max_attempts=4):
    """GET with exponential back-off on 429 / transient network errors."""
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)


def _has_internet(timeout=2):
    """Fast connectivity probe used by offline-first UI controls."""
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
            return True
    except OSError:
        return False


def _local_library_search(papers, query, year_start=None, year_end=None, sort_by='relevance'):
    """Search saved papers locally so the app remains useful without internet."""
    query_terms = [t for t in re.findall(r'\w+', (query or '').lower()) if t]
    cur_year = datetime.date.today().year
    max_year = year_end or cur_year
    out = []

    for p in papers or []:
        title = str(p.get('Title', ''))
        authors = str(p.get('Authors', ''))
        abstract = str(p.get('Abstract', ''))
        year_raw = str(p.get('Year', ''))
        text_blob = f"{title} {authors} {abstract}".lower()

        if query_terms and not all(term in text_blob for term in query_terms):
            continue

        year_num = None
        if year_raw.isdigit():
            year_num = int(year_raw)
            if year_start and year_num < year_start:
                continue
            if year_num > max_year:
                continue

        title_l = title.lower()
        abstract_l = abstract.lower()
        score = 0
        for term in query_terms:
            if term in title_l:
                score += 3
            if term in abstract_l:
                score += 1

        item = {
            'Title': title or 'Untitled',
            'Authors': authors or 'N/A',
            'Abstract': (abstract or '')[:600],
            'URL': p.get('URL') or '#',
            'Year': year_raw or 'n.d.',
            'Venue': p.get('Venue') or 'N/A',
            'Citations': p.get('Citations', 0),
            'Source': p.get('Source') or 'Saved Library',
            'PDF': p.get('PDF') or '',
            '_local_score': score,
        }
        out.append(item)

    if sort_by == 'citations':
        out.sort(key=lambda x: int(x.get('Citations', 0) or 0), reverse=True)
    elif sort_by == 'year':
        out.sort(
            key=lambda x: int(x.get('Year')) if str(x.get('Year', '')).isdigit() else -1,
            reverse=True,
        )
    else:
        out.sort(key=lambda x: x.get('_local_score', 0), reverse=True)

    for row in out:
        row.pop('_local_score', None)
    return out


class ResearchAssistant:
    def __init__(self):
        self.api_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    @st.cache_data(show_spinner=False)
    def search_articles(_self, query, limit=10, year_start=None, year_end=None,
                        sort_by='relevance', api_key=''):
        fields = 'title,authors,abstract,url,venue,year,citationCount,openAccessPdf'
        params = {'query': query, 'limit': limit, 'fields': fields}
        if year_start:
            params['year'] = f"{year_start}-{year_end or ''}"
        if sort_by == 'citations':
            params['sort'] = 'citationCount:desc'
        elif sort_by == 'year':
            params['sort'] = 'year:desc'
        headers = {'x-api-key': api_key} if api_key else {}

        for attempt in range(4):
            try:
                response = requests.get(_self.api_url, params=params,
                                        headers=headers, timeout=15)
                if response.status_code == 429:
                    wait = 2 ** attempt   # 1s → 2s → 4s → 8s
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json().get('data', [])
            except requests.RequestException:
                if attempt == 3:
                    raise   # raise so @st.cache_data does NOT cache the failure
                time.sleep(2 ** attempt)
        return []

    def calculate_sample_size(self, population, confidence=0.95, margin_error=0.05):
        # Cochran's Formula
        if population <= 0:
            raise ValueError("Population must be greater than zero.")
        if margin_error <= 0:
            raise ValueError("Margin of error must be greater than zero.")
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = 0.5
        q = 1 - p
        n_0 = (z**2 * p * q) / (margin_error**2)
        n = n_0 / (1 + (n_0 - 1) / population)
        return int(np.ceil(n))

    @st.cache_data(show_spinner=False)
    def search_pubmed(_self, query, limit=10, year_start=None, year_end=None):
        base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
        cur_year = datetime.date.today().year
        params = {'db': 'pubmed', 'retmode': 'json', 'term': query, 'retmax': limit}
        if year_start:
            params.update({'datetype': 'pdat', 'mindate': str(year_start),
                           'maxdate': str(year_end or cur_year)})
        r = _retry_get(base + 'esearch.fcgi', params=params, timeout=15)
        ids = r.json().get('esearchresult', {}).get('idlist', [])
        if not ids:
            return []
        r2 = _retry_get(base + 'efetch.fcgi',
                        params={'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'},
                        timeout=20)
        root = ET.fromstring(r2.content)
        results = []
        for article in root.findall('.//PubmedArticle'):
            try:
                medline = article.find('MedlineCitation')
                art = medline.find('Article')
                title = (art.findtext('ArticleTitle') or '').rstrip('.')
                author_list = art.find('AuthorList')
                authors_out = []
                if author_list is not None:
                    for a in author_list.findall('Author')[:4]:
                        last = a.findtext('LastName', '')
                        fore = a.findtext('ForeName', '') or a.findtext('Initials', '')
                        if last:
                            authors_out.append(f"{last} {fore}".strip())
                    if len(author_list.findall('Author')) > 4:
                        authors_out.append('et al.')
                abstract_parts = art.findall('.//AbstractText')
                abstract = ' '.join((at.text or '') for at in abstract_parts if at.text)
                journal = (art.findtext('.//Title') or
                           art.findtext('.//ISOAbbreviation') or 'N/A')
                pub_date = medline.find('.//PubDate')
                year = pub_date.findtext('Year', 'n.d.') if pub_date is not None else 'n.d.'
                pmid = medline.findtext('PMID', '')
                results.append({
                    'Title': title, 'Authors': ', '.join(authors_out),
                    'Abstract': abstract[:600], 'URL': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                    'Year': year, 'Venue': journal, 'Citations': 0,
                    'Source': 'PubMed', 'PDF': '',
                })
            except Exception:
                continue
        return results

    @st.cache_data(show_spinner=False)
    def search_europe_pmc(_self, query, limit=10, year_start=None, year_end=None):
        q = query
        cur_year = datetime.date.today().year
        if year_start:
            q += f' AND (FIRST_PDATE:[{year_start}-01-01 TO {year_end or cur_year}-12-31])'
        params = {'query': q, 'format': 'json', 'pageSize': limit,
                  'resultType': 'core', 'sort': 'CITED desc'}
        r = _retry_get('https://www.ebi.ac.uk/europepmc/webservices/rest/search',
                       params=params, timeout=15)
        items = r.json().get('resultList', {}).get('result', [])
        results = []
        for item in items:
            al = (item.get('authorList') or {}).get('author', [])
            authors = [a.get('fullName', '') for a in al[:4]]
            if len(al) > 4:
                authors.append('et al.')
            doi = item.get('doi', '')
            url = (f'https://doi.org/{doi}' if doi else
                   f'https://europepmc.org/article/{item.get("source","")}/{item.get("id","")}')
            pdf_url = ''
            if item.get('inEPMC') == 'Y' and item.get('pmcid'):
                pdf_url = f'https://europepmc.org/articles/{item["pmcid"]}/pdf'
            results.append({
                'Title': (item.get('title') or '').rstrip('.'),
                'Authors': ', '.join(authors),
                'Abstract': (item.get('abstractText') or '')[:600],
                'URL': url, 'Year': str(item.get('pubYear', 'n.d.')),
                'Venue': item.get('journalTitle', 'N/A'),
                'Citations': item.get('citedByCount', 0),
                'Source': 'Europe PMC', 'PDF': pdf_url,
            })
        return results

    @st.cache_data(show_spinner=False)
    def search_openalex(_self, query, limit=10, year_start=None, year_end=None):
        def reconstruct(inv):
            if not inv:
                return ''
            pw = {}
            for w, positions in inv.items():
                for p in positions:
                    pw[p] = w
            return ' '.join(pw[i] for i in sorted(pw))

        params = {
            'search': query, 'per-page': limit,
            'select': 'title,authorships,abstract_inverted_index,primary_location,publication_year,cited_by_count,doi,open_access',
            'mailto': 'research@researchnavigator.app',
        }
        if year_start:
            params['filter'] = f'publication_year:{year_start}-{year_end or datetime.date.today().year}'
        r = _retry_get('https://api.openalex.org/works', params=params, timeout=15)
        results = []
        for item in r.json().get('results', []):
            auths = item.get('authorships', [])
            authors = [a.get('author', {}).get('display_name', '') for a in auths[:4]]
            if len(auths) > 4:
                authors.append('et al.')
            doi = item.get('doi') or ''
            src = ((item.get('primary_location') or {}).get('source') or {})
            venue = src.get('display_name', 'N/A')
            oa = item.get('open_access') or {}
            results.append({
                'Title': item.get('title') or '',
                'Authors': ', '.join(authors),
                'Abstract': reconstruct(item.get('abstract_inverted_index'))[:600],
                'URL': doi or '#', 'Year': str(item.get('publication_year', 'n.d.')),
                'Venue': venue, 'Citations': item.get('cited_by_count', 0),
                'Source': 'OpenAlex', 'PDF': oa.get('oa_url') or '',
            })
        return results

    @st.cache_data(show_spinner=False)
    def search_arxiv(_self, query, limit=10, year_start=None, year_end=None):
        params = {'search_query': f'all:{query}', 'start': 0,
                  'max_results': limit, 'sortBy': 'relevance', 'sortOrder': 'descending'}
        r = _retry_get('https://export.arxiv.org/api/query', params=params, timeout=15)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(r.content)
        results = []
        for entry in root.findall('atom:entry', ns):
            title = (entry.findtext('atom:title', '', ns) or '').strip().replace('\n', ' ')
            abstract = (entry.findtext('atom:summary', '', ns) or '').strip().replace('\n', ' ')[:600]
            all_authors = entry.findall('atom:author', ns)
            authors = [a.findtext('atom:name', '', ns) for a in all_authors[:4]]
            if len(all_authors) > 4:
                authors.append('et al.')
            published = entry.findtext('atom:published', '', ns)
            year = published[:4] if published else 'n.d.'
            if year_start and year.isdigit():
                if int(year) < year_start or int(year) > (year_end or datetime.date.today().year):
                    continue
            url, pdf_url = '#', ''
            for link in entry.findall('atom:link', ns):
                if link.get('rel') == 'alternate':
                    url = link.get('href', '#')
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href', '')
            cat = entry.find('atom:category', ns)
            venue = f'arXiv [{cat.get("term", "")}]' if cat is not None else 'arXiv'
            results.append({
                'Title': title, 'Authors': ', '.join(authors),
                'Abstract': abstract, 'URL': url, 'Year': year,
                'Venue': venue, 'Citations': 0, 'Source': 'arXiv', 'PDF': pdf_url,
            })
        return results

    @st.cache_data(show_spinner=False)
    def search_crossref(_self, query, limit=10, year_start=None, year_end=None):
        params = {
            'query': query, 'rows': limit,
            'select': 'title,author,abstract,URL,published,container-title,is-referenced-by-count,link',
        }
        if year_start:
            params['filter'] = f'from-pub-date:{year_start},until-pub-date:{year_end or datetime.date.today().year}'
        headers = {'User-Agent': 'ResearchNavigator/1.0 (mailto:research@researchnavigator.app)'}
        r = _retry_get('https://api.crossref.org/works', params=params,
                       headers=headers, timeout=15)
        results = []
        for item in r.json().get('message', {}).get('items', []):
            titles = item.get('title') or []
            title = titles[0] if titles else 'Untitled'
            authors_raw = item.get('author') or []
            authors = [f"{a.get('family','')} {(a.get('given','') or '')[:1]}".strip()
                       for a in authors_raw[:4]]
            if len(authors_raw) > 4:
                authors.append('et al.')
            abstract = re.sub(r'<[^>]+>', '', item.get('abstract') or '')[:600]
            pub = item.get('published') or {}
            parts = (pub.get('date-parts') or [[]])[0]
            year = str(parts[0]) if parts else 'n.d.'
            venue = (item.get('container-title') or ['N/A'])[0]
            url = item.get('URL', '#')
            pdf_url = ''
            for link in (item.get('link') or []):
                if link.get('content-type') == 'application/pdf':
                    pdf_url = link.get('URL', '')
                    break
            results.append({
                'Title': title, 'Authors': ', '.join(authors),
                'Abstract': abstract, 'URL': url, 'Year': year,
                'Venue': venue, 'Citations': item.get('is-referenced-by-count', 0),
                'Source': 'CrossRef', 'PDF': pdf_url,
            })
        return results

    @st.cache_data(show_spinner=False)
    def search_doaj(_self, query, limit=10, year_start=None, year_end=None):
        url = f'https://doaj.org/api/v3/search/articles/{urllib.parse.quote(query)}'
        r = _retry_get(url, params={'pageSize': limit, 'page': 1}, timeout=15)
        results = []
        for item in r.json().get('results', []):
            bib = item.get('bibjson', {})
            title = bib.get('title', 'Untitled')
            all_authors = bib.get('author', [])
            authors = [a.get('name', '') for a in all_authors[:4]]
            if len(all_authors) > 4:
                authors.append('et al.')
            year_str = str(bib.get('year', 'n.d.'))
            if year_start and year_str.isdigit():
                if int(year_str) < year_start or int(year_str) > (year_end or datetime.date.today().year):
                    continue
            abstract = (bib.get('abstract') or '')[:600]
            venue = (bib.get('journal') or {}).get('title', 'N/A')
            url_link, pdf_url = '#', ''
            for ident in bib.get('identifier', []):
                if ident.get('type') == 'doi':
                    url_link = f'https://doi.org/{ident["id"]}'
            for link in bib.get('link', []):
                if link.get('type') in ('fulltext', 'pdf'):
                    pdf_url = link.get('url', '')
                    break
            results.append({
                'Title': title, 'Authors': ', '.join(authors),
                'Abstract': abstract, 'URL': url_link, 'Year': year_str,
                'Venue': venue, 'Citations': 0, 'Source': 'DOAJ', 'PDF': pdf_url,
            })
        return results

    @st.cache_data(show_spinner=False)
    def search_urology_journals(_self, query, limit=10, year_start=None, year_end=None):
        base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
        cur_year = datetime.date.today().year
        journals = [
            'Journal of Urology',
            'BJU International',
            'European Urology',
            'Urology',
            'Urologic Oncology',
            'World Journal of Urology',
            'Neurourology and Urodynamics',
            'International Urology and Nephrology',
            'Investigative and Clinical Urology',
            'Canadian Urological Association Journal',
            'Urology Annals',
            'African Journal of Urology',
            'Current Urology',
        ]
        journal_filter = ' OR '.join(f'"{j}"[Journal]' for j in journals)
        term = f'({query}) AND ({journal_filter})'
        params = {'db': 'pubmed', 'retmode': 'json', 'term': term, 'retmax': limit}
        if year_start:
            params.update({'datetype': 'pdat', 'mindate': str(year_start),
                           'maxdate': str(year_end or cur_year)})
        r = _retry_get(base + 'esearch.fcgi', params=params, timeout=15)
        ids = r.json().get('esearchresult', {}).get('idlist', [])
        if not ids:
            return []
        r2 = _retry_get(base + 'efetch.fcgi',
                        params={'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'},
                        timeout=20)
        root = ET.fromstring(r2.content)
        results = []
        for article in root.findall('.//PubmedArticle'):
            try:
                medline = article.find('MedlineCitation')
                art = medline.find('Article')
                title = (art.findtext('ArticleTitle') or '').rstrip('.')
                author_list = art.find('AuthorList')
                authors_out = []
                if author_list is not None:
                    for a in author_list.findall('Author')[:4]:
                        last = a.findtext('LastName', '')
                        fore = a.findtext('ForeName', '') or a.findtext('Initials', '')
                        if last:
                            authors_out.append(f"{last} {fore}".strip())
                    if len(author_list.findall('Author')) > 4:
                        authors_out.append('et al.')
                abstract_parts = art.findall('.//AbstractText')
                abstract = ' '.join((at.text or '') for at in abstract_parts if at.text)
                journal = (art.findtext('.//Title') or
                           art.findtext('.//ISOAbbreviation') or 'N/A')
                pub_date = medline.find('.//PubDate')
                year = pub_date.findtext('Year', 'n.d.') if pub_date is not None else 'n.d.'
                pmid = medline.findtext('PMID', '')
                results.append({
                    'Title': title, 'Authors': ', '.join(authors_out),
                    'Abstract': abstract[:800], 'URL': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                    'Year': year, 'Venue': journal, 'Citations': 0,
                    'Source': 'Urology Journals', 'PDF': '',
                })
            except Exception:
                continue
        return results

# ── Citation formatting helpers ──────────────────────────────────────────────
def format_reference(paper, style='APA 7th', number=None):
    """Return a formatted bibliographic reference string."""
    authors = paper.get('Authors', 'Unknown Author')
    year    = paper.get('Year', 'n.d.')
    title   = paper.get('Title', 'Untitled')
    venue   = paper.get('Venue', '') or ''
    url     = paper.get('URL', '') or ''
    venue_clean = venue if venue not in ('N/A', '') else ''

    if style == 'APA 7th':
        ref = f"{authors} ({year}). {title}."
        if venue_clean:
            ref += f" *{venue_clean}*."
        if url and url != '#':
            ref += f" {url}"
    elif style == 'Vancouver':
        n = f"{number}. " if number else ""
        ref = f"{n}{authors}. {title}."
        if venue_clean:
            ref += f" {venue_clean}."
        ref += f" {year}."
        if url and url != '#':
            ref += f" Available from: {url}"
    elif style == 'Harvard':
        ref = f"{authors} {year}, '{title}'"
        if venue_clean:
            ref += f", *{venue_clean}*"
        if url and url != '#':
            ref += f", viewed at {url}"
        ref += "."
    else:
        ref = f"{authors} ({year}). {title}."
    return ref


def in_text_cite(paper, style='APA 7th', number=None):
    """Return a formatted in-text citation string."""
    first_author = ((paper.get('Authors') or '').split(',')[0].strip()
                    or 'Unknown Author')
    year = paper.get('Year', 'n.d.')
    if style == 'Vancouver':
        return f"[{number}]" if number else "[?]"
    return f"({first_author}, {year})"


def evidence_badge(citations):
    """Return an evidence quality label based on citation count."""
    if citations is None:
        return "Unknown"
    try:
        c = int(citations)
    except (ValueError, TypeError):
        return "Unknown"
    if c >= 100:
        return "High (≥100 citations)"
    if c >= 20:
        return "Moderate (20–99 citations)"
    return "Emerging (<20 citations)"


def _paper_library_id(paper):
    """Create a stable library identifier for a paper record."""
    raw = "|".join([
        (paper.get('Title') or '').strip().lower(),
        str(paper.get('Year', '')).strip().lower(),
        (paper.get('Source') or '').strip().lower(),
        (paper.get('URL') or '').strip().lower(),
    ])
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


_LIBRARY_FILE = Path(__file__).resolve().with_name("saved_papers.json")


def _load_saved_papers():
    """Load the saved library from disk."""
    try:
        if not _LIBRARY_FILE.exists():
            return []
        with _LIBRARY_FILE.open("r", encoding="utf-8") as handle:
            papers = json.load(handle)
        if not isinstance(papers, list):
            return []
        cleaned = []
        for paper in papers:
            if isinstance(paper, dict):
                paper = dict(paper)
                paper.setdefault('Paper ID', _paper_library_id(paper))
                cleaned.append(paper)
        return cleaned
    except Exception:
        return []


def _save_saved_papers(papers):
    """Persist the saved library to disk."""
    try:
        with _LIBRARY_FILE.open("w", encoding="utf-8") as handle:
            json.dump(papers, handle, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception:
        return False


def _library_session():
    """Return library list from session, initializing if needed."""
    if 'saved_papers' not in st.session_state:
        st.session_state.saved_papers = _load_saved_papers()
    return st.session_state.saved_papers


def _add_to_library(entry):
    """Add one entry to library and persist it."""
    lib = _library_session()
    entry = dict(entry)
    entry.setdefault('Paper ID', _paper_library_id(entry))
    if any(p.get('Paper ID') == entry['Paper ID'] for p in lib):
        return False, "exists"
    lib.append(entry)
    if _save_saved_papers(lib):
        return True, "saved"
    return True, "session_only"


def _reload_library_from_disk():
    """Reload saved library from disk into session and return before/after counts."""
    before = len(st.session_state.get('saved_papers', []))
    latest = _load_saved_papers()
    st.session_state.saved_papers = latest
    after = len(latest)
    if before == after:
        st.session_state.library_reload_notice = (
            f"Library reload complete. No change detected ({after} paper(s))."
        )
    else:
        st.session_state.library_reload_notice = (
            f"Library reloaded from disk: {before} → {after} paper(s)."
        )
    return before, after


# Initialize App
assistant = ResearchAssistant()

if 'library_reload_notice' not in st.session_state:
    st.session_state.library_reload_notice = ''

# Load persistent library for all modules on app startup.
if 'saved_papers' not in st.session_state:
    st.session_state.saved_papers = _load_saved_papers()

_disk_now = _load_saved_papers()
if len(_disk_now) > len(st.session_state.saved_papers):
    st.session_state.saved_papers = _disk_now

st.title("Research Assistant App")
st.sidebar.title("Navigation")
module = st.sidebar.radio("Go to:", [
    "Conceptualization",
    "Literature Search",
    "Literature Review",
    "Methodology & Stats",
    "Data Viz & Tables",
    "Manuscript Drafter"
])

st.sidebar.divider()
st.sidebar.subheader("API Settings")
api_key = st.sidebar.text_input(
    "Semantic Scholar API Key (optional)",
    type="password",
    help="Free key from semanticscholar.org/product/api raises rate limits. Leave blank for anonymous access."
)
if st.sidebar.button("Clear Search Cache"):
    assistant.search_articles.clear()
    assistant.search_pubmed.clear()
    assistant.search_europe_pmc.clear()
    assistant.search_openalex.clear()
    assistant.search_arxiv.clear()
    assistant.search_crossref.clear()
    assistant.search_doaj.clear()
    st.sidebar.success("Cache cleared.")

st.sidebar.divider()
st.sidebar.subheader("Citation Settings")
if 'citation_style' not in st.session_state:
    st.session_state.citation_style = 'APA 7th'
st.session_state.citation_style = st.sidebar.selectbox(
    "Reference Style",
    ["APA 7th", "Vancouver", "Harvard"],
    index=["APA 7th", "Vancouver", "Harvard"].index(st.session_state.citation_style),
    help="Applied to in-text citations and the reference list in the manuscript."
)

st.sidebar.divider()
st.sidebar.subheader("Offline-First")
if 'offline_mode' not in st.session_state:
    st.session_state.offline_mode = False
st.session_state.offline_mode = st.sidebar.toggle(
    "Offline Mode",
    value=st.session_state.offline_mode,
    help=(
        "Disables all web API calls. Literature Search will use your local saved library only, "
        "while all non-search modules continue to work offline."
    ),
)
_online_now = _has_internet(timeout=1)
if st.session_state.offline_mode:
    st.sidebar.warning("Offline mode is ON")
elif _online_now:
    st.sidebar.success("Network detected")
else:
    st.sidebar.warning("No network detected")

# --- Module 1: Conceptualization ---
if module == "Conceptualization":
    st.header("Research Question Conceptualization")
    st.caption("Work through 6 iterative stages — each stage deepens and refines the one before it.")

    # ── session state keys ───────────────────────────────────────────────────
    for k in ['cq_stage','cq_topic','cq_domain','cq_scope','cq_affected',
              'cq_severity','cq_known','cq_unknown','cq_iv','cq_dv',
              'cq_moderators','cq_mediators','cq_gap_theory','cq_gap_empirical',
              'cq_gap_practice','cq_q_selected','cq_paradigm','cq_theory',
              'cq_theory_link','cq_significance_who','cq_significance_how',
              'cq_significance_why','cq_design','cq_population','cq_comparison',
              'cq_timeframe']:
        if k not in st.session_state:
            st.session_state[k] = '' if k != 'cq_stage' else 1

    stage = st.session_state.cq_stage

    # ── stage progress bar ───────────────────────────────────────────────────
    stage_labels = [
        "1 · Topic & Domain",
        "2 · Problem Deconstruction",
        "3 · Variable Mapping",
        "4 · Gap Triangulation",
        "5 · Question Iteration",
        "6 · Framework & Full Narrative"
    ]
    prog_cols = st.columns(6)
    for i, (col, label) in enumerate(zip(prog_cols, stage_labels), start=1):
        if i < stage:
            col.success(label)
        elif i == stage:
            col.info(f"**{label}**")
        else:
            col.markdown(f"<div style='color:#aaa;font-size:0.8em'>{label}</div>", unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Topic & Domain
    # ════════════════════════════════════════════════════════════════════════
    if stage == 1:
        st.subheader("Stage 1 — Topic & Domain Scoping")
        st.markdown(
            "Begin with your **broadest idea**. We will narrow it progressively. "
            "Good research starts from a wide lens before zooming in."
        )
        s1a, s1b = st.columns(2)
        with s1a:
            st.session_state.cq_topic = st.text_area(
                "What is your broad research topic?",
                value=st.session_state.cq_topic,
                height=90,
                placeholder="e.g., Management of benign prostatic hyperplasia in elderly men"
            )
            st.session_state.cq_domain = st.selectbox(
                "Research Domain",
                ["Clinical / Medical", "Nursing / Allied Health", "Public Health / Epidemiology",
                 "Social / Behavioural Sciences", "Education", "Psychology",
                 "Environmental / Agricultural", "Engineering / Technology", "Economics / Policy", "Other"],
                index=["Clinical / Medical", "Nursing / Allied Health", "Public Health / Epidemiology",
                       "Social / Behavioural Sciences", "Education", "Psychology",
                       "Environmental / Agricultural", "Engineering / Technology",
                       "Economics / Policy", "Other"].index(st.session_state.cq_domain)
                if st.session_state.cq_domain else 0
            )
        with s1b:
            st.session_state.cq_scope = st.text_input(
                "Geographic / Institutional Scope",
                value=st.session_state.cq_scope,
                placeholder="e.g., tertiary hospitals in Ghana"
            )
            st.markdown("**Guiding Questions for this stage:**")
            st.markdown(
                "- Is this topic researchable (observable, measurable)?\n"
                "- Is there existing literature to build on?\n"
                "- Do you have access to the study population?\n"
                "- Is the scope feasible within your resources?"
            )

        if st.button("Proceed to Stage 2 →"):
            if not st.session_state.cq_topic or not st.session_state.cq_scope:
                st.warning("Please complete the topic and scope before proceeding.")
            else:
                st.session_state.cq_stage = 2
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Problem Deconstruction
    # ════════════════════════════════════════════════════════════════════════
    elif stage == 2:
        st.subheader("Stage 2 — Problem Deconstruction")
        st.markdown(
            f"Topic: *{st.session_state.cq_topic}*  \n"
            "Now **break down the problem** into its observable, affected, and unexplored components. "
            "This stage transforms a topic into a researchable problem."
        )
        p1, p2 = st.columns(2)
        with p1:
            st.session_state.cq_affected = st.text_area(
                "Who is affected and how?",
                value=st.session_state.cq_affected, height=80,
                placeholder="e.g., Elderly men (>60 yrs) in sub-Saharan Africa experience delayed diagnosis...")
            st.session_state.cq_severity = st.text_area(
                "What is the severity / burden of the problem?",
                value=st.session_state.cq_severity, height=80,
                placeholder="e.g., BPH affects ~40% of men >50, with significant impact on QoL and healthcare costs...")
        with p2:
            st.session_state.cq_known = st.text_area(
                "What is already known about this problem?",
                value=st.session_state.cq_known, height=80,
                placeholder="e.g., Alpha-blockers are well-established first-line agents; 5-ARIs reduce prostate volume...")
            st.session_state.cq_unknown = st.text_area(
                "What remains unknown or underexplored? (the gap)",
                value=st.session_state.cq_unknown, height=80,
                placeholder="e.g., Comparative effectiveness in low-resource settings with limited diagnostic tools is poorly studied...")

        with st.expander("Problem Deconstruction Tips"):
            st.markdown("""
| Element | Question to ask | What to write |
|---|---|---|
| **Who** | Who bears the burden? | Specific population + demographics |
| **What** | What exactly is the problem? | Measurable phenomenon |
| **Where** | In what context? | Setting, system, geography |
| **How much** | What is the scale/severity? | Statistics, rates, costs |
| **Why it matters** | So what? | Consequences of not addressing it |
| **Gap** | What is missing in knowledge/practice? | The specific deficit |
""")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("← Back to Stage 1"):
                st.session_state.cq_stage = 1; st.rerun()
        with bc2:
            if st.button("Proceed to Stage 3 →"):
                if not st.session_state.cq_affected or not st.session_state.cq_unknown:
                    st.warning("Please complete all fields before proceeding.")
                else:
                    st.session_state.cq_stage = 3; st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Variable Mapping
    # ════════════════════════════════════════════════════════════════════════
    elif stage == 3:
        st.subheader("Stage 3 — Variable Mapping")
        st.markdown(
            "Identify the **key variables** and their conceptual relationships. "
            "This is the architectural blueprint of your study."
        )
        v1, v2 = st.columns(2)
        with v1:
            st.session_state.cq_iv = st.text_input(
                "Independent Variable(s) / Exposure / Predictor",
                value=st.session_state.cq_iv,
                placeholder="e.g., type of medical therapy (alpha-blocker vs. 5-ARI vs. combination)")
            st.session_state.cq_dv = st.text_input(
                "Dependent Variable(s) / Outcome",
                value=st.session_state.cq_dv,
                placeholder="e.g., IPSS score, urinary flow rate, adverse events at 12 months")
            st.session_state.cq_population = st.text_input(
                "Target Population (PICO: P)",
                value=st.session_state.cq_population,
                placeholder="e.g., men aged ≥50 diagnosed with BPH")
        with v2:
            st.session_state.cq_moderators = st.text_input(
                "Moderating Variables (change the strength of IV→DV)",
                value=st.session_state.cq_moderators,
                placeholder="e.g., age group, comorbidities, disease severity")
            st.session_state.cq_mediators = st.text_input(
                "Mediating / Confounding Variables",
                value=st.session_state.cq_mediators,
                placeholder="e.g., baseline IPSS, prostate volume, adherence to therapy")
            st.session_state.cq_comparison = st.text_input(
                "Comparison / Control (PICO: C) — optional",
                value=st.session_state.cq_comparison,
                placeholder="e.g., watchful waiting / placebo")
            st.session_state.cq_timeframe = st.text_input(
                "Timeframe (PICO: T) — optional",
                value=st.session_state.cq_timeframe,
                placeholder="e.g., over 12 months")

        if st.session_state.cq_iv and st.session_state.cq_dv:
            st.divider()
            st.markdown("**Conceptual Relationship Map**")
            mods = f"  ← moderated by: *{st.session_state.cq_moderators}*" if st.session_state.cq_moderators else ""
            meds = f"\n*mediated/confounded by: {st.session_state.cq_mediators}*" if st.session_state.cq_mediators else ""
            st.code(
                f"[ {st.session_state.cq_iv} ]\n"
                f"        │{mods}\n"
                f"        ▼{meds}\n"
                f"[ {st.session_state.cq_dv} ]",
                language=None
            )

        with st.expander("Variable Type Guide"):
            st.markdown("""
| Variable Role | Definition | Example |
|---|---|---|
| **Independent (IV)** | The presumed cause / exposure | Type of medication |
| **Dependent (DV)** | The measured effect / outcome | Symptom severity score |
| **Moderator** | Changes IV→DV relationship strength | Age, sex, disease stage |
| **Mediator** | Explains *how* IV affects DV | Hormonal pathway, adherence |
| **Confounder** | Distorts IV→DV relationship | Pre-existing comorbidity |
| **Control** | Held constant to isolate IV effect | Baseline characteristics |
""")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("← Back to Stage 2"):
                st.session_state.cq_stage = 2; st.rerun()
        with bc2:
            if st.button("Proceed to Stage 4 →"):
                if not st.session_state.cq_iv or not st.session_state.cq_dv:
                    st.warning("IV and DV are required.")
                else:
                    st.session_state.cq_stage = 4; st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Gap Triangulation
    # ════════════════════════════════════════════════════════════════════════
    elif stage == 4:
        st.subheader("Stage 4 — Three-Angle Gap Triangulation")
        st.markdown(
            "A robust research gap is not just 'no one studied this'. "
            "Triangulate the gap from **three angles** — theoretical, empirical, and practical — "
            "to build an airtight justification."
        )

        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("##### 🔵 Theoretical Gap")
            st.caption("What does existing theory fail to explain or predict about your phenomenon?")
            st.session_state.cq_gap_theory = st.text_area(
                "Theoretical gap",
                value=st.session_state.cq_gap_theory, height=130,
                label_visibility='collapsed',
                placeholder="e.g., Current pharmacodynamic models do not account for patient adherence variability in resource-limited settings...")
        with g2:
            st.markdown("##### 🟡 Empirical Gap")
            st.caption("What has not been studied, replicated, or measured in the literature?")
            st.session_state.cq_gap_empirical = st.text_area(
                "Empirical gap",
                value=st.session_state.cq_gap_empirical, height=130,
                label_visibility='collapsed',
                placeholder="e.g., No RCT or comparative study has examined alpha-blocker vs. 5-ARI outcomes specifically in African populations...")
        with g3:
            st.markdown("##### 🔴 Practice Gap")
            st.caption("What disconnect exists between what evidence recommends and what practitioners actually do?")
            st.session_state.cq_gap_practice = st.text_area(
                "Practice gap",
                value=st.session_state.cq_gap_practice, height=130,
                label_visibility='collapsed',
                placeholder="e.g., Clinical guidelines recommend combination therapy but most facilities in Ghana use monotherapy due to cost constraints...")

        if (st.session_state.cq_gap_theory and
                st.session_state.cq_gap_empirical and
                st.session_state.cq_gap_practice):
            st.divider()
            st.markdown("**Synthesised Gap Statement** *(auto-generated — refine as needed)*")
            gap_synth = (
                f"Despite theoretical frameworks that attempt to explain {st.session_state.cq_iv}, "
                f"{st.session_state.cq_gap_theory.strip().rstrip('.')}. "
                f"From an empirical standpoint, {st.session_state.cq_gap_empirical.strip().rstrip('.')}. "
                f"Furthermore, in practice, {st.session_state.cq_gap_practice.strip().rstrip('.')}. "
                f"This convergence of theoretical, empirical, and practical deficits in the context of "
                f"{st.session_state.cq_population or st.session_state.cq_affected} "
                f"constitutes a compelling rationale for the present study."
            )
            st.info(gap_synth)

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("← Back to Stage 3"):
                st.session_state.cq_stage = 3; st.rerun()
        with bc2:
            if st.button("Proceed to Stage 5 →"):
                if not (st.session_state.cq_gap_theory and
                        st.session_state.cq_gap_empirical and
                        st.session_state.cq_gap_practice):
                    st.warning("Please complete all three gap angles.")
                else:
                    st.session_state.cq_stage = 5; st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Iterative Question Refinement
    # ════════════════════════════════════════════════════════════════════════
    elif stage == 5:
        st.subheader("Stage 5 — Iterative Research Question Refinement")
        st.markdown(
            "Three progressively sharpened versions of your research question are generated below — "
            "from broad to highly specific. **Select the one that best fits** your study, "
            "then refine it further if needed."
        )

        iv   = st.session_state.cq_iv
        dv   = st.session_state.cq_dv
        pop  = st.session_state.cq_population or st.session_state.cq_affected
        comp = st.session_state.cq_comparison
        time = st.session_state.cq_timeframe
        tc   = f" within {time}" if time else ""
        cc   = f" compared to {comp}" if comp else ""

        q_v1 = f"What is the effect of {iv} on {dv} among {pop}?"
        q_v2 = (f"To what extent does {iv} affect {dv} among {pop}{cc}{tc}?")
        q_v3 = (
            f"Among {pop} in {st.session_state.cq_scope}, "
            f"what is the comparative effect of {iv}{cc} on {dv}{tc}, "
            f"and what factors moderate this relationship?"
        )

        design_options = [
            "Descriptive", "Correlational", "Comparative", "Experimental / RCT",
            "Qualitative (Phenomenological)", "Qualitative (Grounded Theory)",
            "Mixed Methods (Explanatory Sequential)",
            "Mixed Methods (Exploratory Sequential)",
            "Case Study", "Systematic Review / Meta-Analysis"
        ]

        st.session_state.cq_design = st.selectbox(
            "Confirm Study Design",
            design_options,
            index=design_options.index(st.session_state.cq_design)
            if st.session_state.cq_design in design_options else 0
        )

        design_q_map = {
            "Descriptive":                            f"What is the prevalence/pattern of {dv} among {pop} in {st.session_state.cq_scope}{tc}?",
            "Correlational":                          f"What is the relationship between {iv} and {dv} among {pop}{tc}?",
            "Comparative":                            f"How does {dv} differ between {iv}{cc} among {pop}{tc}?",
            "Experimental / RCT":                     q_v2,
            "Qualitative (Phenomenological)":         f"How do {pop} experience or perceive {iv} and its impact on {dv}?",
            "Qualitative (Grounded Theory)":          f"How does the process of {iv} unfold among {pop} in {st.session_state.cq_scope}?",
            "Mixed Methods (Explanatory Sequential)": f"What is the effect of {iv} on {dv} among {pop}{tc}, and how do participants explain these outcomes?",
            "Mixed Methods (Exploratory Sequential)": f"What factors shape {dv} among {pop}, and how can these inform an intervention involving {iv}?",
            "Case Study":                             f"How and why does {iv} influence {dv} within the specific context of {st.session_state.cq_scope}?",
            "Systematic Review / Meta-Analysis":      f"What does the existing evidence reveal about the effect of {iv} on {dv} among {pop}?"
        }

        st.divider()
        st.markdown("#### Three Iterations")
        iter_cols = st.columns(3)
        with iter_cols[0]:
            st.markdown("**Version 1 — Broad**")
            st.info(q_v1)
            st.caption("Simple, accessible. Good starting point.")
        with iter_cols[1]:
            st.markdown("**Version 2 — Focused**")
            st.info(q_v2)
            st.caption("Adds comparison + timeframe. Most common form.")
        with iter_cols[2]:
            st.markdown("**Version 3 — Precise**")
            st.info(q_v3)
            st.caption("Context-specific, includes moderators. Suitable for advanced studies.")

        st.markdown(f"**Version 4 — Design-Aligned ({st.session_state.cq_design})**")
        st.success(design_q_map.get(st.session_state.cq_design, q_v2))

        st.divider()
        st.markdown("#### Your Final Research Question")
        st.caption("Select a version above as your base, then refine it here:")
        st.session_state.cq_q_selected = st.text_area(
            "Final Research Question",
            value=st.session_state.cq_q_selected or design_q_map.get(st.session_state.cq_design, q_v2),
            height=90
        )

        st.markdown("#### Sub-Questions / Specific Objectives")
        sub_q = [
            f"SQ1: What is the sociodemographic profile of {pop}?",
            f"SQ2: What is the prevalence/level of {dv} among {pop}?",
            f"SQ3: What is the association between {iv} and {dv} among {pop}{tc}?",
        ]
        if comp:
            sub_q.append(f"SQ4: How does {dv} compare between {iv} and {comp} groups?")
        if st.session_state.cq_moderators:
            sub_q.append(f"SQ{len(sub_q)+1}: Do {st.session_state.cq_moderators} moderate the {iv}→{dv} relationship?")
        for sq in sub_q:
            st.markdown(f"- {sq}")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("← Back to Stage 4"):
                st.session_state.cq_stage = 4; st.rerun()
        with bc2:
            if st.button("Proceed to Stage 6 →"):
                if not st.session_state.cq_q_selected.strip():
                    st.warning("Please confirm your research question.")
                else:
                    st.session_state.cq_stage = 6; st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 6 — Theoretical Framework & Full Narrative
    # ════════════════════════════════════════════════════════════════════════
    elif stage == 6:
        st.subheader("Stage 6 — Theoretical Framework & Full Narrative")

        f1, f2 = st.columns(2)
        with f1:
            st.session_state.cq_paradigm = st.selectbox(
                "Research Paradigm",
                ["Positivism / Post-Positivism",
                 "Interpretivism / Constructivism",
                 "Pragmatism",
                 "Critical Theory / Transformative"],
                index=["Positivism / Post-Positivism",
                       "Interpretivism / Constructivism",
                       "Pragmatism",
                       "Critical Theory / Transformative"].index(
                    st.session_state.cq_paradigm) if st.session_state.cq_paradigm else 0
            )
            paradigm_desc = {
                "Positivism / Post-Positivism":   "Objective reality; quantitative measurement; hypothesis testing.",
                "Interpretivism / Constructivism": "Subjective meaning; qualitative understanding; lived experience.",
                "Pragmatism":                      "What works; mixed methods; problem-centred.",
                "Critical Theory / Transformative":"Power structures; emancipation; participatory action."
            }
            st.caption(paradigm_desc[st.session_state.cq_paradigm])

        with f2:
            theory_suggestions = {
                "Clinical / Medical":              ["Biomedical Model", "Health Belief Model", "Biopsychosocial Model", "Clinical Decision Theory"],
                "Nursing / Allied Health":         ["Orem's Self-Care Theory", "Roy's Adaptation Model", "Leininger's Cultural Care Theory", "Peplau's Interpersonal Relations"],
                "Public Health / Epidemiology":    ["Social Determinants of Health", "Health Belief Model", "Ecological Model", "Theory of Planned Behaviour"],
                "Social / Behavioural Sciences":   ["Social Cognitive Theory", "Theory of Planned Behaviour", "Symbolic Interactionism", "Social Learning Theory"],
                "Education":                       ["Constructivism (Vygotsky)", "Bloom's Taxonomy", "Experiential Learning Theory", "Self-Determination Theory"],
                "Psychology":                      ["Cognitive Behavioural Theory", "Maslow's Hierarchy", "Self-Determination Theory", "Attachment Theory"],
                "Environmental / Agricultural":    ["Ecosystem Services Framework", "Sustainable Livelihoods", "Diffusion of Innovations", "Systems Theory"],
                "Engineering / Technology":        ["Systems Theory", "Technology Acceptance Model", "Diffusion of Innovations", "Human Factors Theory"],
                "Economics / Policy":              ["Rational Choice Theory", "Public Choice Theory", "Institutional Theory", "Systems Theory"],
                "Other":                           ["Systems Theory", "Complexity Theory", "Ecological Model", "Grounded Theory"]
            }
            domain = st.session_state.cq_domain or "Other"
            available_theories = theory_suggestions.get(domain, theory_suggestions["Other"])
            st.session_state.cq_theory = st.selectbox(
                "Underpinning Theory / Framework",
                available_theories + ["Other (specify below)"],
                index=available_theories.index(st.session_state.cq_theory)
                if st.session_state.cq_theory in available_theories else 0
            )
            st.session_state.cq_theory_link = st.text_area(
                "How does this theory underpin your study?",
                value=st.session_state.cq_theory_link, height=75,
                placeholder="e.g., The Health Belief Model posits that individuals' perceptions of susceptibility and severity drive health-seeking behaviour. This study uses it to explain...")

        st.divider()
        st.markdown("#### Study Significance")
        sig1, sig2, sig3 = st.columns(3)
        with sig1:
            st.session_state.cq_significance_who = st.text_input(
                "Who benefits from this study?",
                value=st.session_state.cq_significance_who,
                placeholder="e.g., clinicians, policymakers, patients with BPH")
        with sig2:
            st.session_state.cq_significance_how = st.text_input(
                "How will they benefit?",
                value=st.session_state.cq_significance_how,
                placeholder="e.g., evidence-based treatment selection, cost reduction")
        with sig3:
            st.session_state.cq_significance_why = st.text_input(
                "Why is now the right time?",
                value=st.session_state.cq_significance_why,
                placeholder="e.g., rising BPH burden + new generic drug availability")

        # ── Generate Full Narrative ─────────────────────────────────────────
        st.divider()
        if st.button("Generate Full Conceptualisation Narrative", type="primary"):
            iv   = st.session_state.cq_iv
            dv   = st.session_state.cq_dv
            pop  = st.session_state.cq_population or st.session_state.cq_affected
            tc   = f" within {st.session_state.cq_timeframe}" if st.session_state.cq_timeframe else ""
            cc   = f" compared to {st.session_state.cq_comparison}" if st.session_state.cq_comparison else ""

            # --- Paragraph 1: Background & Burden ---
            p1 = (
                f"{st.session_state.cq_topic} represents a significant concern within the field of "
                f"{st.session_state.cq_domain}. {st.session_state.cq_severity.strip()} "
                f"This phenomenon is particularly pronounced among {pop} in {st.session_state.cq_scope}, "
                f"where {st.session_state.cq_affected.strip()}"
            )

            # --- Paragraph 2: State of Knowledge ---
            p2 = (
                f"The existing body of literature has made notable contributions to the understanding of "
                f"this subject. {st.session_state.cq_known.strip()} "
                f"However, despite these advances, {st.session_state.cq_unknown.strip()}"
            )

            # --- Paragraph 3: Gap Triangulation ---
            p3 = (
                f"A critical appraisal of the literature reveals a convergence of gaps across three dimensions. "
                f"Theoretically, {st.session_state.cq_gap_theory.strip().rstrip('.')}. "
                f"Empirically, {st.session_state.cq_gap_empirical.strip().rstrip('.')}. "
                f"From a practice standpoint, {st.session_state.cq_gap_practice.strip().rstrip('.')}. "
                f"This tri-dimensional gap underscores the urgent need for rigorous investigation."
            )

            # --- Paragraph 4: Variable Relationships ---
            mods_clause = f", moderated by {st.session_state.cq_moderators}," if st.session_state.cq_moderators else ""
            med_clause  = f" Potential mediating and confounding variables include {st.session_state.cq_mediators}." if st.session_state.cq_mediators else ""
            p4 = (
                f"The present study conceptualises {iv} as the independent variable{mods_clause} "
                f"with {dv} as the primary dependent outcome among {pop}{cc}{tc}. "
                f"This relationship is anchored in {st.session_state.cq_theory}, "
                f"which {st.session_state.cq_theory_link.strip() or 'provides the theoretical lens through which the study variables are examined'}."
                f"{med_clause}"
            )

            # --- Paragraph 5: Research Question & Objectives ---
            sub_q = [
                f"(i) describe the sociodemographic profile of {pop}",
                f"(ii) determine the prevalence or level of {dv}",
                f"(iii) examine the relationship between {iv} and {dv}{tc}",
            ]
            if st.session_state.cq_comparison:
                sub_q.append(f"(iv) compare {dv} between {iv} and {st.session_state.cq_comparison}")
            if st.session_state.cq_moderators:
                sub_q.append(f"({'v' if len(sub_q)==4 else 'iv'}) assess the moderating role of {st.session_state.cq_moderators}")
            p5 = (
                f"Guided by a {st.session_state.cq_design} research design within a "
                f"{st.session_state.cq_paradigm} paradigm, this study is directed by the central question: "
                f'"{st.session_state.cq_q_selected.strip()}" '
                f"Specifically, the study seeks to: {'; '.join(sub_q)}."
            )

            # --- Paragraph 6: Significance ---
            p6 = (
                f"The significance of this study extends to multiple stakeholders. "
                f"{st.session_state.cq_significance_who or 'Clinicians, researchers, and policymakers'} "
                f"stand to benefit through {st.session_state.cq_significance_how or 'enhanced evidence-based decision-making'}. "
                f"The timeliness of this inquiry is further justified by the fact that "
                f"{st.session_state.cq_significance_why or 'the current evidence base is insufficient to guide context-specific practice'}."
            )

            full_narrative = f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}\n\n{p5}\n\n{p6}"

            st.subheader("Full Conceptualisation Narrative")
            for label, para in [
                ("Background & Burden", p1),
                ("State of Knowledge", p2),
                ("Gap Triangulation", p3),
                ("Conceptual Framework & Variables", p4),
                ("Research Question & Design", p5),
                ("Significance", p6)
            ]:
                with st.expander(f"**{label}**", expanded=True):
                    st.write(para)

            st.divider()
            st.markdown("##### Hypotheses")
            st.markdown(f"**H₀:** There is no significant {('relationship between ' + iv + ' and ' + dv) if 'Correlat' in st.session_state.cq_design else ('effect of ' + iv + ' on ' + dv)} among {pop}.")
            st.markdown(f"**H₁:** There is a significant {('relationship between ' + iv + ' and ' + dv) if 'Correlat' in st.session_state.cq_design else ('effect of ' + iv + ' on ' + dv)} among {pop}{tc}.")

            st.divider()
            export = (
                "RESEARCH CONCEPTUALISATION NARRATIVE\n"
                "=" * 50 + "\n\n"
                f"RESEARCH QUESTION\n{st.session_state.cq_q_selected}\n\n"
                f"STUDY DESIGN: {st.session_state.cq_design}\n"
                f"PARADIGM: {st.session_state.cq_paradigm}\n"
                f"THEORY: {st.session_state.cq_theory}\n\n"
                "FULL NARRATIVE\n" + "-" * 30 + "\n\n"
                f"1. BACKGROUND & BURDEN\n{p1}\n\n"
                f"2. STATE OF KNOWLEDGE\n{p2}\n\n"
                f"3. GAP TRIANGULATION\n{p3}\n\n"
                f"4. CONCEPTUAL FRAMEWORK & VARIABLES\n{p4}\n\n"
                f"5. RESEARCH QUESTION & DESIGN\n{p5}\n\n"
                f"6. SIGNIFICANCE\n{p6}\n\n"
                f"HYPOTHESES\n"
                f"H0: No significant effect of {iv} on {dv} among {pop}.\n"
                f"H1: Significant effect of {iv} on {dv} among {pop}{tc}.\n"
            )
            st.download_button(
                "Download Full Narrative (.txt)",
                data=export,
                file_name="research_conceptualisation.txt",
                mime="text/plain"
            )

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("← Back to Stage 5"):
                st.session_state.cq_stage = 5; st.rerun()
        with bc2:
            if st.button("🔄 Start Over"):
                for k in list(st.session_state.keys()):
                    if k.startswith('cq_'):
                        del st.session_state[k]
                st.rerun()

# --- Module 2: Literature Search ---
elif module == "Literature Search":
    st.header("Literature Search")
    st.caption("Search 7 academic databases simultaneously — open-access and subscription sources.")

    if st.session_state.get('offline_mode'):
        st.info(
            "Offline mode is enabled. Search runs only against your saved local library. "
            "Import CSV in Library Diagnostics to expand offline coverage."
        )

    if st.session_state.library_reload_notice:
        st.success(st.session_state.library_reload_notice)
        st.session_state.library_reload_notice = ''

    query = st.text_input("Search keywords",
                          placeholder="e.g., prostate hyperplasia treatment outcomes Africa")

    with st.expander("Library Diagnostics"):
        _session_count = len(st.session_state.get('saved_papers', []))
        _disk_list = _load_saved_papers()
        _disk_count = len(_disk_list)
        st.write(f"Session library count: {_session_count}")
        st.write(f"Disk library count: {_disk_count}")
        st.write(f"Library file: {_LIBRARY_FILE}")
        if not _LIBRARY_FILE.exists():
            st.caption("No library file found on disk yet. Save or import papers to create it.")
        if st.button("Reload Library from Disk", key="reload_lib_from_disk"):
            _reload_library_from_disk()
            st.rerun()

        st.markdown("##### Import Library from CSV")
        _lib_upload = st.file_uploader(
            "Upload a library CSV",
            type=['csv'],
            key='library_csv_import',
            help="Required: Title. Optional: Authors, Year, Venue, Citations, Source, URL, PDF, Abstract."
        )
        if _lib_upload is not None and st.button("Import Uploaded Library", key='import_library_csv'):
            try:
                _df = pd.read_csv(_lib_upload)
                _col_map = {str(c).strip().lower(): c for c in _df.columns}
                if 'title' not in _col_map:
                    st.error("CSV must include a Title column.")
                else:
                    def _csv_get(row, key, default=''):
                        col = _col_map.get(key)
                        if not col:
                            return default
                        val = row.get(col, default)
                        if pd.isna(val):
                            return default
                        return val

                    _incoming = []
                    for _, _row in _df.iterrows():
                        _title = str(_csv_get(_row, 'title', '')).strip()
                        if not _title:
                            continue
                        _entry = {
                            'Title': _title,
                            'Authors': str(_csv_get(_row, 'authors', 'N/A')).strip() or 'N/A',
                            'Year': str(_csv_get(_row, 'year', 'n.d.')).strip() or 'n.d.',
                            'Venue': str(_csv_get(_row, 'venue', 'N/A')).strip() or 'N/A',
                            'Citations': _csv_get(_row, 'citations', 0),
                            'Source': str(_csv_get(_row, 'source', 'Imported')).strip() or 'Imported',
                            'URL': str(_csv_get(_row, 'url', '#')).strip() or '#',
                            'PDF': str(_csv_get(_row, 'pdf', '')).strip(),
                            'Abstract': str(_csv_get(_row, 'abstract', '')).strip(),
                        }
                        _entry['Paper ID'] = _paper_library_id(_entry)
                        _incoming.append(_entry)

                    _existing = _library_session()
                    _by_id = {
                        (p.get('Paper ID') or _paper_library_id(p)): {
                            **p,
                            'Paper ID': p.get('Paper ID') or _paper_library_id(p)
                        }
                        for p in _existing
                    }
                    _added = 0
                    for _p in _incoming:
                        if _p['Paper ID'] not in _by_id:
                            _by_id[_p['Paper ID']] = _p
                            _added += 1

                    _merged = list(_by_id.values())
                    st.session_state.saved_papers = _merged
                    if _save_saved_papers(_merged):
                        st.session_state.library_reload_notice = (
                            f"Library import complete: added {_added} paper(s), total {len(_merged)}."
                        )
                    else:
                        st.session_state.library_reload_notice = (
                            f"Imported {_added} paper(s) into session, but failed to persist to disk."
                        )
                    st.rerun()
            except Exception as _exc:
                st.error(f"Could not import library CSV: {_exc}")

    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
    with fc1:
        result_limit = st.slider("Max Results per Source", 5, 20, 8)
    with fc2:
        year_start = st.number_input("From Year", min_value=1900, max_value=2026, value=2015, step=1)
    with fc3:
        year_end = st.number_input("To Year", min_value=1900, max_value=2026, value=2026, step=1)
    with fc4:
        sort_by = st.selectbox("Sort By", ["relevance", "citations", "year"])

    # Source selector
    _ALL_SOURCES = ["Semantic Scholar", "PubMed", "Europe PMC",
                    "OpenAlex", "arXiv", "CrossRef", "DOAJ",
                    "Urology Journals"]
    if 'search_sources' not in st.session_state:
        st.session_state.search_sources = ["Semantic Scholar", "PubMed", "OpenAlex",
                                           "Urology Journals"]
    st.session_state.search_sources = st.multiselect(
        "Databases to search",
        options=_ALL_SOURCES,
        default=st.session_state.search_sources,
        help=(
            "**Semantic Scholar** — 200M+ papers, all fields | "
            "**PubMed** — biomedical (NLM) | "
            "**Europe PMC** — life sciences + full-text | "
            "**OpenAlex** — 250M+ works, all fields | "
            "**arXiv** — preprints (STEM) | "
            "**CrossRef** — DOI registry, all fields | "
            "**DOAJ** — open-access journals only | "
            "**Urology Journals** — PubMed filter across core urology titles"
        )
    )

    def _normalize_ss(papers):
            out = []
            for p in papers:
                authors = ', '.join(a.get('name', '') for a in (p.get('authors') or [])[:4])
                if len(p.get('authors') or []) > 4:
                    authors += ' et al.'
                out.append({
                    'Title':    p.get('title') or 'Untitled',
                    'Authors':  authors,
                    'Abstract': (p.get('abstract') or '')[:600],
                    'URL':      p.get('url') or '#',
                    'Year':     str(p.get('year') or 'N/A'),
                    'Venue':    p.get('venue') or 'N/A',
                    'Citations': p.get('citationCount', 0),
                    'Source':   'Semantic Scholar',
                    'PDF':      (p.get('openAccessPdf') or {}).get('url') or '',
                })
            return out

    if st.button("Search", type="primary") and query:
        selected = st.session_state.search_sources or ["Semantic Scholar"]

        if st.session_state.get('offline_mode'):
            deduped = _local_library_search(
                st.session_state.get('saved_papers', []),
                query,
                year_start=year_start,
                year_end=year_end,
                sort_by=sort_by,
            )
            selected = ["Saved Library"]
            source_errors = {}
            source_counts = {'Saved Library': len(deduped)}
            st.info("Results per source — **Saved Library**: " + str(len(deduped)))
        else:

            _source_fns = {
                "Semantic Scholar": lambda: _normalize_ss(
                    assistant.search_articles(query, result_limit, year_start, year_end, sort_by, api_key)
                ),
                "PubMed":      lambda: assistant.search_pubmed(query, result_limit, year_start, year_end),
                "Europe PMC":  lambda: assistant.search_europe_pmc(query, result_limit, year_start, year_end),
                "OpenAlex":    lambda: assistant.search_openalex(query, result_limit, year_start, year_end),
                "arXiv":       lambda: assistant.search_arxiv(query, result_limit, year_start, year_end),
                "CrossRef":    lambda: assistant.search_crossref(query, result_limit, year_start, year_end),
                "DOAJ":        lambda: assistant.search_doaj(query, result_limit, year_start, year_end),
                "Urology Journals": lambda: assistant.search_urology_journals(
                    query, result_limit, year_start, year_end
                ),
            }

            all_results, source_errors, source_counts = [], {}, {}

            with st.spinner(f"Querying {len(selected)} database(s)…"):
                with ThreadPoolExecutor(max_workers=min(len(selected), 6)) as ex:
                    futures = {ex.submit(_source_fns[s]): s
                               for s in selected if s in _source_fns}
                    for future in as_completed(futures):
                        src = futures[future]
                        try:
                            res = future.result()
                            source_counts[src] = len(res)
                            all_results.extend(res)
                        except requests.RequestException as e:
                            source_errors[src] = ("Rate limit — wait and retry."
                                                  if '429' in str(e) else str(e))
                        except Exception as e:
                            source_errors[src] = str(e)

            # Deduplicate by normalized title + year + source for safer merging
            seen, deduped = set(), []
            for r in all_results:
                key = (
                    str(r.get('Title', '')).lower().strip(),
                    str(r.get('Year', '')).strip(),
                    str(r.get('Source', '')).lower().strip(),
                )
                if key[0] and key not in seen:
                    seen.add(key)
                    deduped.append(r)

            # Client-side sort
            if sort_by == 'citations':
                deduped.sort(key=lambda x: int(x.get('Citations', 0) or 0), reverse=True)
            elif sort_by == 'year':
                deduped.sort(
                    key=lambda x: int(x.get('Year')) if str(x.get('Year', '')).isdigit() else -1,
                    reverse=True,
                )

            # Source summary
            summary_parts = [f"**{s}**: {source_counts.get(s, 0)}" for s in selected]
            st.info("Results per source — " + " | ".join(summary_parts))

        if source_errors:
            with st.expander(f"⚠️ {len(source_errors)} source error(s)"):
                for src, err in source_errors.items():
                    st.warning(f"**{src}:** {err}")

        if deduped:
            st.success(
                f"**{len(deduped)}** unique result(s) across {len(selected)} source(s) for **{query}**"
            )
            rows = []
            _bulk_save_options = {}
            for i, paper in enumerate(deduped):
                title     = paper.get('Title', 'Untitled')
                year      = paper.get('Year', 'N/A')
                venue     = paper.get('Venue', 'N/A')
                abstract  = paper.get('Abstract') or 'No abstract available.'
                url       = paper.get('URL') or '#'
                citations = paper.get('Citations', 0)
                authors   = paper.get('Authors', 'N/A')
                pdf_url   = paper.get('PDF') or ''
                source    = paper.get('Source', '')

                rows.append({'Title': title, 'Year': year, 'Venue': venue,
                             'Citations': citations, 'Authors': authors,
                             'Abstract': abstract, 'URL': url, 'Source': source})

                entry = {
                    'Paper ID': _paper_library_id(paper),
                    'Title': title, 'Year': year, 'Authors': authors,
                    'Venue': venue, 'URL': url, 'Citations': citations,
                    'Abstract': abstract[:600], 'PDF': pdf_url,
                    'Source': source,
                }
                label = f"[{i+1}] {title[:90]} ({year}) — {source}"
                _bulk_save_options[label] = entry

                with st.expander(f"[{i+1}] {title} ({year})  —  `{source}`"):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**Authors:** {authors or 'N/A'}")
                        st.markdown(
                            f"**Venue:** {venue} &nbsp;|&nbsp; "
                            f"**Citations:** {citations} &nbsp;|&nbsp; "
                            f"**Source:** {source}"
                        )
                        st.write(abstract)
                    with col_b:
                        if url and url != '#':
                            st.markdown(f"[Open Paper]({url})")
                        if pdf_url:
                            st.markdown(f"[Open Access PDF]({pdf_url})")
                            if st.toggle("View PDF in Browser", key=f"view_pdf_{i}"):
                                _safe = _safe_pdf_url(pdf_url)
                                if _safe:
                                    st.components.v1.iframe(_safe, height=PDF_VIEWER_HEIGHT, scrolling=True)
                                else:
                                    st.warning("PDF cannot be displayed inline (URL must use HTTPS).")
                        if st.button("Save to Library", key=f"save_{i}"):
                            _added, _status = _add_to_library(entry)
                            if not _added:
                                st.info("Already in library.")
                            elif _status == "saved":
                                st.success("Saved!")
                            else:
                                st.warning("Saved in session, but failed to persist to disk.")

            st.divider()
            st.markdown("#### Save Multiple Results to Library")
            _bulk_selected = st.multiselect(
                "Select result(s) to save",
                options=list(_bulk_save_options.keys()),
                key='bulk_save_selection'
            )
            if st.button("Save Selected Results", key='save_selected_results'):
                _added = 0
                _session_only = 0
                for _label in _bulk_selected:
                    _entry = _bulk_save_options.get(_label)
                    if not _entry:
                        continue
                    _ok, _status = _add_to_library(_entry)
                    if _ok and _status in ("saved", "session_only"):
                        _added += 1
                    if _ok and _status == "session_only":
                        _session_only += 1
                if _added:
                    if _session_only == 0:
                        st.success(f"Saved {_added} paper(s) to library.")
                    else:
                        st.warning(
                            f"Saved {_added} paper(s), but {_session_only} could not be persisted to disk."
                        )
                else:
                    st.info("No new papers were added.")

            if st.button("Save All Results to Library", key='save_all_results'):
                _added_all = 0
                _session_only_all = 0
                for _entry in _bulk_save_options.values():
                    _ok, _status = _add_to_library(_entry)
                    if _ok and _status in ("saved", "session_only"):
                        _added_all += 1
                    if _ok and _status == "session_only":
                        _session_only_all += 1
                if _added_all:
                    if _session_only_all == 0:
                        st.success(f"Saved {_added_all} result(s) to library.")
                    else:
                        st.warning(
                            f"Saved {_added_all} result(s), but {_session_only_all} could not be persisted to disk."
                        )
                else:
                    st.info("All current results are already in library.")

            df_results = pd.DataFrame(rows)
            st.download_button(
                "Export Results to CSV",
                data=df_results.to_csv(index=False).encode('utf-8-sig'),
                file_name=f"search_{query.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No results found. Try different keywords, a wider year range, or add more sources.")

    # Saved Papers Library
    if st.session_state.saved_papers:
        st.divider()
        st.subheader(f"My Library ({len(st.session_state.saved_papers)} papers)")
        df_lib = pd.DataFrame(st.session_state.saved_papers)
        show_cols = [c for c in ['Title', 'Authors', 'Year', 'Venue', 'Citations', 'Source']
                     if c in df_lib.columns]
        st.dataframe(df_lib[show_cols], width='stretch', hide_index=True)
        with st.expander("Manage Saved Papers"):
            remove_ids = []
            for idx, paper in enumerate(st.session_state.saved_papers):
                paper_id = paper.get('Paper ID') or _paper_library_id(paper)
                with st.container():
                    st.markdown(f"**{paper.get('Title', 'Untitled')}**")
                    st.caption(
                        f"{paper.get('Authors', 'N/A')} | {paper.get('Year', 'n.d.')} | "
                        f"{paper.get('Venue', 'N/A')} | {paper.get('Source', 'Saved Library')}"
                    )
                    if paper.get('Abstract'):
                        st.write(paper.get('Abstract'))
                    link_cols = st.columns(2)
                    with link_cols[0]:
                        if paper.get('URL') and paper.get('URL') != '#':
                            st.markdown(f"[Open Paper]({paper.get('URL')})")
                    with link_cols[1]:
                        if paper.get('PDF'):
                            st.markdown(f"[Open Access PDF]({paper.get('PDF')})")
                            if st.toggle("View PDF in Browser", key=f"view_lib_pdf_{paper_id}"):
                                _safe = _safe_pdf_url(paper.get('PDF'))
                                if _safe:
                                    st.components.v1.iframe(_safe, height=PDF_VIEWER_HEIGHT, scrolling=True)
                                else:
                                    st.warning("PDF cannot be displayed inline (URL must use HTTPS).")
                    if st.checkbox("Remove this paper", key=f"remove_saved_{paper_id}"):
                        remove_ids.append(paper_id)
                    st.divider()
            if remove_ids:
                st.session_state.saved_papers = [
                    p for p in st.session_state.saved_papers
                    if (p.get('Paper ID') or _paper_library_id(p)) not in remove_ids
                ]
                if not _save_saved_papers(st.session_state.saved_papers):
                    st.warning("Removed in session, but failed to persist to disk.")
                st.rerun()
        st.download_button(
            "Export Library to CSV",
            data=df_lib.to_csv(index=False).encode('utf-8-sig'),
            file_name="my_library.csv",
            mime="text/csv"
        )
        if st.button("Clear Library"):
            st.session_state.saved_papers = []
            if not _save_saved_papers(st.session_state.saved_papers):
                st.warning("Cleared in session, but failed to persist to disk.")
            st.rerun()

# --- Module: Literature Review ---
elif module == "Literature Review":
    st.header("Literature Review Synthesizer")
    st.caption(
        "Analyze articles in relation to your research question and generate "
        "a structured, thematic literature review draft."
    )
    st.info(
        "Select relevant articles, save them with abstracts into a focused review list, "
        "clear the list when needed, and export the selected articles as CSV."
    )

    if st.session_state.library_reload_notice:
        st.success(st.session_state.library_reload_notice)
        st.session_state.library_reload_notice = ''

    # ── Local helpers ────────────────────────────────────────────────────────
    _LR_STOP = {
        'a','an','the','and','or','but','in','on','at','to','for','of','with',
        'by','from','as','is','was','are','were','be','been','being','have',
        'has','had','do','does','did','will','would','could','should','may',
        'might','shall','can','this','that','these','those','it','its','they',
        'them','their','we','our','you','your','he','his','she','her','i',
        'my','me','who','which','what','how','when','where','why','all','each',
        'every','both','few','more','most','other','some','such','than','then',
        'there','so','if','not','no','nor','into','through','during','before',
        'after','above','below','between','out','off','over','under','again',
        'further','among','also','about','against','because','while','although',
        'however','therefore','thus','hence','whereas','upon','used','need',
    }

    def _lr_tokenize(text):
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in _LR_STOP]

    def _lr_keywords(rq_text, top_n=15):
        tokens = _lr_tokenize(rq_text)
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        return sorted(freq, key=lambda x: -freq[x])[:top_n]

    def _lr_score(paper, keywords):
        if not keywords:
            return 0.0
        title    = (paper.get('Title', '') or '').lower()
        abstract = (paper.get('Abstract', '') or '').lower()
        hits = sum(
            (3 if kw in title else 0) + (1 if kw in abstract else 0)
            for kw in keywords
        )
        return round(min(hits / (4 * len(keywords)), 1.0), 3)

    def _lr_key_sentences(abstract, keywords, n=2):
        if not abstract or not keywords:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())
        scored = [(sum(1 for kw in keywords if kw in s.lower()), s)
                  for s in sentences]
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:n] if s.strip()]

    def _lr_clusters(papers, keywords):
        clusters = {}
        for p in papers:
            text = ((p.get('Title', '') or '') + ' ' + (p.get('Abstract', '') or '')).lower()
            hit_kws = [kw for kw in keywords if kw in text]
            theme = hit_kws[0] if hit_kws else 'general'
            clusters.setdefault(theme, []).append(p)
        merged = {}
        for theme, ps in clusters.items():
            if len(ps) >= 2:
                merged[theme] = ps
            else:
                merged.setdefault('general', []).extend(ps)
        if len(merged) > 6:
            sorted_themes = sorted(merged.items(), key=lambda x: -len(x[1]))
            top = dict(sorted_themes[:5])
            others = [p for _, ps in sorted_themes[5:] for p in ps]
            if others:
                top['other'] = others
            return top
        return merged

    def _lr_para(theme_kw, papers, rq, cite_style):
        cite_tags = [in_text_cite(p, style=cite_style, number=i + 1)
                     for i, p in enumerate(papers)]
        cite_str = '; '.join(cite_tags)
        rq_kws = _lr_tokenize(rq)
        snippet_parts = []
        for i, p in enumerate(papers[:2]):
            abst = p.get('Abstract', '')
            sents = _lr_key_sentences(abst, rq_kws, n=1)
            if sents and i < len(cite_tags):
                snippet_parts.append(
                    f'{cite_tags[i]} found that "{sents[0][:130].rstrip()}\u2026"'
                )
        rq_short = rq[:120] + ('\u2026' if len(rq) > 120 else '')
        parts = [f"With regard to {theme_kw}, several studies provide relevant evidence."]
        if snippet_parts:
            parts.append(' '.join(snippet_parts))
        parts.append(
            f"Taken together, {cite_str} contribute to understanding this dimension "
            f"in relation to the research question: *{rq_short}* "
            f"[Expand with your own critical analysis and synthesis of the above findings.]"
        )
        return ' '.join(parts)

    # ── Research Question Input ─────────────────────────────────────────────
    lr_rq = st.text_area(
        "Research Question",
        value=st.session_state.get('cq_q_selected', ''),
        height=70,
        placeholder="e.g., What is the effect of alpha-blockers on symptom scores among men with BPH?",
        help="Auto-filled from Conceptualization (Stage 5) if completed."
    )

    st.divider()
    st.markdown("#### Article Sources")

    src_tab1, src_tab2, src_tab3 = st.tabs([
        "From Saved Library", "Upload CSV", "Paste Manually"
    ])

    lr_papers = []
    if 'lr_selected_articles' not in st.session_state:
        st.session_state.lr_selected_articles = []

    with src_tab1:
        _lr_saved = list(st.session_state.get('saved_papers', []) or [])
        _disk_saved = _load_saved_papers()

        # Keep session and disk library views in sync so recently saved papers
        # are consistently available in Literature Review.
        _merged = {}
        for _p in _lr_saved + _disk_saved:
            _pid = _p.get('Paper ID') or _paper_library_id(_p)
            _merged[_pid] = {**_p, 'Paper ID': _pid}
        if _merged:
            _lr_saved = list(_merged.values())
            st.session_state.saved_papers = _lr_saved

        if st.button("Refresh Library", key='lr_refresh_library'):
            _reload_library_from_disk()
            st.rerun()

        if not _lr_saved:
            st.info(
                "No papers saved yet. Use **Literature Search** to find articles "
                "and click **Save to Library** on relevant results."
            )
        else:
            st.success(f"{len(_lr_saved)} paper(s) in your library.")
            _preview_rows = [{
                'Title': _p.get('Title', 'Untitled'),
                'Authors': _p.get('Authors', 'N/A'),
                'Year': _p.get('Year', 'n.d.'),
                'Source': _p.get('Source', 'Saved Library'),
            } for _p in _lr_saved]
            st.dataframe(pd.DataFrame(_preview_rows), width='stretch', hide_index=True)

            _paper_options = {
                f"{_p.get('Title', 'Untitled')} ({_p.get('Year', 'n.d.')}) [{_p.get('Source', 'Saved')}]": _p
                for _p in _lr_saved
            }
            lr_use_all = st.checkbox("Use all library papers", value=True, key='lr_use_all')
            if lr_use_all:
                lr_papers = list(_lr_saved)
            else:
                lr_sel = st.multiselect(
                    "Select papers to include",
                    options=list(_paper_options.keys()),
                    key='lr_sel_titles'
                )
                lr_papers = [_paper_options[k] for k in lr_sel if k in _paper_options]

    with src_tab2:
        lr_upload = st.file_uploader(
            "Upload CSV with columns: Title, Abstract "
            "(and optionally Authors, Year, Venue, Citations, URL)",
            type=['csv'], key='lr_upload'
        )
        if lr_upload:
            try:
                df_up = pd.read_csv(lr_upload)
                if {'Title', 'Abstract'}.issubset(set(df_up.columns)):
                    lr_papers = df_up.fillna('').to_dict('records')
                    for p in lr_papers:
                        p.setdefault('Authors', 'N/A')
                        p.setdefault('Year', 'n.d.')
                        p.setdefault('Venue', 'N/A')
                        p.setdefault('Citations', 0)
                        p.setdefault('URL', '#')
                        p.setdefault('Source', 'Uploaded')
                    st.success(f"{len(lr_papers)} article(s) loaded from CSV.")
                else:
                    missing_c = {'Title', 'Abstract'} - set(df_up.columns)
                    st.error(f"CSV is missing required columns: {missing_c}")
            except Exception as _exc:
                st.error(f"Could not parse CSV: {_exc}")

    with src_tab3:
        st.caption(
            "Paste one article per block, blocks separated by `---`. "
            "Each block should have lines like `Title: ...`, `Authors: ...`, "
            "`Year: ...`, `Abstract: ...`"
        )
        manual_raw = st.text_area(
            "Paste articles here",
            height=220,
            placeholder=(
                "Title: Effect of alpha-blockers on BPH\n"
                "Authors: Smith J, Doe A\n"
                "Year: 2022\n"
                "Abstract: This study examined...\n\n"
                "---\n\n"
                "Title: Next article\n"
                "Authors: Jones B\n"
                "Year: 2021\n"
                "Abstract: ..."
            ),
            key='lr_manual'
        )
        if manual_raw.strip():
            _key_map = {
                'title': 'Title', 'authors': 'Authors', 'author': 'Authors',
                'year': 'Year', 'abstract': 'Abstract', 'venue': 'Venue',
                'journal': 'Venue', 'citations': 'Citations', 'url': 'URL',
            }
            _blocks = re.split(r'\n\s*---+\s*\n', manual_raw.strip())
            _parsed = []
            for _block in _blocks:
                _p = {}
                for _line in _block.strip().split('\n'):
                    if ':' in _line:
                        _k, _, _v = _line.partition(':')
                        _k = _k.strip().lower()
                        if _k in _key_map:
                            _p[_key_map[_k]] = _v.strip()
                if _p.get('Title'):
                    _p.setdefault('Authors', 'N/A')
                    _p.setdefault('Year', 'n.d.')
                    _p.setdefault('Abstract', '')
                    _p.setdefault('Venue', 'N/A')
                    _p.setdefault('Citations', 0)
                    _p.setdefault('URL', '#')
                    _p.setdefault('Source', 'Manual')
                    _parsed.append(_p)
            if _parsed:
                lr_papers = _parsed
                st.success(f"{len(_parsed)} article(s) parsed from manual input.")

    st.divider()

    _btn_disabled = not (lr_papers and lr_rq.strip())
    if _btn_disabled:
        st.warning(
            "⚠️ To analyze, add articles (use tabs above) and enter a research question above."
        )
        with st.expander("📖 How to Use Literature Review Synthesizer", expanded=not lr_papers):
            st.markdown("""
**Step 1: Add Articles**
- Use the tabs above to load articles:
  - **From Saved Library**: Papers you saved in Literature Search
  - **Upload CSV**: Import from a spreadsheet (columns: Title, Abstract, Authors, Year, Venue, URL)
  - **Paste Manually**: Enter article details manually

**Step 2: Enter Research Question**
- Write your research question in the field above
- The synthesizer will extract keywords and match them to your articles

**Step 3: Click Analyze & Synthesize**
- Relevance ranking will show which papers best match your question
- Per-article summaries will highlight relevant passages
- Thematic synthesis groups articles and generates draft paragraphs
- Export the full literature review as text

**Step 4: Select & Export**
- Mark relevant articles and export them as CSV
- Use the generated paragraphs in your manuscript
            """)
    
    if st.button("Analyze & Synthesize", type="primary", key='lr_run', disabled=_btn_disabled):
        _kws = _lr_keywords(lr_rq, top_n=15)
        _cite_style = st.session_state.get('citation_style', 'APA 7th')

        for _p in lr_papers:
            _p['_lr_rel'] = _lr_score(_p, _kws)
        lr_papers.sort(key=lambda x: -x['_lr_rel'])

        st.subheader("Analysis Results")
        st.markdown("**Keywords extracted from research question:**")
        st.write("  ".join(f"`{kw}`" for kw in _kws))
        st.divider()

        # ── Relevance table ───────────────────────────────────────────
        st.markdown("#### Relevance Ranking")
        _rank_rows = [{
            'Rank': _i,
            'Title': _p.get('Title', 'Untitled')[:75],
            'Authors': (_p.get('Authors', 'N/A') or 'N/A')[:40],
            'Year': _p.get('Year', 'n.d.'),
            'Relevance': f"{_p['_lr_rel']:.0%}",
            'Citations': _p.get('Citations', 0),
            'Source': _p.get('Source', ''),
        } for _i, _p in enumerate(lr_papers, start=1)]
        st.dataframe(pd.DataFrame(_rank_rows), width='stretch', hide_index=True)

        # ── Per-article summaries ─────────────────────────────────────
        st.divider()
        st.markdown("#### Per-Article Summaries in Relation to Your Research Question")
        for _i, _p in enumerate(lr_papers, start=1):
            _rel = _p['_lr_rel']
            _rel_label = "High" if _rel >= 0.4 else ("Moderate" if _rel >= 0.2 else "Low")
            with st.expander(
                f"[{_i}] {_p.get('Title','Untitled')[:75]} "
                f"({_p.get('Year','n.d.')}) — {_rel_label} relevance ({_rel:.0%})"
            ):
                _abst = _p.get('Abstract', '')
                _key_sents = _lr_key_sentences(_abst, _kws, n=2)
                st.markdown(
                    f"**Authors:** {_p.get('Authors','N/A')}  |  "
                    f"**Venue:** {_p.get('Venue','N/A')}  |  "
                    f"**Year:** {_p.get('Year','n.d.')}"
                )
                st.markdown(
                    f"**Citation:** {in_text_cite(_p, style=_cite_style, number=_i)}"
                )
                if _key_sents:
                    st.markdown("**Most Relevant Sentences from Abstract:**")
                    for _s in _key_sents:
                        st.info(_s)
                elif _abst:
                    st.caption(_abst[:400] + ('\u2026' if len(_abst) > 400 else ''))
                else:
                    st.caption("*No abstract available.*")
                _url = _p.get('URL', '')
                if _url and _url != '#':
                    st.markdown(f"[Open Full Article]({_url})")

        # ── Relevant article selection ───────────────────────────────
        st.divider()
        st.markdown("#### Select Relevant Articles for Your Review")
        st.caption(
            "Mark the papers that are most relevant, then save them as a focused review set."
        )

        _selection_rows = []
        for _i, _p in enumerate(lr_papers, start=1):
            _sel_key = f"lr_pick_{_i}"
            _selected = st.checkbox(
                f"[{_i}] {_p.get('Title', 'Untitled')[:85]}",
                key=_sel_key,
                help="Select this paper for the focused review set."
            )
            if _selected:
                _selection_rows.append((_i, _p))

        c_save, c_clear = st.columns(2)
        with c_save:
            if st.button("Save Selected Articles", key='lr_save_selected'):
                _existing_keys = {
                    (
                        (p.get('Title') or '').strip().lower(),
                        str(p.get('Year', 'n.d.')).strip(),
                        (p.get('Source') or '').strip().lower(),
                    )
                    for p in st.session_state.lr_selected_articles
                }
                _added = 0
                for _rank, _p in _selection_rows:
                    _entry = {
                        'Title': _p.get('Title', 'Untitled'),
                        'Authors': _p.get('Authors', 'N/A'),
                        'Year': _p.get('Year', 'n.d.'),
                        'Venue': _p.get('Venue', 'N/A'),
                        'Citations': _p.get('Citations', 0),
                        'Source': _p.get('Source', 'Selected'),
                        'URL': _p.get('URL', '#'),
                        'PDF': _p.get('PDF', ''),
                        'Abstract': _p.get('Abstract', ''),
                        'Relevance': f"{_p.get('_lr_rel', 0):.0%}",
                        'Rank': _rank,
                    }
                    _entry_key = (
                        _entry['Title'].strip().lower(),
                        str(_entry['Year']).strip(),
                        _entry['Source'].strip().lower(),
                    )
                    if _entry_key not in _existing_keys:
                        st.session_state.lr_selected_articles.append(_entry)
                        _existing_keys.add(_entry_key)
                        _added += 1
                if _added:
                    st.success(f"Saved {_added} relevant article(s) to your review set.")
                else:
                    st.info("No new articles were added.")
        with c_clear:
            if st.button("Clear Saved Review Set", key='lr_clear_selected'):
                st.session_state.lr_selected_articles = []
                st.rerun()

        if st.session_state.lr_selected_articles:
            st.markdown("#### Saved Relevant Articles")
            _saved_review_df = pd.DataFrame(st.session_state.lr_selected_articles)
            _show_cols = [
                c for c in ['Rank', 'Title', 'Authors', 'Year', 'Venue',
                            'Citations', 'Source', 'Relevance', 'Abstract']
                if c in _saved_review_df.columns
            ]
            st.dataframe(_saved_review_df[_show_cols], width='stretch', hide_index=True)
            _export_cols = [
                c for c in ['Rank', 'Title', 'Authors', 'Year', 'Venue',
                            'Citations', 'Source', 'Relevance', 'Abstract',
                            'URL', 'PDF']
                if c in _saved_review_df.columns
            ]
            st.download_button(
                "Export Relevant Articles to CSV",
                data=_saved_review_df[_export_cols].to_csv(index=False).encode('utf-8-sig'),
                file_name="relevant_articles_for_review.csv",
                mime="text/csv"
            )

        # ── Thematic clusters & draft ─────────────────────────────────
        st.divider()
        st.markdown("#### Thematic Synthesis")
        _clusters = _lr_clusters(lr_papers, _kws)
        _theme_names = list(_clusters.keys())

        if _theme_names:
            _tabs = st.tabs([f"Theme: {t.title()}" for t in _theme_names])
            _draft_parts = []
            for _tab, _theme in zip(_tabs, _theme_names):
                with _tab:
                    _tpapers = _clusters[_theme]
                    st.markdown(f"**{len(_tpapers)} article(s) in this theme**")
                    _para = _lr_para(_theme, _tpapers, lr_rq, _cite_style)
                    st.text_area(
                        "Draft paragraph (edit placeholders in [ ] before using):",
                        value=_para, height=165, key=f"lr_para_{_theme}"
                    )
                    st.markdown("**Articles in this theme:**")
                    for _j, _tp in enumerate(_tpapers, start=1):
                        st.markdown(
                            f"{_j}. {format_reference(_tp, style=_cite_style, number=_j)}"
                        )
                    _draft_parts.append((_theme, _para))

            # ── Full draft ────────────────────────────────────────────
            st.divider()
            st.markdown("#### Full Draft Literature Review")
            _full_draft = (
                f"RESEARCH QUESTION\n{lr_rq}\n\n{'=' * 60}\n\n"
            )
            for _theme, _para in _draft_parts:
                _full_draft += f"{_theme.upper()}\n{'-' * 40}\n{_para}\n\n"
            _full_draft += f"\nREFERENCE LIST\n{'=' * 60}\n\n"
            for _i, _p in enumerate(lr_papers, start=1):
                _full_draft += format_reference(_p, style=_cite_style, number=_i) + "\n\n"

            st.text_area(
                "Full draft (copy or download):",
                value=_full_draft, height=350, key='lr_full_draft'
            )
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "Download Literature Review Draft (.txt)",
                    data=_full_draft,
                    file_name="literature_review_draft.txt",
                    mime="text/plain"
                )
            with col_dl2:
                if st.button("Send to Manuscript Drafter", key='lr_send_ms'):
                    _existing = st.session_state.get('ms_background', '')
                    st.session_state['ms_background'] = (
                        _existing + ('\n\n' if _existing else '') +
                        _full_draft[:3000]
                    )
                    st.success(
                        "Added to Manuscript Drafter → Introduction/Background. "
                        "Navigate there to review and edit."
                    )

# --- Module 3: Methodology & Stats ---
elif module == "Methodology & Stats":
    st.header("Statistical Tool Design")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Sample Size & Power",
        "Statistical Test Selector",
        "Effect Size Calculator",
        "Outcomes Analysis Guide"
    ])

    # ── Tab 1: Sample Size & Power ──────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Size Calculator (Cochran)")
            pop        = st.number_input("Target Population Size", value=1000, min_value=1)
            confidence = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1,
                                      format_func=lambda x: f"{int(x*100)}%")
            margin     = st.slider("Margin of Error", 0.01, 0.10, 0.05, step=0.01,
                                   format="%.2f")
            try:
                n = assistant.calculate_sample_size(pop, confidence=confidence,
                                                    margin_error=margin)
                st.success(f"Recommended Sample Size: **{n}**")
            except ValueError as e:
                st.error(str(e))

        with col2:
            st.subheader("Statistical Power Analysis")
            effect_input = st.selectbox("Expected Effect Size",
                                        ["Small (d=0.2)", "Medium (d=0.5)", "Large (d=0.8)"])
            alpha        = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
            desired_power = st.slider("Desired Power (1−β)", 0.70, 0.95, 0.80, step=0.05,
                                      format="%.2f")
            effect_map   = {"Small (d=0.2)": 0.2, "Medium (d=0.5)": 0.5, "Large (d=0.8)": 0.8}
            d            = effect_map[effect_input]
            z_alpha      = stats.norm.ppf(1 - alpha / 2)
            z_beta       = stats.norm.ppf(desired_power)
            n_power      = int(np.ceil(2 * ((z_alpha + z_beta) / d) ** 2))
            st.success(f"Required n per group: **{n_power}**")
            st.caption(f"Total (2 groups): **{n_power * 2}**  |  Formula: 2·((Zα/2+Zβ)/d)²")

    # ── Tab 2: Statistical Test Selector ────────────────────────────────────
    with tab2:
        st.subheader("Statistical Test Decision Tool")

        s1, s2 = st.columns(2)
        with s1:
            study_type = st.selectbox("Study Type / Design", [
                "Experimental / RCT",
                "Quasi-Experimental (Pre–Post)",
                "Cross-Sectional",
                "Prospective Cohort",
                "Retrospective Cohort",
                "Case-Control",
                "Correlational / Survey",
                "Predictive / Regression",
                "Longitudinal / Repeated Measures",
                "Diagnostic Accuracy",
                "Systematic Review / Meta-Analysis",
                "Qualitative"
            ])
            dep_type = st.selectbox("Dependent Variable Type", [
                "Continuous (Normally Distributed)",
                "Continuous (Non-Normal / Skewed)",
                "Ordinal (Likert / Ranked)",
                "Binary / Dichotomous",
                "Nominal / Categorical (>2 groups)",
                "Count / Rate",
                "Survival / Time-to-Event",
                "Multiple Outcomes"
            ])
        with s2:
            groups      = st.selectbox("Number of Groups / Comparisons",
                                       ["1 (single sample)", "2 (independent)",
                                        "2 (paired/matched)", "3+ (independent)",
                                        "3+ (repeated measures)", "Not applicable"])
            covariate   = st.checkbox("Covariates / Confounders to control for?")
            multilevel  = st.checkbox("Nested / Clustered data?")

        # Decision matrix
        TEST_MAP = {
            ("Experimental / RCT", "Continuous (Normally Distributed)", "2 (independent)"):
                ("Independent Samples T-Test", "Compare means of 2 independent groups.", "Cohen's d"),
            ("Experimental / RCT", "Continuous (Normally Distributed)", "2 (paired/matched)"):
                ("Paired Samples T-Test", "Compare pre–post means within the same group.", "Cohen's d (paired)"),
            ("Experimental / RCT", "Continuous (Normally Distributed)", "3+ (independent)"):
                ("One-Way ANOVA", "Compare means across 3+ independent groups; follow with Tukey HSD post-hoc.", "η²"),
            ("Experimental / RCT", "Continuous (Normally Distributed)", "3+ (repeated measures)"):
                ("Repeated Measures ANOVA", "Compare means across 3+ time points; check sphericity (Mauchly's test).", "η²"),
            ("Experimental / RCT", "Continuous (Non-Normal / Skewed)", "2 (independent)"):
                ("Mann-Whitney U Test", "Non-parametric alternative to independent T-test.", "r = Z/√N"),
            ("Experimental / RCT", "Continuous (Non-Normal / Skewed)", "2 (paired/matched)"):
                ("Wilcoxon Signed-Rank Test", "Non-parametric alternative to paired T-test.", "r = Z/√N"),
            ("Experimental / RCT", "Continuous (Non-Normal / Skewed)", "3+ (independent)"):
                ("Kruskal-Wallis H Test", "Non-parametric alternative to One-Way ANOVA; follow with Dunn's post-hoc.", "η²H"),
            ("Experimental / RCT", "Binary / Dichotomous", "2 (independent)"):
                ("Chi-Square Test / Fisher's Exact", "Compare proportions; use Fisher's when cell counts < 5.", "Risk Ratio (RR) / NNT"),
            ("Quasi-Experimental (Pre–Post)", "Continuous (Normally Distributed)", "2 (paired/matched)"):
                ("Paired T-Test / ANCOVA", "Control for baseline differences using ANCOVA with pre-score as covariate.", "Cohen's d"),
            ("Quasi-Experimental (Pre–Post)", "Continuous (Non-Normal / Skewed)", "2 (paired/matched)"):
                ("Wilcoxon Signed-Rank Test", "Non-parametric pre–post comparison.", "r = Z/√N"),
            ("Cross-Sectional", "Binary / Dichotomous", "Not applicable"):
                ("Chi-Square / Logistic Regression", "For association; use binary logistic regression with covariates.", "OR / Cramér's V"),
            ("Cross-Sectional", "Continuous (Normally Distributed)", "Not applicable"):
                ("Pearson Correlation / Multiple Regression", "Assess linear relationships and predictors.", "Pearson r / R²"),
            ("Cross-Sectional", "Ordinal (Likert / Ranked)", "Not applicable"):
                ("Spearman ρ / Ordinal Logistic Regression", "For ranked/Likert data; avoid treating ordinal as interval.", "Spearman ρ"),
            ("Prospective Cohort", "Binary / Dichotomous", "Not applicable"):
                ("Risk Ratio (RR) / Cox Regression", "RR for unadjusted; Cox proportional hazards with covariates.", "HR / RR"),
            ("Prospective Cohort", "Survival / Time-to-Event", "Not applicable"):
                ("Kaplan-Meier + Log-Rank / Cox PH", "KM curves for visualization; Cox PH for adjusted hazard ratios.", "Hazard Ratio (HR)"),
            ("Retrospective Cohort", "Binary / Dichotomous", "Not applicable"):
                ("Odds Ratio / Logistic Regression", "OR for retrospective design; adjust for confounders.", "OR / aOR"),
            ("Case-Control", "Binary / Dichotomous", "2 (matched/paired)" if False else "Not applicable"):
                ("Odds Ratio / Conditional Logistic Regression", "OR is the appropriate measure; conditional logistic for matched pairs.", "OR"),
            ("Case-Control", "Binary / Dichotomous", "Not applicable"):
                ("Odds Ratio / Logistic Regression", "OR is the appropriate measure for case-control designs.", "OR"),
            ("Correlational / Survey", "Continuous (Normally Distributed)", "Not applicable"):
                ("Pearson Correlation", "Measures linear association between two continuous variables.", "r"),
            ("Correlational / Survey", "Ordinal (Likert / Ranked)", "Not applicable"):
                ("Spearman ρ / Kendall τ", "For ordinal/non-normal data; Kendall τ preferred for small samples.", "ρ / τ"),
            ("Correlational / Survey", "Binary / Dichotomous", "Not applicable"):
                ("Point-Biserial Correlation / Phi Coefficient", "Point-biserial for continuous+binary; phi for binary+binary.", "rpb / φ"),
            ("Predictive / Regression", "Continuous (Normally Distributed)", "Not applicable"):
                ("Multiple Linear Regression", "Check assumptions: linearity, homoscedasticity, normality of residuals, no multicollinearity.", "R² / Adjusted R²"),
            ("Predictive / Regression", "Binary / Dichotomous", "Not applicable"):
                ("Binary Logistic Regression", "Reports OR; assess model fit with Hosmer-Lemeshow, Nagelkerke R².", "OR / Nagelkerke R²"),
            ("Predictive / Regression", "Nominal / Categorical (>2 groups)", "Not applicable"):
                ("Multinomial Logistic Regression", "Use with nominal DV with >2 unordered categories.", "OR"),
            ("Predictive / Regression", "Ordinal (Likert / Ranked)", "Not applicable"):
                ("Ordinal Logistic Regression", "For ordered outcome categories; check proportional odds assumption.", "OR"),
            ("Predictive / Regression", "Count / Rate", "Not applicable"):
                ("Poisson / Negative Binomial Regression", "Poisson for counts; Negative Binomial if overdispersed (Var > Mean).", "IRR"),
            ("Longitudinal / Repeated Measures", "Continuous (Normally Distributed)", "3+ (repeated measures)"):
                ("Mixed-Effects / Linear Mixed Model (LMM)", "Handles missing data and individual variation over time.", "Cohen's d / η²"),
            ("Longitudinal / Repeated Measures", "Binary / Dichotomous", "3+ (repeated measures)"):
                ("Generalized Estimating Equations (GEE)", "For correlated binary outcomes over time (population-averaged effects).", "OR"),
            ("Diagnostic Accuracy", "Binary / Dichotomous", "Not applicable"):
                ("Sensitivity, Specificity, ROC-AUC, LR+/LR−", "Calculate AUC for discrimination; optimal cutoff via Youden's index.", "AUC / Youden's J"),
            ("Systematic Review / Meta-Analysis", "Continuous (Normally Distributed)", "Not applicable"):
                ("Fixed / Random Effects Meta-Analysis (SMD)", "Use Random Effects when heterogeneity (I² > 50%); report I², τ², funnel plot.", "SMD / Hedges' g"),
            ("Systematic Review / Meta-Analysis", "Binary / Dichotomous", "Not applicable"):
                ("Meta-Analysis of Proportions / OR / RR", "Pool OR or RR; assess heterogeneity with I², Cochran's Q.", "OR / RR / I²"),
            ("Qualitative", "Multiple Outcomes", "Not applicable"):
                ("Thematic Analysis / Content Analysis / Grounded Theory",
                 "Choose based on paradigm: Thematic for patterns, Grounded Theory for theory-building, Content Analysis for frequency.", "N/A"),
        }

        key = (study_type, dep_type, groups)
        result = TEST_MAP.get(key)

        # Fallback: partial match on study_type + dep_type
        if not result:
            for k, v in TEST_MAP.items():
                if k[0] == study_type and k[1] == dep_type:
                    result = v
                    break
        if not result:
            for k, v in TEST_MAP.items():
                if k[0] == study_type:
                    result = v
                    break

        if result:
            test_name, rationale, effect_metric = result
            st.divider()
            st.success(f"**Recommended Test:** {test_name}")
            st.info(f"**Rationale:** {rationale}")
            st.markdown(f"**Effect Size Metric:** `{effect_metric}`")

            if covariate:
                st.warning("**With covariates →** Upgrade to ANCOVA (continuous DV) or multiple/logistic regression. Include covariates as predictors.")
            if multilevel:
                st.warning("**Clustered/nested data →** Use Multilevel Modelling (HLM/LMM) or GEE to account for non-independence.")
        else:
            st.info("No exact match found. Adjust your selections or consult a statistician for complex designs.")

        # Reference table
        with st.expander("Quick Reference: All Study Types & Tests"):
            ref_data = {
                "Study Type": ["RCT", "RCT", "Cross-Sectional", "Cohort", "Case-Control",
                               "Correlational", "Regression", "Longitudinal", "Diagnostic", "Meta-Analysis"],
                "Outcome Type": ["Continuous", "Binary", "Binary/Cat.", "Binary/Survival",
                                 "Binary", "Continuous/Ordinal", "Any", "Repeated", "Binary", "Any"],
                "Primary Test": ["T-test / ANOVA / Mann-Whitney", "Chi-Square / Fisher's",
                                 "Chi-Square / Logistic Reg.", "RR / Cox Regression",
                                 "OR / Logistic Reg.", "Pearson r / Spearman ρ",
                                 "Linear / Logistic Reg.", "LMM / GEE",
                                 "ROC-AUC / Sensitivity-Specificity", "Fixed/Random Effects"],
                "Effect Size": ["Cohen's d / η²", "RR / NNT", "OR / Cramér's V", "HR / RR",
                                "OR", "r / ρ", "R² / OR", "Cohen's d / OR", "AUC / Youden's J",
                                "SMD / I²"]
            }
            st.dataframe(pd.DataFrame(ref_data), width='stretch', hide_index=True)

    # ── Tab 3: Effect Size Calculator ────────────────────────────────────────
    with tab3:
        st.subheader("Effect Size Calculator")
        eff_type = st.radio("Calculate:", ["Cohen's d (means)", "Odds Ratio (2×2 table)",
                                            "Pearson r → d", "NNT from RCT"], horizontal=True)

        if eff_type == "Cohen's d (means)":
            e1, e2 = st.columns(2)
            with e1:
                m1 = st.number_input("Group 1 Mean", value=10.0)
                s1 = st.number_input("Group 1 SD",   value=2.0,  min_value=0.01)
                n1 = st.number_input("Group 1 n",    value=30,   min_value=2, step=1)
            with e2:
                m2 = st.number_input("Group 2 Mean", value=12.0)
                s2 = st.number_input("Group 2 SD",   value=2.5,  min_value=0.01)
                n2 = st.number_input("Group 2 n",    value=30,   min_value=2, step=1)
            pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            d = abs(m1 - m2) / pooled_sd
            label = "Small" if d < 0.5 else ("Medium" if d < 0.8 else "Large")
            st.metric("Cohen's d", f"{d:.3f}", delta=f"{label} effect")
            st.caption("Interpretation: d < 0.2 Negligible | 0.2–0.5 Small | 0.5–0.8 Medium | > 0.8 Large")

        elif eff_type == "Odds Ratio (2×2 table)":
            st.markdown("Enter cell counts from your 2×2 contingency table:")
            t1, t2 = st.columns(2)
            with t1:
                a = st.number_input("Exposed + Outcome (a)", value=40, min_value=0, step=1)
                c = st.number_input("Unexposed + Outcome (c)", value=20, min_value=0, step=1)
            with t2:
                b = st.number_input("Exposed + No Outcome (b)", value=60, min_value=0, step=1)
                d_val = st.number_input("Unexposed + No Outcome (d)", value=80, min_value=0, step=1)
            if b > 0 and c > 0:
                OR = (a * d_val) / (b * c)
                log_OR = np.log(OR)
                se_log = np.sqrt(1/max(a,1) + 1/max(b,1) + 1/max(c,1) + 1/max(d_val,1))
                ci_lo  = np.exp(log_OR - 1.96 * se_log)
                ci_hi  = np.exp(log_OR + 1.96 * se_log)
                st.metric("Odds Ratio", f"{OR:.3f}")
                st.info(f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
                st.caption("OR > 1 = increased odds in exposed group; CI not crossing 1 = significant")
            else:
                st.warning("Cell counts b and c must be > 0.")

        elif eff_type == "Pearson r → d":
            r = st.slider("Pearson r", -1.0, 1.0, 0.30, step=0.01)
            if abs(r) < 1:
                d_from_r = (2 * r) / np.sqrt(1 - r**2)
                label = "Small" if abs(d_from_r) < 0.5 else ("Medium" if abs(d_from_r) < 0.8 else "Large")
                st.metric("Equivalent Cohen's d", f"{d_from_r:.3f}", delta=label)
                st.caption("r interpretation: |r| < 0.1 Negligible | 0.1–0.3 Small | 0.3–0.5 Medium | > 0.5 Large")

        elif eff_type == "NNT from RCT":
            st.markdown("Enter event rates for treatment and control groups:")
            cer = st.slider("Control Event Rate (CER)", 0.01, 1.0, 0.40, step=0.01)
            eer = st.slider("Experimental Event Rate (EER)", 0.01, 1.0, 0.25, step=0.01)
            arr = cer - eer
            rr  = eer / cer if cer > 0 else 0
            rri = (cer - eer) / cer if cer > 0 else 0
            nnt = 1 / arr if arr != 0 else float('inf')
            c1e, c2e, c3e, c4e = st.columns(4)
            c1e.metric("ARR", f"{arr:.3f}")
            c2e.metric("RR",  f"{rr:.3f}")
            c3e.metric("RRI", f"{rri:.1%}")
            c4e.metric("NNT", f"{nnt:.1f}" if nnt != float('inf') else "∞")
            st.caption("NNT = Number Needed to Treat. Lower NNT = more effective intervention.")

    # ── Tab 4: Outcomes Analysis Guide ──────────────────────────────────────
    with tab4:
        st.subheader("Outcomes Analysis Reference Guide")

        out_section = st.selectbox("Select Analysis Area", [
            "Reliability & Internal Consistency",
            "Validity Assessment",
            "Correlation Interpretation",
            "Regression Diagnostics",
            "Survival Analysis",
            "Diagnostic Test Metrics",
            "Meta-Analysis Heterogeneity"
        ])

        guides = {
            "Reliability & Internal Consistency": {
                "tool": "Cronbach's Alpha (α) / McDonald's Omega (ω)",
                "table": {
                    "α Value": ["≥ 0.90", "0.80–0.89", "0.70–0.79", "0.60–0.69", "< 0.60"],
                    "Interpretation": ["Excellent", "Good", "Acceptable", "Questionable", "Unacceptable"],
                    "Action": ["Use as-is", "Use as-is", "Use with caution", "Revise items", "Do not use"]
                },
                "note": "Also report ICC (Intraclass Correlation) for inter-rater reliability. ICC > 0.75 = good; > 0.90 = excellent."
            },
            "Validity Assessment": {
                "tool": "Construct / Content / Criterion Validity",
                "table": {
                    "Validity Type": ["Content", "Construct (Convergent)", "Construct (Discriminant)",
                                     "Criterion (Concurrent)", "Criterion (Predictive)"],
                    "Method": ["Expert panel / CVI ≥ 0.80", "AVE ≥ 0.50 / Factor loading ≥ 0.40",
                               "HTMT < 0.85 / Fornell-Larcker", "Correlation with gold standard",
                               "Regression on future outcome"],
                    "Threshold": ["CVI ≥ 0.80", "AVE ≥ 0.50", "HTMT < 0.85", "r ≥ 0.70", "β significant"]
                },
                "note": "For scale validation, run CFA (Confirmatory Factor Analysis). Report CFI ≥ 0.95, RMSEA ≤ 0.06, SRMR ≤ 0.08."
            },
            "Correlation Interpretation": {
                "tool": "Pearson r / Spearman ρ",
                "table": {
                    "|r| Range": ["0.00–0.09", "0.10–0.29", "0.30–0.49", "0.50–0.69", "0.70–0.89", "0.90–1.00"],
                    "Strength": ["Negligible", "Weak", "Moderate", "Strong", "Very Strong", "Nearly Perfect"],
                    "Cohen Label": ["—", "Small", "Medium", "—", "Large", "—"]
                },
                "note": "Always report r and p-value together. Large samples make even negligible r values statistically significant."
            },
            "Regression Diagnostics": {
                "tool": "Linear / Logistic Regression Assumptions",
                "table": {
                    "Assumption": ["Linearity", "Independence", "Homoscedasticity", "Normality of residuals",
                                  "No multicollinearity", "No influential outliers"],
                    "How to Check": ["Residuals vs Fitted plot", "Durbin-Watson (1.5–2.5)",
                                     "Scale-Location plot", "Q-Q plot / Shapiro-Wilk",
                                     "VIF < 5 (strict: < 3)", "Cook's D < 4/n"],
                    "Fix if Violated": ["Transform DV / add polynomial", "Use clustered SE",
                                        "Transform DV / robust SE", "Bootstrap CI / transform",
                                        "Remove/combine predictors", "Winsorize / remove outlier"]
                },
                "note": "For logistic regression: report Nagelkerke R², Hosmer-Lemeshow goodness-of-fit, and classification accuracy."
            },
            "Survival Analysis": {
                "tool": "Kaplan-Meier / Cox Proportional Hazards",
                "table": {
                    "Metric": ["Median Survival", "Log-Rank Test", "Hazard Ratio (HR)", "Proportional Hazards Assumption",
                               "Cumulative Incidence"],
                    "Description": ["Time at which 50% of subjects have experienced the event",
                                    "Compares survival curves between groups (p < 0.05 = significant difference)",
                                    "HR > 1 = increased hazard; HR < 1 = protective",
                                    "Test with Schoenfeld residuals; p > 0.05 = assumption met",
                                    "Use competing risks model if multiple event types"],
                    "Report": ["Median + 95% CI", "χ² + p-value", "HR + 95% CI", "p-value per covariate", "CIF curve"]
                },
                "note": "Always report number at risk below KM curves and censor times."
            },
            "Diagnostic Test Metrics": {
                "tool": "Sensitivity, Specificity, Predictive Values, ROC-AUC",
                "table": {
                    "Metric": ["Sensitivity (Recall)", "Specificity", "PPV", "NPV", "LR+", "LR−", "AUC"],
                    "Formula": ["TP/(TP+FN)", "TN/(TN+FP)", "TP/(TP+FP)", "TN/(TN+FN)",
                                "Sensitivity/(1−Specificity)", "(1−Sensitivity)/Specificity", "AUROC"],
                    "Target": ["> 0.80 for screening", "> 0.80 for confirmation", "High = few false positives",
                               "High = few false negatives", "> 10 = strong positive", "< 0.1 = strong negative",
                               "> 0.90 excellent | 0.70–0.90 acceptable"]
                },
                "note": "Youden's Index (J = Sensitivity + Specificity − 1) identifies the optimal cut-off on the ROC curve."
            },
            "Meta-Analysis Heterogeneity": {
                "tool": "I², Cochran's Q, τ², Prediction Interval",
                "table": {
                    "Statistic": ["I²", "I²", "I²", "Cochran's Q", "τ²", "Prediction Interval"],
                    "Value": ["< 25%", "25%–75%", "> 75%", "p < 0.10", "Near 0", "Wide PI"],
                    "Interpretation": ["Low heterogeneity", "Moderate heterogeneity", "High heterogeneity",
                                       "Significant heterogeneity", "Homogeneous studies",
                                       "True effects vary substantially across populations"]
                },
                "note": "Use Random Effects model when I² > 50%. Investigate heterogeneity with subgroup analysis and meta-regression."
            }
        }

        selected_guide = guides[out_section]
        st.markdown(f"**Primary Tool:** {selected_guide['tool']}")
        st.dataframe(pd.DataFrame(selected_guide['table']), width='stretch', hide_index=True)
        st.info(selected_guide['note'])

# --- Module 4: Data Viz & Tables ---
elif module == "Data Viz & Tables":
    st.header("Statistical Tables & Figures")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Descriptive Statistics")
        st.table(df.describe())
        
        st.write("### Distribution Plot")
        target_col = st.selectbox("Select Column", df.columns)
        st.bar_chart(df[target_col].value_counts())

# --- Module 5: Manuscript Drafter ---
elif module == "Manuscript Drafter":
    st.header("Draft Manuscript Tool")

    ms_tab1, ms_tab2, ms_tab3 = st.tabs([
        "Section-by-Section Drafter",
        "Literature Argument Builder",
        "Full Manuscript Preview"
    ])

    # ── helpers stored in session state ─────────────────────────────────────
    for key in ['ms_title','ms_background','ms_problem','ms_objectives','ms_hypothesis',
                'ms_design','ms_population','ms_sampling','ms_instrument','ms_procedure',
                'ms_analysis','ms_results','ms_discussion','ms_conclusion','ms_limitations',
                'ms_recommendations']:
        if key not in st.session_state:
            st.session_state[key] = ''

    # ── Tab 1: Section-by-Section Drafter ───────────────────────────────────
    with ms_tab1:
        st.subheader("Guided Section Writing")

        sec = st.selectbox("Jump to Section", [
            "Title & Abstract",
            "1 · Introduction / Background",
            "2 · Statement of the Problem",
            "3 · Objectives & Hypotheses",
            "4 · Review of Related Literature",
            "5 · Methodology",
            "6 · Results",
            "7 · Discussion",
            "8 · Conclusion, Limitations & Recommendations"
        ])

        # ── Title & Abstract ──────────────────────────────────────────────
        if sec == "Title & Abstract":
            st.markdown("**Tips:** A good title names the variables, population, and setting. "
                        "The abstract should cover Background, Objective, Methods, Results (if available), and Conclusion (250 words max).")
            st.session_state.ms_title = st.text_input(
                "Manuscript Title",
                value=st.session_state.ms_title,
                placeholder="e.g., Effect of X on Y among Z: A Cross-Sectional Study")

        # ── Introduction ─────────────────────────────────────────────────
        elif sec == "1 · Introduction / Background":
            st.markdown("""**Writing Guide — Introduction (funnel structure):**
1. **Global/global context** — broad scope of the topic (2–3 sentences)
2. **Narrowing to local context** — prevalence, burden, or relevance in your setting
3. **What is known** — brief summary of existing evidence
4. **What is not known / gap** — transition to the problem statement
5. **Significance** — why this study matters now""")
            st.session_state.ms_background = st.text_area(
                "Write your Introduction here",
                value=st.session_state.ms_background,
                height=220,
                placeholder="Start broad: 'BPH affects approximately 50% of men aged 51–60 worldwide...'")
            with st.expander("Need a starter scaffold?"):
                topic_seed = st.text_input("Enter your topic for a scaffold", key="intro_seed")
                if st.button("Generate Scaffold", key="gen_intro"):
                    if topic_seed:
                        scaffold = (
                            f"{topic_seed} represents a significant public health concern with widespread "
                            f"implications for affected populations globally. Despite increasing attention in "
                            f"the literature, gaps remain in understanding its full scope within specific "
                            f"contexts. Existing studies have largely focused on [X], leaving [Y] underexplored. "
                            f"This study therefore aims to address this gap by examining [Z]."
                        )
                        st.code(scaffold, language=None)
                        st.caption("Copy this scaffold into the text area above and refine it.")

        # ── Problem Statement ─────────────────────────────────────────────
        elif sec == "2 · Statement of the Problem":
            st.markdown("""**Writing Guide — Statement of the Problem:**
- State what the problem is and why it is a problem
- Cite evidence of the gap (statistics, prior studies)
- End with a clear statement of what this study investigates
- Typical length: 1–2 paragraphs""")
            st.session_state.ms_problem = st.text_area(
                "Statement of the Problem",
                value=st.session_state.ms_problem,
                height=160,
                placeholder="e.g., Despite the prevalence of BPH, few studies have examined...")
            with st.expander("Problem Statement Template"):
                st.markdown("""
> *[Topic] poses a [severity] challenge to [population/setting]. While [existing knowledge], 
> [gap in knowledge] remains poorly understood. In [context/country], [specific evidence of problem]. 
> This study therefore sought to [purpose of study].*
""")

        # ── Objectives & Hypotheses ───────────────────────────────────────
        elif sec == "3 · Objectives & Hypotheses":
            st.markdown("""**Writing Guide:**
- **General objective**: One sentence — what the study aims to achieve overall
- **Specific objectives**: 3–5 SMART objectives using action verbs (determine, assess, compare, identify)
- **Hypotheses**: State H₀ and H₁ per key relationship tested""")
            st.session_state.ms_objectives = st.text_area(
                "Objectives",
                value=st.session_state.ms_objectives,
                height=150,
                placeholder="General Objective:\nSpecific Objectives:\n  1.\n  2.\n  3.")
            st.session_state.ms_hypothesis = st.text_area(
                "Hypotheses",
                value=st.session_state.ms_hypothesis,
                height=100,
                placeholder="H₀: There is no significant...\nH₁: There is a significant...")

        # ── Literature Review ─────────────────────────────────────────────
        elif sec == "4 · Review of Related Literature":
            st.markdown("""**Writing Guide — Literature Review:**
- Organize thematically (not chronologically)
- Each paragraph = one theme/concept; open with a topic sentence
- Synthesize across sources (avoid stringing summaries)
- End each theme with a gap or transition statement
- Close the review with a conceptual/theoretical framework""")
            st.info("Use the **Literature Argument Builder** tab to auto-draft arguments from your saved library.")

        # ── Methodology ───────────────────────────────────────────────────
        elif sec == "5 · Methodology":
            st.markdown("**Each sub-section should be detailed enough for replication.**")
            cols_m = st.columns(2)
            with cols_m[0]:
                st.session_state.ms_design = st.text_area(
                    "Research Design",
                    value=st.session_state.ms_design,
                    height=80,
                    placeholder="e.g., A descriptive cross-sectional design was employed...")
                st.session_state.ms_population = st.text_area(
                    "Population & Setting",
                    value=st.session_state.ms_population,
                    height=80,
                    placeholder="e.g., The study population comprised all adult patients...")
                st.session_state.ms_sampling = st.text_area(
                    "Sampling Technique & Sample Size",
                    value=st.session_state.ms_sampling,
                    height=80,
                    placeholder="e.g., A stratified random sampling technique was used...")
            with cols_m[1]:
                st.session_state.ms_instrument = st.text_area(
                    "Instrument / Data Collection",
                    value=st.session_state.ms_instrument,
                    height=80,
                    placeholder="e.g., A structured questionnaire adapted from [Author, Year]...")
                st.session_state.ms_procedure = st.text_area(
                    "Procedure",
                    value=st.session_state.ms_procedure,
                    height=80,
                    placeholder="e.g., After obtaining ethical clearance, data were collected over...")
                st.session_state.ms_analysis = st.text_area(
                    "Data Analysis",
                    value=st.session_state.ms_analysis,
                    height=80,
                    placeholder="e.g., Data were analyzed using SPSS v.26. Descriptive statistics...")
            with st.expander("Methodology Sentence Starters"):
                st.markdown("""
| Sub-section | Starter Phrases |
|---|---|
| Design | *"A [design] was employed to [purpose]..."* |
| Population | *"The target population consisted of all [group] in [setting] during [period]..."* |
| Sampling | *"Using [technique], a sample of [n] participants was selected because [justification]..."* |
| Instrument | *"Data were collected using a [type] questionnaire comprising [n] items measuring [constructs]..."* |
| Validity | *"Content validity was established through expert review (CVI = [value])..."* |
| Reliability | *"Internal consistency was confirmed with Cronbach's alpha (α = [value])..."* |
| Analysis | *"Inferential statistics ([test]) were used to test the hypotheses at α = 0.05 significance level..."* |
| Ethics | *"Ethical approval was obtained from [committee]. Participation was voluntary and confidential..."* |
""")

        # ── Results ───────────────────────────────────────────────────────
        elif sec == "6 · Results":
            st.markdown("""**Writing Guide — Results:**
- Present findings in logical order (demographic profile → descriptive → inferential)
- Every table/figure must be referenced in text
- Report test statistics fully: *F*(df1, df2) = value, *p* = value, effect size
- Do not interpret in Results — save for Discussion""")
            st.session_state.ms_results = st.text_area(
                "Results",
                value=st.session_state.ms_results,
                height=200,
                placeholder="Table 1 presents the demographic profile of respondents...")
            with st.expander("APA Reporting Templates"):
                st.markdown("""
- **T-test:** *t*([df]) = [value], *p* = [.xxx], Cohen's *d* = [value]
- **ANOVA:** *F*([df1], [df2]) = [value], *p* = [.xxx], η² = [value]
- **Chi-square:** χ²([df], *N* = [n]) = [value], *p* = [.xxx], Cramér's *V* = [value]
- **Correlation:** *r*([df]) = [value], *p* = [.xxx]
- **Regression:** *β* = [value], *t* = [value], *p* = [.xxx], *R*² = [value]
- **Odds Ratio:** OR = [value], 95% CI [lo, hi], *p* = [.xxx]
""")

        # ── Discussion ────────────────────────────────────────────────────
        elif sec == "7 · Discussion":
            st.markdown("""**Writing Guide — Discussion (inverted funnel):**
1. Restate key findings (without raw numbers)
2. Interpret each finding — what does it mean?
3. Compare & contrast with prior literature (agree/disagree + cite reasons)
4. Explain unexpected or contradictory findings
5. Theoretical/practical implications""")
            st.session_state.ms_discussion = st.text_area(
                "Discussion",
                value=st.session_state.ms_discussion,
                height=220,
                placeholder="The findings revealed that... This is consistent with [Author, Year] who found that...")
            with st.expander("Discussion Sentence Starters"):
                st.markdown("""
- *"The findings of this study revealed that [result], which is consistent with [Author, Year]..."*
- *"Contrary to [Author, Year], the present study found that..."*
- *"This finding may be explained by [mechanism/theory]..."*
- *"The observed [result] has practical implications for [stakeholders], suggesting that..."*
- *"A plausible explanation for the discrepancy could be [factor], as [Author, Year] noted..."*
""")

        # ── Conclusion ────────────────────────────────────────────────────
        elif sec == "8 · Conclusion, Limitations & Recommendations":
            st.markdown("""**Writing Guide:**
- **Conclusion**: Summarize what was found and its significance (no new data)
- **Limitations**: Be honest; briefly explain how they were mitigated
- **Recommendations**: Separate for practice, policy, and future research""")
            cols_c = st.columns(3)
            with cols_c[0]:
                st.session_state.ms_conclusion = st.text_area(
                    "Conclusion",
                    value=st.session_state.ms_conclusion, height=160,
                    placeholder="In conclusion, this study established that...")
            with cols_c[1]:
                st.session_state.ms_limitations = st.text_area(
                    "Limitations",
                    value=st.session_state.ms_limitations, height=160,
                    placeholder="This study was limited by its cross-sectional design...")
            with cols_c[2]:
                st.session_state.ms_recommendations = st.text_area(
                    "Recommendations",
                    value=st.session_state.ms_recommendations, height=160,
                    placeholder="Based on the findings, it is recommended that...")

            st.divider()
            st.markdown("#### Evidence-Based Conclusion Scaffold Generator")
            st.caption(
                "Generates a conclusion grounded in your results and cited literature. "
                "Requires saved papers in your library and content in the Results section."
            )
            saved_for_conc = st.session_state.get('saved_papers', [])
            results_text   = st.session_state.get('ms_results', '').strip()
            rq_text        = st.session_state.get('cq_q_selected', '').strip()
            cite_style     = st.session_state.get('citation_style', 'APA 7th')

            if st.button("Generate Evidence-Based Conclusion Scaffold"):
                if not results_text:
                    st.warning("Please write your Results section first (Section 6).")
                else:
                    # Build citation string from top 3 most-cited saved papers
                    sorted_papers = sorted(
                        saved_for_conc,
                        key=lambda x: int(x.get('Citations', 0) or 0),
                        reverse=True
                    )
                    top_papers = sorted_papers[:3]
                    cite_tags = [in_text_cite(p, style=cite_style, number=i+1)
                                 for i, p in enumerate(top_papers)]
                    cite_str = "; ".join(cite_tags) if cite_tags else "[Author, Year]"

                    rq_clause = f"the research question — *{rq_text}* —" if rq_text else "the stated research objectives"

                    scaffold_conclusion = (
                        f"In conclusion, this study addressed {rq_clause} and generated findings with "
                        f"substantive theoretical and practical significance. "
                        f"The results demonstrated that [restate key finding from results], "
                        f"a conclusion corroborated by prior evidence in the literature {cite_str}. "
                        f"Specifically, the observed [finding] is consistent with the proposition that [theoretical link], "
                        f"and extends existing knowledge by [novel contribution of this study]. "
                        f"These findings carry implications for [practitioners/policy-makers/educators] in [context], "
                        f"suggesting that [actionable implication]. "
                        f"Overall, this study contributes to the evidence base on [topic] and reinforces "
                        f"the need for [continued attention/further investigation] in this domain."
                    )

                    st.info(scaffold_conclusion)
                    if st.button("Use as Conclusion Draft"):
                        st.session_state.ms_conclusion = scaffold_conclusion
                        st.success("Conclusion populated. Scroll up to edit it.")

    # ── Tab 2: Literature Argument Builder ──────────────────────────────────
    with ms_tab2:
        st.subheader("Build Arguments from Reviewed Literature")

        saved = st.session_state.get('saved_papers', [])
        if not saved:
            st.info("No papers in your library yet. Go to **Literature Search**, search for papers, "
                    "and click **Save to Library** on relevant results.")
        else:
            st.success(f"{len(saved)} paper(s) in your library available for argument building.")
            df_lib = pd.DataFrame(saved)
            st.dataframe(df_lib[['Title','Authors','Year','Venue','Citations']],
                         width='stretch', hide_index=True)

            # Evidence quality overview
            with st.expander("Evidence Quality Summary"):
                for p in saved:
                    badge = evidence_badge(p.get('Citations', 0))
                    st.markdown(
                        f"- **{p['Title'][:80]}** ({p.get('Year','n.d.')}) — "
                        f"Citations: {p.get('Citations',0)} — *{badge}*"
                    )

            st.divider()

            st.markdown("#### Argument Composer")
            theme = st.text_input("Thematic Argument / Claim",
                                  placeholder="e.g., Medical management is the first-line treatment for BPH")
            stance = st.radio("Argument Stance", ["Supportive", "Contradictory", "Mixed / Nuanced"],
                              horizontal=True)
            selected_titles = st.multiselect(
                "Select papers to cite for this argument",
                options=[p['Title'] for p in saved]
            )

            # Show abstract previews for selected papers
            if selected_titles:
                with st.expander("Abstract previews of selected papers"):
                    for p in saved:
                        if p['Title'] in selected_titles:
                            st.markdown(f"**{p['Title']}** ({p.get('Year','n.d.')})")
                            abst = p.get('Abstract', '')
                            st.caption(abst[:400] + "…" if len(abst) > 400 else abst or "*No abstract saved.*")
                            st.divider()

            if st.button("Draft Argument Paragraph") and theme and selected_titles:
                cite_style = st.session_state.get('citation_style', 'APA 7th')
                cited = [p for p in saved if p['Title'] in selected_titles]
                cite_tags = [in_text_cite(p, style=cite_style, number=i+1)
                             for i, p in enumerate(cited)]
                cite_str = "; ".join(cite_tags)

                # Extract key evidence snippets from abstracts
                evidence_snippets = []
                for p in cited:
                    abst = p.get('Abstract', '')
                    if abst:
                        # Use the first ~120 chars of abstract as a concrete evidence anchor
                        snippet = abst[:120].rstrip() + "…"
                        evidence_snippets.append(
                            f"{in_text_cite(p, style=cite_style, number=cited.index(p)+1)} noted that \"{snippet}\""
                        )

                evidence_block = (" ".join(evidence_snippets[:2])
                                  if evidence_snippets
                                  else f"Evidence from the literature {cite_str} supports this claim.")

                stance_openers = {
                    "Supportive": (
                        f"{theme}. A growing body of evidence supports this position. "
                        f"{evidence_block} "
                        f"Collectively, {cite_str} demonstrate consistent findings regarding this phenomenon. "
                        f"These converging results strengthen the evidence base and suggest that "
                        f"[specific implication for practice or policy]."
                    ),
                    "Contradictory": (
                        f"While {theme.lower()} has been proposed, the evidence presents a more complex picture. "
                        f"{evidence_block} "
                        f"{cite_str} challenge this assertion by demonstrating [contradictory finding]. "
                        f"This discrepancy may be attributable to [methodological difference / contextual factor], "
                        f"underscoring the need for [future research direction]."
                    ),
                    "Mixed / Nuanced": (
                        f"The literature presents mixed evidence regarding {theme.lower()}. "
                        f"On one hand, {cite_tags[0] if cite_tags else '[Author]'} reported [supporting finding]. "
                        f"On the other hand, {cite_tags[-1] if len(cite_tags) > 1 else '[Author]'} "
                        f"reported [contradictory finding]. "
                        f"{evidence_block} "
                        f"This inconsistency may reflect differences in [sample, setting, or measurement], "
                        f"suggesting that [nuanced, evidence-informed conclusion]."
                    )
                }

                st.divider()
                st.markdown("**Drafted Argument Paragraph** *(edit placeholders in [ ] before using)*")
                draft_para = stance_openers[stance]
                st.text_area("", value=draft_para, height=180, key="arg_out")

                # Append to discussion
                if st.button("Append to Discussion Section"):
                    st.session_state.ms_discussion += f"\n\n{draft_para}"
                    st.success("Appended to Discussion. Go to Section 7 to edit.")

            st.divider()
            st.markdown("#### Synthesis Table")
            st.caption("Auto-generated from your library — use as a reference for your literature review.")
            synth_rows = []
            for p in saved:
                synth_rows.append({
                    "Author(s) & Year": f"{p.get('Authors','N/A')} ({p.get('Year','n.d.')})",
                    "Title": p.get('Title',''),
                    "Venue": p.get('Venue','N/A'),
                    "Citations": p.get('Citations', 0),
                    "Key Theme / Notes": ""
                })
            synth_df = pd.DataFrame(synth_rows)
            edited_synth = st.data_editor(synth_df, width='stretch', hide_index=True,
                                          column_config={"Key Theme / Notes": st.column_config.TextColumn(width="large")})
            st.download_button(
                "Export Synthesis Table (CSV)",
                data=edited_synth.to_csv(index=False),
                file_name="synthesis_table.csv",
                mime="text/csv"
            )

    # ── Tab 3: Full Manuscript Preview & Download ────────────────────────────
    with ms_tab3:
        st.subheader("Full Manuscript Preview")

        ms = st.session_state
        sections = {
            "Introduction / Background":           ms.ms_background,
            "Statement of the Problem":            ms.ms_problem,
            "Objectives & Hypotheses":             ms.ms_objectives + ("\n\n" + ms.ms_hypothesis if ms.ms_hypothesis else ""),
            "Methodology — Design":                ms.ms_design,
            "Methodology — Population & Setting":  ms.ms_population,
            "Methodology — Sampling":              ms.ms_sampling,
            "Methodology — Instrument":            ms.ms_instrument,
            "Methodology — Procedure":             ms.ms_procedure,
            "Methodology — Data Analysis":         ms.ms_analysis,
            "Results":                             ms.ms_results,
            "Discussion":                          ms.ms_discussion,
            "Conclusion":                          ms.ms_conclusion,
            "Limitations":                         ms.ms_limitations,
            "Recommendations":                     ms.ms_recommendations,
        }

        filled   = {k: v for k, v in sections.items() if v.strip()}
        missing  = [k for k, v in sections.items() if not v.strip()]
        progress = len(filled) / len(sections)

        st.progress(progress, text=f"Manuscript completion: {int(progress*100)}%  ({len(filled)}/{len(sections)} sections)")
        if missing:
            with st.expander(f"{len(missing)} section(s) not yet written"):
                for m in missing:
                    st.markdown(f"- {m}")

        if ms.ms_title.strip():
            st.markdown(f"# {ms.ms_title}")

        full_md = f"# {ms.ms_title}\n\n" if ms.ms_title.strip() else ""
        for heading, content in sections.items():
            if content.strip():
                st.markdown(f"## {heading}")
                st.write(content)
                full_md += f"## {heading}\n\n{content}\n\n"

        # ── Auto-generated References section ────────────────────────────
        ref_papers = st.session_state.get('saved_papers', [])
        cite_style = st.session_state.get('citation_style', 'APA 7th')
        if ref_papers:
            st.markdown("---")
            st.markdown("## References")
            ref_md = "## References\n\n"
            # Sort alphabetically by first author (APA/Harvard) or leave for Vancouver numbered
            if cite_style == 'Vancouver':
                sorted_refs = ref_papers  # keep insertion order for numbering
            else:
                sorted_refs = sorted(
                    ref_papers,
                    key=lambda p: p.get('Authors', 'ZZZ').split(',')[0].strip().lower()
                )
            for i, p in enumerate(sorted_refs, start=1):
                ref_str = format_reference(p, style=cite_style, number=i)
                st.markdown(f"{ref_str}")
                ref_md += f"{ref_str}\n\n"
            full_md += ref_md
        else:
            st.info("No references yet — save papers from the Literature Search module to auto-build the reference list.")

        st.divider()
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download Full Manuscript (.md)",
                data=full_md,
                file_name=f"{ms.ms_title.replace(' ','_') or 'manuscript'}.md",
                mime="text/markdown",
                disabled=not full_md.strip()
            )
        with dl2:
            # Plain text version
            plain = full_md.replace("# ", "").replace("## ", "\n").replace("**", "")
            st.download_button(
                "Download as Plain Text (.txt)",
                data=plain,
                file_name=f"{ms.ms_title.replace(' ','_') or 'manuscript'}.txt",
                mime="text/plain",
                disabled=not plain.strip()
            )
