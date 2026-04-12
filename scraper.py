"""
Uber Eats Help Center Scraper
==============================
Scrapes customer-facing help articles from help.uber.com/ubereats
and saves them to data/knowledge_base.json.

Usage:
    pip install requests beautifulsoup4
    python scraper.py

Note: The Uber help center uses client-side rendering for some pages.
This scraper fetches what's available via server-side HTML. For pages
that don't return content, it falls back to the existing knowledge base.

Run this periodically to refresh the knowledge base with the latest
Uber Eats policies and help articles.
"""

import json, os, time, re
import requests
from bs4 import BeautifulSoup

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.json")

# Known Uber Eats help article URLs (customer-facing)
ARTICLE_URLS = [
    # Order Issues
    ("Wrong or missing items", "Order Issues", "https://help.uber.com/en/ubereats/restaurants/article/wrong-or-missing-items?nodeId=598f96f1-2d4f-4143-90de-dc2fdbf505cf"),
    ("My order is wrong", "Order Issues", "https://help.uber.com/en/ubereats/restaurants/article/my-order-is-wrong?nodeId=93fe8ec6-1f78-4279-a574-177d122fda26"),
    ("Missing items", "Order Issues", "https://help.uber.com/en/ubereats/stores/article/missing-items?nodeId=63604d59-46cd-465e-9e7d-7ad3a5fbb238"),
    # Delivery
    ("Order never arrived", "Delivery Issues", "https://help.uber.com/en/ubereats/restaurants/article/order-never-arrived?nodeId=23e258f9-8040-49f1-a74f-0fdf9fef6cba"),
    ("My order never arrived", "Delivery Issues", "https://help.uber.com/en/ubereats/restaurants/article/my-order-never-arrived?nodeId=dd225ffc-ba29-4bba-a85e-a9d1feecc25e"),
    ("My order is taking longer than expected", "Delivery Issues", "https://help.uber.com/en/ubereats/restaurants/article/my-order-is-taking-longer-than-expected?nodeId=496e837c-e360-4f40-8efb-a07465381697"),
    ("Delayed order arrival", "Delivery Issues", "https://help.uber.com/en/ubereats/restaurants/article/delayed-order-arrival?nodeId=a733429a-624b-44b6-99b9-041a9928d089"),
    # Cancellation
    ("What is the Uber Eats cancellation policy", "Cancellation", "https://help.uber.com/en/ubereats/restaurants/article/what-is-the-uber-eats-cancellation-policy-?nodeId=78c4c326-15ba-4d6a-9fd2-d8724f525477"),
    ("I was charged for cancelling my order", "Cancellation", "https://help.uber.com/en/ubereats/restaurants/article/i-was-charged-for-cancelling-my-order?nodeId=92f2ec03-65de-4fb8-9c50-b8fb8ea6a8aa"),
    ("Cancel my order", "Cancellation", "https://help.uber.com/en/ubereats/restaurants/article/cancel-my-order?nodeId=476b770f-f9d9-4c63-ae00-12e3e8155e9e"),
    ("My order was canceled FAQ", "Cancellation", "https://help.uber.com/en/ubereats/restaurants/article/my-order-was-canceled-faq?nodeId=f1f17a0b-33bd-4e99-94ce-016de5bea7a9"),
    # Membership
    ("Uber One cancellation and refund", "Membership", "https://help.uber.com/en/ubereats/restaurants/article/uber-one-cancellation-and-refund?nodeId=2c21ad9e-2795-4b62-b3f8-95155ce9dd27"),
    ("Eats Pass cancellation and refund", "Membership", "https://help.uber.com/en/ubereats/restaurants/article/eats-pass-cancellation-and-refund?nodeId=84054a89-35d2-4794-a50e-d2f96690eef0"),
    # Promo
    ("How do I apply a promo code", "Promotions", "https://help.uber.com/en/ubereats/restaurants/article/how-do-i-apply-a-promo-code?nodeId=a9f04d28-534e-4898-ac4f-eb6c82398ff9"),
    ("How do promo codes work", "Promotions", "https://help.uber.com/ubereats/restaurants/article/how-do-promo-codes-work?nodeId=ad0d2531-ada6-4aa7-9b01-a17187399571"),
    # Support
    ("How to contact support", "Support", "https://help.uber.com/en/ubereats/restaurants/article/how-to-contact-support?nodeId=7ad3ef19-012f-43e6-a711-113935b0cb7b"),
    ("Chat Support for Sign-In Issues", "Support", "https://help.uber.com/en/ubereats/restaurants/article/chat-support-for-sign-in-issues?nodeId=200c6fb2-94a8-47d2-b08f-c362b8450e46"),
    # General
    ("What is Uber Eats", "General", "https://help.uber.com/en/ubereats/restaurants/article/what-is-uber-eats-?nodeId=fbf73e2a-c21f-4a48-8333-c874ae195fd1"),
    ("How to place an order on Uber Eats", "Ordering", "https://help.uber.com/en/ubereats/restaurants/article/how-to-place-an-order-on-uber-eats/?nodeId=509d1b2f-087c-4dac-9e94-6ab248e87491"),
    # Refunds (merchant side, but has customer-relevant policy info)
    ("Managing refunds for missing or incorrect orders", "Refunds", "https://help.uber.com/en/merchants-and-restaurants/article/managing-refunds-for-missing-or-incorrect-orders-?nodeId=9aa57e9b-8bbf-4aa7-91d6-96ca77682dd2"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_article(url: str) -> str | None:
    """Fetch and extract text content from an Uber help article."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try different content selectors used by Uber's help center
        content = None
        for selector in [
            "div[data-testid='rich-text-content']",
            "div.article-content",
            "div.rich-text",
            "main article",
            "main",
        ]:
            el = soup.select_one(selector)
            if el and len(el.get_text(strip=True)) > 50:
                content = el.get_text(separator=" ", strip=True)
                break

        if not content:
            # Fallback: get all paragraph text from body
            paragraphs = soup.find_all("p")
            texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
            if texts:
                content = " ".join(texts)

        if content:
            # Clean up
            content = re.sub(r"\s+", " ", content).strip()
            # Truncate if too long
            if len(content) > 2000:
                content = content[:2000]
            return content

    except Exception as e:
        print(f"  Error fetching {url}: {e}")

    return None


def run_scraper():
    """Scrape all articles and save to knowledge_base.json."""
    print(f"Scraping {len(ARTICLE_URLS)} Uber Eats help articles...")

    # Load existing knowledge base as fallback
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for a in json.load(f):
                existing[a["title"]] = a

    articles = []
    scraped = 0
    fallback = 0

    for i, (title, category, url) in enumerate(ARTICLE_URLS):
        print(f"  [{i+1}/{len(ARTICLE_URLS)}] {title}...", end=" ")
        content = scrape_article(url)

        if content and len(content) > 50:
            articles.append({
                "id": f"doc_{i+1:03d}",
                "title": title,
                "category": category,
                "source_url": url,
                "content": content,
            })
            scraped += 1
            print(f"OK ({len(content)} chars)")
        elif title in existing:
            articles.append(existing[title])
            fallback += 1
            print("FALLBACK (used existing)")
        else:
            print("SKIPPED (no content)")

        time.sleep(1)  # Be respectful

    # Add any existing articles not in the scrape list
    scraped_titles = {a["title"] for a in articles}
    for title, article in existing.items():
        if title not in scraped_titles:
            articles.append(article)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(articles)} total articles")
    print(f"  Scraped fresh: {scraped}")
    print(f"  Fallback (existing): {fallback}")
    print(f"  Carried over: {len(articles) - scraped - fallback}")
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"\nNote: Delete data/chroma_db/ to rebuild the vector store with new content.")


if __name__ == "__main__":
    run_scraper()
