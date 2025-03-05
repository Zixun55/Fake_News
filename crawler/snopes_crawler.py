import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

base_url = 'https://www.snopes.com/fact-check/rating/mixture/?pagenum={}'

# true: 173
# false: 358
# mostly-true: 25
# mostly-false: 37
# mixture: 85(半對半錯)
start_page = 1
end_page = 85

all_article_links = []

for page in range(start_page, end_page + 1):
    print(f'Processing page {page}...')
    
    url = base_url.format(page)
    
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    for item in soup.find_all('div', class_='article_wrapper'):
        link = item.find('a', href=True)
        if link:
            all_article_links.append(link['href'])

print(f'Total articles collected: {len(all_article_links)}')

data = []

for article_url in all_article_links:
    print(f'\nProcessing article: {article_url}')
    
    article_resp = requests.get(article_url)
    article_soup = BeautifulSoup(article_resp.text, 'html.parser')
        
    date = article_soup.find('h3', class_='publish_date').text.strip() if article_soup.find('h3', class_='publish_date') else 'No Date available'
    claim_section = article_soup.find("div", class_="claim_cont")
    claim = claim_section.get_text(strip=True) if claim_section else "No Claim available"
    classification = article_soup.find('img', class_='lazy-image').get('alt', 'None') if article_soup.find('img', class_='lazy-image') else 'No Classification available'
    
    source_section = article_soup.find('div', id='sources_rows')
    if source_section:
        paragraphs = source_section.find_all('p')
        source_texts = [" ".join(p.stripped_strings) for p in paragraphs if p.get_text(strip=True)]
        if source_texts:
            source = " | ".join(source_texts)
        else:
            source = 'No Source available'
    else:
        source = 'No Source available'

    explanation_section = article_soup.find('article', id='article-content')
    if explanation_section:
        paragraphs = explanation_section.find_all('p')
        if len(paragraphs) > 1:
            explanation = " | ".join([re.sub(r'\s+', ' ', p.get_text()).strip() for p in paragraphs[1:]])  # 跳過第一個 <p>
        else:
            explanation = 'No Explanation available'
    else:
        explanation = 'No Explanation available'

    data.append({
        'Date': date,
        'Claim': claim,
        'Explanation': explanation,
        'Source': source,
        'Classification': classification
    })

df = pd.DataFrame(data)

df = df.drop(df[df['Date'].str.startswith('dicho el')].index)

df['Date'] = df['Date'].apply(
    lambda x: re.sub(r'^Published\s+', '', x).strip() if x.startswith('Published ') else x
)

csv_filename = 'snopes_mixture.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8')