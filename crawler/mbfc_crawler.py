import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

base_url = 'https://mediabiasfactcheck.com/tag/daily-fact-check/page/{}/'

start_page = 21
end_page = 27

all_article_links = []

for page in range(start_page, end_page + 1):
    print(f'Processing page {page}...')
    
    url = base_url.format(page)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    for item in soup.find_all('div', class_='content-thumb'):
        link = item.find('a', href=True)
        if link:
            all_article_links.append(link['href'])

print(f'Total articles collected: {len(all_article_links)}')

data = []
explanation = []

for article_url in all_article_links:
    print(f'\nProcessing article: {article_url}')
    
    article_resp = requests.get(article_url)
    article_soup = BeautifulSoup(article_resp.text, 'html.parser')

    rows = article_soup.find_all('tr')
    
    date = article_soup.find('h1', class_='entry-title').text.strip() if article_soup.find('h1', class_='entry-title') else 'No date available'
    
    for index, row in enumerate(rows):
        print(index)
        columns = row.find_all('td')
        spans = columns[1].find_all('span')
        
        explanation = 'No Explanation available'
        
        for span in spans:
            rating = span.text.strip()
            if 'rating: ' in rating and rating.endswith(')'):
                explanation = rating
                break
                
        if spans:
            span_text = spans[0].text.strip()
            if ': ' in span_text: 
                post, claim = span_text.rsplit(': ', 1)
            else:
                post = span_text
                claim = 'No Claim available'
        else:
            post = 'No Post available'
            claim = 'No Claim available'            
        
        source = ', '.join(a['href'] for a in row.find_all('a', href=True)) if row else 'No Source available'
        classification = columns[0].text.replace('\n', ' ').replace('  ', ' ').strip() if columns[0] else 'No Classification available'

        data.append({
            'Post': post,
            'Date': date,
            'Claim': claim,
            'Explanation': explanation,
            'Source': source,
            'Classification': classification
        })

df = pd.DataFrame(data)

df['Date'] = df['Date'].apply(
    lambda x: re.search(r'\b\d{2}/\d{2}/\d{4}\b', x).group(0) if isinstance(x, str) and re.search(r'\b\d{2}/\d{2}/\d{4}\b', x) else ''
)

df['Post'], df['Claim'] = zip(*df.apply(
    lambda x: x['Post'].rsplit(': ', 1) if x['Claim'] == 'No Claim available' and ': ' in x['Post'] else (x['Post'], x['Claim']),
    axis=1
))

csv_filename = 'mbfc.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8')