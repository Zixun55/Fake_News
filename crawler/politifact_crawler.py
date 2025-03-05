import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

base_url = 'https://www.politifact.com/factchecks/list/?page={}&ruling=true'

start_page = 1
end_page = 88

all_article_links = []

for page in range(start_page, end_page + 1):
    print(f'Processing page {page}...')
    
    url = base_url.format(page)
    
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    for item in soup.find_all('li', class_='o-listicle__item'):
        link = item.find('div', class_='m-statement__quote').find('a', href=True)
        if link:
            full_link = 'https://www.politifact.com' + link['href']
            all_article_links.append(full_link)

print(f'Total articles collected: {len(all_article_links)}')

data = []

for article_url in all_article_links:
    print(f'\nProcessing article: {article_url}')
    
    article_resp = requests.get(article_url)
    article_soup = BeautifulSoup(article_resp.text, 'html.parser')
    
    post = article_soup.find('a', class_='m-statement__name').text.strip() if article_soup.find('a', class_='m-statement__name') else 'No Post available'
    date = article_soup.find('div', class_='m-statement__desc').text.strip() if article_soup.find('div', class_='m-statement__desc') else 'No Date available'
    claim = article_soup.find('div', class_='m-statement__quote').text.strip() if article_soup.find('div', class_='m-statement__quote') else 'No Claim available'
    classification = article_soup.find('img', class_='c-image__thumb', height='196').get('alt', 'None') if article_soup.find('img', class_='c-image__thumb', height='196') else 'No Classification available'
    explanation = article_soup.find('div', class_='short-on-time').text.strip() if article_soup.find('div', class_='short-on-time') else 'No Explanation available'
    explanation = re.sub(r'\n+', '|', explanation).strip()
    source = ', '.join(a['href'] for a in article_soup.find(id='sources', class_='m-superbox').find_all('a', href=True)) if article_soup.find(id='sources', class_='m-superbox') else 'No Source available'
    source = re.sub(r'\n+', '\n', source).strip()
    
    data.append({
        'Post': post,
        'Date': date,
        'Claim': claim,
        'Explanation': explanation,
        'Source': source,
        'Classification': classification
    })

df = pd.DataFrame(data)

df = df.drop(df[df['Date'].str.startswith('dicho el')].index)
df['Date'] = df['Date'].apply(
    lambda x: re.search(r'\b\w+ \d{1,2}, \d{4}\b', x).group(0) if isinstance(x, str) and re.search(r'\b\w+ \d{1,2}, \d{4}\b', x) else ''
)

csv_filename = 'politifact_true.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8')