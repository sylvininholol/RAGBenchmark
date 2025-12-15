import csv
from bs4 import BeautifulSoup
import bs4
import psycopg2

def get_all_headlines_ge(content):
    "Pega todas as headlines da página inicial do Globo Esporte (ge)"
    ge = BeautifulSoup(content, 'html.parser')
    news = ge.findAll('div', class_=lambda x: x and ('feed-post' in x and 'type-materia' in x))
    return news


def organize_ge_news(news: bs4.Tag) -> dict:
    """Transforma o html de UMA notícia em um dict com titulo e resumo"""

    splitted_news = {'titulo': None, 'resumo': None, 'link': None}

    link_tag = news.find('a', class_='feed-post-link')
    if not link_tag:
        link_tag = news.find('a', href=True)

    if link_tag:
        splitted_news['link'] = link_tag.get('href')
        
        titulo_tag = link_tag.find(['p', 'h2', 'h3'], class_=lambda x: x and ('feed-post-body-title-text' in x or 'post__title' in x or 'post-title' in x))
        if not titulo_tag:
            titulo_tag = link_tag.find(['p', 'h2', 'h3'], attrs={'elementtiming':'text-ssr'})
        if not titulo_tag:
            titulo_tag = link_tag.find(['p', 'h2', 'h3'])
            
        if titulo_tag:
            splitted_news['titulo'] = titulo_tag.get_text(strip=True)
        else:
            if link_tag.get_text(strip=True) and len(link_tag.get_text(strip=True)) > 10:
                 splitted_news['titulo'] = link_tag.get_text(strip=True)

    resumo_tag = news.select_one('div.feed-post-body-resumo p')
    if resumo_tag:
        splitted_news['resumo'] = resumo_tag.get_text(strip=True)
    else:
        resumo_tag = news.find('p', class_=lambda x: x and ('feed-post-body-text' in x or 'post__excerpt' in x))
        if resumo_tag:
            splitted_news['resumo'] = resumo_tag.get_text(strip=True)
        else:
            resumo_relacionados = news.select('a.bstn-relatedtext')
            if resumo_relacionados:
                textos = [item.get_text(strip=True) for item in resumo_relacionados]
                concatenado_resumo = ". ".join(textos)
                if concatenado_resumo:
                    concatenado_resumo += "."
                splitted_news['resumo'] = concatenado_resumo
    if not splitted_news['link'] or not splitted_news['titulo']:
        return {}

    return splitted_news


def save_headlines_csv(file, organized_news):
    if not organized_news:
        print(f"No news to save to {file}.")
        return

    all_keys = set()
    for news in organized_news:
        all_keys.update(news.keys())
    
    fieldnames = list(all_keys)

    with open(file, "w", newline="", encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for news_item in organized_news:
            w.writerow(news_item)


def get_complete_news(content, headline):
    complete_news = {**headline, "texto": None}
    news_soup = BeautifulSoup(content, 'html.parser')

    article_body_paragraphs = news_soup.select('div.content-text p, p.content-text__container')
    
    if not article_body_paragraphs:
        article_body_paragraphs = news_soup.select('article p, div.m-content-text p, div.text-container p')

    if article_body_paragraphs:
        texts = [item.get_text(strip=True) for item in article_body_paragraphs if item.get_text(strip=True)]
        concatenated_text = " ".join(texts)
        if concatenated_text:
            complete_news['texto'] = concatenated_text.strip()

    return complete_news


def save_headlines_pgsql(news_list, db_config):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    for news in news_list:
        cur.execute('''
            INSERT INTO scraping (titulo, texto, resumo, link)
            VALUES (%s, %s, %s, %s)
        ''', (
            news.get('titulo') or 'Sem Título',
            news.get('texto') or 'Sem Texto',
            news.get('resumo') or 'Sem Resumo',
            news.get('link') or 'Sem Link'
        ))
    conn.commit()
    cur.close()
    conn.close()