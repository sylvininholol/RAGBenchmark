import requests
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from webdriver_manager.chrome import ChromeDriverManager
from utils import get_all_headlines_ge, organize_ge_news, save_headlines_csv, get_complete_news, save_headlines_pgsql

load_dotenv()

service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()

# Descomenta isso se não quiser que abra o chrome (modo headless)
options.add_argument('--headless')

options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=service, options=options)

url = 'https://ge.globo.com/'

try:
    driver.get(url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "feed-post"))
    )

    more_content_to_load = True
    max_veja_mais_clicks = 25
    current_veja_mais_clicks = 0

    while more_content_to_load:
        last_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            print("Tentando clicar no botão 'Veja Mais'...")

            if current_veja_mais_clicks >= max_veja_mais_clicks:
                print(f"Atingido o limite de {max_veja_mais_clicks} cliques no 'Veja Mais'. Parando de carregar conteúdo.")
                more_content_to_load = False
                break

            try:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'load-more')]//a[text()='Veja mais']"))
                )
                load_more_button.click()
                print("Botão 'Veja mais' clicado com sucesso!")
                current_veja_mais_clicks += 1
                time.sleep(1)
            except Exception as e:
                print(f"Botão 'Veja mais' não encontrado ou não clicável, ou erro: {e}")
                more_content_to_load = False
        else:
            print(f"Página scrollada. Novo tamanho do scroll: {new_height}")

        if (not more_content_to_load and current_veja_mais_clicks >= max_veja_mais_clicks):
            break

    content = driver.page_source

except Exception as e:
    print(f"Erro ao acessar a página com Selenium: {e}")
    driver.quit()
    exit()

finally:
    driver.quit()

# pega todas as notícias da página inicial do globo esporte (ge)
ge_news = get_all_headlines_ge(content)
print(f"Encontrei {len(ge_news)} notícias na página inicial.")

organized_news: list[dict] = []
for news_item_html in ge_news:
    organized = organize_ge_news(news_item_html)
    if organized and organized.get('link') and organized.get('titulo'):
        organized_news.append(organized)

print(f"Organizei {len(organized_news)} notícias com link e título.")


complete_news: list[dict] = []
for i, news_item in enumerate(organized_news):
    url = news_item["link"]
    if not url:
        print(f"Pulando a notícia {i+1} por falta de link: {news_item.get('titulo')}")
        continue
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}, timeout=10)
        response.raise_for_status()
        complete_news_item = get_complete_news(response.content, news_item)
        complete_news.append(complete_news_item)
        print(f"Conteúdo da notícia raspado com sucesso: {complete_news_item.get('titulo', 'N/A')}")
    except requests.RequestException as e:
        print(f"Erro ao acessar a página da notícia: {url} - {e}")
        continue
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao processar a notícia {url}: {e}")
        continue

print(f"Total de {len(complete_news)} notícias completas raspadas com sucesso.")

#salva um csv com base na estrutura do organized news
if complete_news:
    db_config = {
        'host': os.getenv("PG_HOST"),
        'port': os.getenv("PG_PORT"),
        'user': os.getenv("PG_USER"),
        'password': os.getenv("PG_PASSWORD"),
        'dbname': os.getenv("PG_DATABASE")
    }
    save_headlines_pgsql(complete_news, db_config)
    print("Notícias salvas no banco de dados PostgreSQL.")
else:
    print("Nenhuma notícia completa foi raspada para salvar.")