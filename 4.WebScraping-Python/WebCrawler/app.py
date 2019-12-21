import requests
from bs4 import BeautifulSoup

# reference: https://www.youtube.com/watch?v=XjNm9bazxn8

def trade_spider(max_pages):
    page = 0
    while page < max_pages:
        url = 'https://www.oreilly.com/search/?query=python&extended_publisher_data=true&highlight=true&include_assessments=false&include_case_studies=true&include_courses=true&include_orioles=true&include_playlists=true&include_collections=true&include_notebooks=true&is_academic_institution_account=false&source=user&sort=relevance&facet_json=true&page=' + str(page)
        #url = 'https://www.amazon.in/s?bbn=1389401031&rh=n%3A976419031%2Cn%3A%21976420031%2Cn%3A1389401031%2Cp_72%3A1318476031&dc&fst=as%3Aoff&qid=1575860236&rnid=1318475031&ref=lp_1389401031_nr_p_72_0' + str(page)
        source_code = requests.get(url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text, features="html.parser")
        for link in soup.findAll('a', {'class': 'title--1IuUh'}):
            href = link.get('href')
            print(href)
        page += 1
trade_spider(2)