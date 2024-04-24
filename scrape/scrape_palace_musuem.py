import requests
from bs4 import BeautifulSoup
import pandas as pd


def fetch_page(id):
    url = f'https://theme.npm.edu.tw/opendata/DigitImageSets.aspx?Key=^^11&pageNo={id}'

    res = requests.get(url)
    if res.status_code != 200:
        raise ValueError(f'Encountered error when visiting {url}: {res.status_code}')

    html_string = res.text
    soup = BeautifulSoup(html_string, 'html.parser')

    # find paintings
    ul_element = soup.find('ul', class_='painting-list')

    paintings_info = []

    if ul_element:
        painting_links = ul_element.find_all('a')
        painting_urls = [link['href'] for link in painting_links]

        # fetch all paintings in this page
        for painting_url in painting_urls:
            painting_info = fetch_detail(f'https://theme.npm.edu.tw/opendata/{painting_url}')
            paintings_info.append(painting_info)
    else:
        print("no painting")
    return paintings_info


def fetch_detail(url):
    res = requests.get(url)
    if res.status_code != 200:
        raise ValueError(f'Encountered error when visiting {url}: {res.status_code}')

    html_string = res.text
    soup = BeautifulSoup(html_string, 'html.parser')

    img_div = soup.find('div', class_='project-img')
    detail_div = soup.find('div', class_='project-detail')

    if detail_div:
        # title
        painting_title = detail_div.find('h3').text.strip()

        # painting details
        painting_info = {'title': painting_title}
        info_list = detail_div.find('ul').find_all('li')
        for info in info_list:
            key, value = info.text.split('：', 1)
            painting_info[key] = value

    if img_div:
        # save painting image
        img_src = img_div.find('img')['src']
        image_response = requests.get(img_src)
        image_content = image_response.content
        image_filename = 'paintings/' + painting_info['文物圖檔編號'] + '.jpg'
        with open(image_filename, 'wb') as f:
            f.write(image_content)

    return painting_info


def fetch_all(start_page=1):
    csv_file = 'paintings.csv'
    df = pd.read_csv(csv_file)

    # fetch all pages
    page = start_page
    while True:
        print(f'Fetching page {page}...')
        paintings_info = fetch_page(page)
        if len(paintings_info) == 0: break

        for info in paintings_info:
            id = info.get('文物圖檔編號', None)
            title = info.get('title', None)
            author = info.get('作者', None)
            dynasty = info.get('朝代', None)
            size = info.get('本幅尺寸', None)
            content = info.get('說明文', None)
            row = {'id': id, 'title': title, 'author': author, 'dynasty': dynasty, 'size': size, 'content': content}
            df = df._append(row, ignore_index=True)

        df.to_csv(csv_file, index=False)
        # next page
        page += 1


if __name__ == '__main__':
    fetch_all(start_page=1)
