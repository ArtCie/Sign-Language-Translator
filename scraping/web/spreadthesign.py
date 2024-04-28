from urllib.request import urlopen, Request
import requests
import ssl
from bs4 import BeautifulSoup

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

URL = "https://spreadthesign.com"
DEFAULT_URL = URL + "/pl.pl/alphabet/29/"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"
}


def download_video(letter: str, link: str):
    response = requests.get(URL + link, headers=DEFAULT_HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')

    video_tag = soup.find('video')

    if video_tag:
        video_url = video_tag['src']

        req = Request(video_url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0')

        with urlopen(req, context=ssl_context) as response, open(f"../../dane/{letter}/2.mp4", 'wb') as out_file:
            out_file.write(response.read())


def get_letters_dict():
    response = requests.get(DEFAULT_URL, headers=DEFAULT_HEADERS)
    bs4 = BeautifulSoup(response.text, 'html.parser')
    alphabet_letter_list_elements = bs4.find_all('ul', class_='alphabet-letter-list')
    extracted_elements_html = [str(element) for element in alphabet_letter_list_elements]
    soup = BeautifulSoup(extracted_elements_html[0], 'html.parser')
    alphabet_dict = {}
    for li in soup.find_all('li'):
        letter = li.get_text(strip=True)
        href = li.find('a')['href']
        alphabet_dict[letter] = href
    return alphabet_dict


alphabet_dict = get_letters_dict()

for letter, link in alphabet_dict.items():
    download_video(letter, link)