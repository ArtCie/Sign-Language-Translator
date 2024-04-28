import json
from urllib.parse import unquote
import os
from urllib.request import urlretrieve, urlopen
import ssl


ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

with open('litery.json') as f:
    result = json.load(f)
    for item in result["items"]:
        url = item["media"]["url"]
        key = unquote(url[url.rfind("/") + 1:-4])
        file_path = f"../dane/{key}"
        os.mkdir(file_path)

        with urlopen(url, context=ssl_context) as response, open(f"{file_path}/1.mp4", 'wb') as out_file:
            out_file.write(response.read())
