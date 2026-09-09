import re

url = input("URL: ").strip()
matches = re.search(r"^https?://(www\.)?twitter\.com/(.+)$",url,re.IGNORECASE)
#username = re.sub(r"^(https?://)?(www\.)?twitter\.com/","",url)
if matches:=re.search(r"^https?://(?:www\.)?twitter\.com/(\w+)$",url,re.IGNORECASE):
    print(f"Username:",matches.group(1))