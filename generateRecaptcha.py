import requests
from bs4 import BeautifulSoup
import re
import os
import time
import sys


def getReCaptcha(N):
    url = 'https://www.google.com/recaptcha/api/challenge?k=6LfIKgsUAAAAAMTcTP80XeGNN4qzrgluksw030Vf'
    captcha_url = 'https://www.google.com/recaptcha/api/image?c={0}&th={1}'
    for i in range(N):
        try:
            r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, allow_redirects=True)
            soup = BeautifulSoup(r.content, 'lxml')
            cregex = re.compile(r"challenge : \'(.*)\'")
            tregex = re.compile(r"t3 : \'(.*)\'")

            c = cregex.search(str(soup)).group(1)
            t = tregex.search(str(soup)).group(1)
            link = captcha_url.format(c, t)

            r = requests.get(link, headers={'User-Agent':'Mozilla/5.0'}, allow_redirects=True)

            fname = 'img{0}.jpeg'.format(i)

            directory = os.getcwd()
            directory = directory + '/input/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            print "Downloading....", fname
            with open(directory + fname, "wb") as handle:
                for data in r.iter_content(chunk_size=1024):
                    handle.write(data)
            time.sleep(0.5)
        except Exception as e:
            print e

if __name__ == '__main__':
    n = sys.argv[1]
    print "Generating", n, "recaptchas"
    getReCaptcha(int(n))
