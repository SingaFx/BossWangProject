import requests
import re
import os
import time
import random
import socket
import struct


def download_reCaptcha(uid):
    public_key = '6LfIKgsUAAAAAMTcTP80XeGNN4qzrgluksw030Vf'
    private_key = '6LfIKgsUAAAAAHyp-LaOAe2MxaGrdkdokBiX1hSw'
    url = 'https://www.google.com/recaptcha/api/challenge?k={0}'.format(public_key)
    captcha_img_url = 'https://www.google.com/recaptcha/api/image?c={0}&th={1}'

    try:
        # Getting reCAPTCHA image
        r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, allow_redirects=True)
        cregex = re.compile(r"challenge : \'(.*)\'")
        tregex = re.compile(r"t3 : \'(.*)\'")

        challenge = cregex.search(str(r.content)).group(1)
        t = tregex.search(str(r.content)).group(1)
        link = captcha_img_url.format(challenge, t)

        r = requests.get(link, headers={'User-Agent':'Mozilla/5.0'}, allow_redirects=True)

        fname = 'img{0}.jpeg'.format(uid)

        directory = os.getcwd()
        directory = directory + '/img/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        print "Get reCAPTCHA image...."
        with open(directory + fname, "wb") as handle:
            for data in r.iter_content(chunk_size=1024):
                handle.write(data)

        return directory + fname, challenge
    except Exception as e:
        print e



def evaluate(response, challenge):
    verify_url = 'https://www.google.com/recaptcha/api/verify'
    # Verify response
    try:
        fk_ip = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
        r = requests.post(verify_url, data={'privatekey':private_key, 'response': response, 'challenge': challenge, 'remoteip': fk_ip})
        if "true" in r.text:
            print True
        else:
            print False
    except Exception as e:
        print e
