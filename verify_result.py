import requests
import re
import os
import time
import random
import socket
import struct


public_key = '6LfIKgsUAAAAAMTcTP80XeGNN4qzrgluksw030Vf'
private_key = '6LfIKgsUAAAAAHyp-LaOAe2MxaGrdkdokBiX1hSw'
url = 'https://www.google.com/recaptcha/api/challenge?k={0}'.format(public_key)
captcha_img_url = 'https://www.google.com/recaptcha/api/image?c={0}&th={1}'
verify_url = 'https://www.google.com/recaptcha/api/verify'
NUM = 10
for i in range(NUM):
    try:
        # Getting reCAPTCHA image
        r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, allow_redirects=True)
        cregex = re.compile(r"challenge : \'(.*)\'")
        tregex = re.compile(r"t3 : \'(.*)\'")

        challenge = cregex.search(str(r.content)).group(1)
        t = tregex.search(str(r.content)).group(1)
        link = captcha_img_url.format(challenge, t)

        r = requests.get(link, headers={'User-Agent':'Mozilla/5.0'}, allow_redirects=True)

        fname = 'img{0}.jpeg'.format(i)

        directory = os.getcwd()
        directory = directory + '/img/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        print "Get reCAPTCHA image...."
        with open(directory + fname, "wb") as handle:
            for data in r.iter_content(chunk_size=1024):
                handle.write(data)
        time.sleep(2)

        # TODO:
        # replace the raw_input with our reCAPTCHA break process
        response = raw_input("Text in reCAPTCHA is ? ")
        print "text is ", response


        # Verify response
        fk_ip = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
        r = requests.post(verify_url, data={'privatekey':private_key, 'response': response, 'challenge': challenge, 'remoteip': fk_ip})
        print r.text
    except Exception as e:
        print e
