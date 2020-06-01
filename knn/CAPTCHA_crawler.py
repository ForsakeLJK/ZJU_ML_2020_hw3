import requests
import random
import time

'''
A trivial crawler to fetch CAPTCHA images
'''

cnt = 0
num = 500

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
    'Accept':'image/webp, image/apng, image/*, */*;q=0.8',
    'Accept-Encoding':'gzip, deflate',
    'Accept-Language':'zh-CN, zh;q=0.9,en;q=0.8,ja;q=0.7'
}

for i in range(num):

    if(i%100 == 0):
        time.sleep(1)

    num = random.random() * 1000
    url = "http://cwcx.zju.edu.cn/WFManager/loginAction_getCheckCodeImg.action?s=" + \
        str(num)
    res = requests.get(url, headers=headers)

    with open('knn/CAPTCHA_train/'+str(cnt) + '.jpg', 'wb') as f:
    # with open('knn/CAPTCHA_test/'+str(cnt) + '.jpg', 'wb') as f:
        f.write(res.content)
    cnt += 1


print("done. {} images fetched.".format(cnt+1))