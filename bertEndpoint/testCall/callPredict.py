import requests

def local_test():
    response = requests.post('http://localhost:8080/predict', headers={'Content-Type': 'application/json'},
    json={'text': 'อินเทอร์เน็ตถูกจำกัด (Great Firewall) แอปและเว็บไซต์ยอดนิยม เช่น Google, YouTube, Facebook, Instagram และ Line ถูกบล็อกในจีน ต้องใช้ VPN ถึงจะเข้าได้ และบางครั้ง VPN ก็ใช้ไม่ได้ 100% เรื่องนี้ก็ต้องเตรียมตัวเยอะๆค่ะ เพราะคนไทยหลายคนชอบหอบงานไปเที่ยวด้วย ทำให้เป็นปัญหากับงานไปอีก'})
    print(response.json())


def local_test():
    response = requests.post('http://localhost:8080/predict', headers={'Content-Type': 'application/json'},
    json={'text': 'อินเทอร์เน็ตถูกจำกัด (Great Firewall) แอปและเว็บไซต์ยอดนิยม เช่น Google, YouTube, Facebook, Instagram และ Line ถูกบล็อกในจีน ต้องใช้ VPN ถึงจะเข้าได้ และบางครั้ง VPN ก็ใช้ไม่ได้ 100% เรื่องนี้ก็ต้องเตรียมตัวเยอะๆค่ะ เพราะคนไทยหลายคนชอบหอบงานไปเที่ยวด้วย ทำให้เป็นปัญหากับงานไปอีก'})
    print(response.json())


def cloud_test():
    response = requests.post('https://ellabert-image-776241027088.asia-southeast1.run.app/predict', headers={'Content-Type': 'application/json'},
    json={'text':
            """
            I want to go even further beyound.
            """})

    print(response.json())

local_test()

#cloud_test()