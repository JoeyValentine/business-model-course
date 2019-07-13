import requests
client_id = "U9o1BOYlza715AymifC7"
client_secret = "Brg1dXOFP8"
url = "https://openapi.naver.com/v1/vision/face" # 얼굴감지
#url = "https://openapi.naver.com/v1/vision/celebrity" # 유명인 얼굴인식
files = {'image': open('D:\\naverCloud\\Lecture\\소프트웨어실습\\kids-143022__340.jpg', 'rb')}
headers = {'X-Naver-Client-Id': client_id, 'X-Naver-Client-Secret': client_secret }
response = requests.post(url,  files=files, headers=headers)
rescode = response.status_code
if(rescode==200):
    print (response.text)
else:
    print("Error Code:" + rescode)