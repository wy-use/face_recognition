
# coding: utf-8

# In[1]:

import cv2

#视频来源
#此处为电脑自带摄像头
cap = cv2.VideoCapture(0)          
#下面两行为网络摄像头来源，ur1为视频流地址，其中admin为摄像头用户名，888888为密码，@后为ip，10554为rtsp端口
#url = 'rtsp://admin:888888@10.5.3.105:10554/tcp/av0_0'
#cap = cv2.VideoCapture(url)

classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")  #分类器

num=0                                                   #目前已经识别几张人脸，初始为0
catch_pic_num=10000                                     #图片最多存储量
path_name="E:/anaconda/face_recognition/text2/image"    #识别出人脸后，人脸的保存路径

while True:
    if not cap.isOpened():
        break
    ret, frame = cap.read()                             #从摄像头读取一帧画面
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #灰度处理，提高效率
    faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))      #人脸检测
    #对同一个画面有可能出现多张人脸，因此，我们需要用一个for循环将所有检测到的人脸都读取出来，然后逐个用矩形框框出来
    if len(faceRects) > 0:          
        for faceRect in faceRects:  
            x, y, w, h = faceRect                       #每张人脸在图像中的起始坐标（左上角，x、y）以及长、宽（h、w）
            img_name = "%s/%d.jpg" % (path_name, num)   #存储人脸
            image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            cv2.imwrite(img_name, image,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            num += 1
            if num > (catch_pic_num):   
                break
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)      #框出人脸
       
    cv2.imshow('window_name', frame)        #显示图像
    if num > (catch_pic_num):               #识别人脸到达设定值退出
        break
    c = cv2.waitKey(10)                     #或者按q键退出
    if c & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:



