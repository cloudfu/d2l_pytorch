# opencv-python
import cv2
from cv2 import CV_16S
# mediapipe人工智能工具包
import mediapipe as mp
import numpy as np
# 进度条库
from tqdm import tqdm
# 时间库
import time

# 导入solution
mp_hands = mp.solutions.hands
# 导入模型
hands = mp_hands.Hands(static_image_mode=False,        # 是静态图片还是连续视频帧
                       max_num_hands=2,                # 最多检测几只手
                       min_detection_confidence=0.7,   # 置信度阈值
                       min_tracking_confidence=0.5)    # 追踪阈值
# 导入绘图函数
mpDraw = mp.solutions.drawing_utils 


# # 处理帧函数
# def process_frame(img):
#     # 水平镜像翻转图像，使图中左右手与真实左右手对应
#     # 参数 1：水平翻转，0：竖直翻转，-1：水平和竖直都翻转
#     img = cv2.flip(img, 1)
#     # BGR转RGB
#     img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # 将RGB图像输入模型，获取预测结果
#     results = hands.process(img_RGB)
    
#     if results.multi_hand_landmarks: # 如果有检测到手
#         # 遍历每一只检测出的手
#         for hand_idx in range(len(results.multi_hand_landmarks)):
#             hand_21 = results.multi_hand_landmarks[hand_idx] # 获取该手的所有关键点坐标
#             mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS) # 可视化
#         # 在三维坐标系中可视化索引为0的手
#         # mpDraw.plot_landmarks(results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
#     return img

def get_angle(v1,v2):
    angle = np.dot(v1,v2)/(np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2)))
    angle = np.arccos(angle)/3.14*180
    
    return angle 
    
    
def get_str_guester(up_fingers,list_lms):
    
    if len(up_fingers)==1 and up_fingers[0]==8:
        
        v1 = list_lms[6]-list_lms[7]
        v2 = list_lms[8]-list_lms[7]
        
        angle = get_angle(v1,v2)
       
        if angle<160:
            str_guester = "9"
        else:
            str_guester = "1"
    
    elif len(up_fingers)==1 and up_fingers[0]==4:
        str_guester = "Good"
    
    elif len(up_fingers)==1 and up_fingers[0]==20:
        str_guester = "Bad"
        
    elif len(up_fingers)==1 and up_fingers[0]==12:
        str_guester = "FXXX"
   
    elif len(up_fingers)==2 and up_fingers[0]==8 and up_fingers[1]==12:
        str_guester = "2"
        
    elif len(up_fingers)==2 and up_fingers[0]==4 and up_fingers[1]==20:
        str_guester = "6"
        
    elif len(up_fingers)==2 and up_fingers[0]==4 and up_fingers[1]==8:
        str_guester = "8"
    
    elif len(up_fingers)==3 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==16:
        str_guester = "3"
    
    elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==12:
  
        dis_8_12 = list_lms[8,:] - list_lms[12,:]
        dis_8_12 = np.sqrt(np.dot(dis_8_12,dis_8_12))
        
        dis_4_12 = list_lms[4,:] - list_lms[12,:]
        dis_4_12 = np.sqrt(np.dot(dis_4_12,dis_4_12))
        
        if dis_4_12/(dis_8_12+1) <3:
            str_guester = "7"
        
        elif dis_4_12/(dis_8_12+1) >5:
            str_guester = "Gun"
        else:
            str_guester = "7"
            
    elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==20:
        str_guester = "ROCK"
    
    elif len(up_fingers)==4 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==16 and up_fingers[3]==20:
        str_guester = "4"
    
    elif len(up_fingers)==5:
        str_guester = "5"
        
    elif len(up_fingers)==0:
        str_guester = "10"
    
    else:
        str_guester = " "
        
    return str_guester



def get_angle(v1,v2):
    angle = np.dot(v1,v2)/(np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2)))
    angle = np.arccos(angle)/3.14*180
    
    return angle 
    
    
def get_str_guester(up_fingers,list_lms):
    
    if len(up_fingers)==1 and up_fingers[0]==8:
        
        v1 = list_lms[6]-list_lms[7]
        v2 = list_lms[8]-list_lms[7]
        
        angle = get_angle(v1,v2)
       
        if angle<160:
            str_guester = "9"
        else:
            str_guester = "1"
    
    elif len(up_fingers)==1 and up_fingers[0]==4:
        str_guester = "Good"
    
    elif len(up_fingers)==1 and up_fingers[0]==20:
        str_guester = "Bad"
        
    elif len(up_fingers)==1 and up_fingers[0]==12:
        str_guester = "FXXX"
   
    elif len(up_fingers)==2 and up_fingers[0]==8 and up_fingers[1]==12:
        str_guester = "2"
        
    elif len(up_fingers)==2 and up_fingers[0]==4 and up_fingers[1]==20:
        str_guester = "6"
        
    elif len(up_fingers)==2 and up_fingers[0]==4 and up_fingers[1]==8:
        str_guester = "8"
    
    elif len(up_fingers)==3 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==16:
        str_guester = "3"
    
    elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==12:
  
        dis_8_12 = list_lms[8,:] - list_lms[12,:]
        dis_8_12 = np.sqrt(np.dot(dis_8_12,dis_8_12))
        
        dis_4_12 = list_lms[4,:] - list_lms[12,:]
        dis_4_12 = np.sqrt(np.dot(dis_4_12,dis_4_12))
        
        if dis_4_12/(dis_8_12+1) <3:
            str_guester = "7"
        
        elif dis_4_12/(dis_8_12+1) >5:
            str_guester = "Gun"
        else:
            str_guester = "7"
            
    elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==20:
        str_guester = "ROCK"
    
    elif len(up_fingers)==4 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==16 and up_fingers[3]==20:
        str_guester = "4"
    
    elif len(up_fingers)==5:
        str_guester = "5"
        
    elif len(up_fingers)==0:
        str_guester = "10"
    
    else:
        str_guester = " "
        
    return str_guester

    
def process_frame_gesture(img):

    image_width = 10
    image_height = 10
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    if results.multi_hand_landmarks:
        # 只处理一只手的手势
        hand = results.multi_hand_landmarks[0]

        # 打印Z轴数据
        print("\r%.2f %.2f %.2f %.2f %.2f %.2f "%(hand.landmark[0].z,hand.landmark[4].z,hand.landmark[8].z,hand.landmark[12].z,hand.landmark[16].z,hand.landmark[20].z),end="")
        
        # 绘制手部连接点
        mpDraw.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)

        # 采集所有关键点的坐标
        list_lms = []    
        for i in range(21):
            pos_x = hand.landmark[i].x*image_width
            pos_y = hand.landmark[i].y*image_height
            list_lms.append([int(pos_x),int(pos_y)])
        
        # 构造凸包点
        list_lms = np.array(list_lms,dtype=np.int32)
        hull_index = [0,1,2,3,6,10,14,19,18,17,10]
        hull = cv2.convexHull(list_lms[hull_index,:])
        # 绘制凸包
        cv2.polylines(img,[hull], True, (0, 255, 0), 2)
            
        # 查找外部的点数
        n_fig = -1
        ll = [4,8,12,16,20] 
        up_fingers = []
        
        for i in ll:
            pt = (int(list_lms[i][0]),int(list_lms[i][1]))
            dist= cv2.pointPolygonTest(hull,pt,True)
            if dist <0:
                up_fingers.append(i)
        
        # print(up_fingers)
        # print(list_lms)
        # print(np.shape(list_lms))
        str_guester = get_str_guester(up_fingers,list_lms)
        
        
        cv2.putText(img,' %s'%(str_guester),(90,90),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,0),4,cv2.LINE_AA)
        
            
            
        for i in ll:
            pos_x = hand.landmark[i].x*image_width
            pos_y = hand.landmark[i].y*image_height
            # 画点
            cv2.circle(img, (int(pos_x),int(pos_y)), 3, (0,255,255),-1)
                

    return img

        

def process_frame(img):
    # 记录该帧开始处理的时间
    start_time = time.time()
    
    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]

    # 水平镜像翻转图像，使图中左右手与真实左右手对应
    # 参数 1：水平翻转，0：竖直翻转，-1：水平和竖直都翻转
    img = cv2.flip(img, 1)
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = hands.process(img_RGB)

    if results.multi_hand_landmarks: # 如果有检测到手

        handness_str = ''
        index_finger_tip_str = ''
        for hand_idx in range(len(results.multi_hand_landmarks)):

            # 获取该手的21个关键点坐标
            hand_21 = results.multi_hand_landmarks[hand_idx]

            # 可视化关键点及骨架连线
            mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)

            # 记录左右手信息
            temp_handness = results.multi_handedness[hand_idx].classification[0].label
            handness_str += '{}:{} '.format(hand_idx, temp_handness)

            # 获取手腕根部深度坐标
            cz0 = hand_21.landmark[0].z

            for i in range(21): # 遍历该手的21个关键点

                # 获取3D坐标
                cx = int(hand_21.landmark[i].x * w)
                cy = int(hand_21.landmark[i].y * h)
                cz = hand_21.landmark[i].z
                depth_z = cz0 - cz

                # 用圆的半径反映深度大小
                radius = max(int(6 * (1 + depth_z*5))*2, 0)

                if i == 0: # 手腕
                    img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
                if i == 8: # 食指指尖
                    img = cv2.circle(img,(cx,cy), radius, (193,182,255), -1)
                    # 将相对于手腕的深度距离显示在画面中
                    index_finger_tip_str += '{}:{:.2f} '.format(hand_idx, depth_z)
                if i in [1,5,9,13,17]: # 指根
                    img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)
                if i in [2,6,10,14,18]: # 第一指节
                    img = cv2.circle(img,(cx,cy), radius, (1,240,255), -1)
                if i in [3,7,11,15,19]: # 第二指节
                    img = cv2.circle(img,(cx,cy), radius, (140,47,240), -1)
                if i in [4,12,16,20]: # 指尖（除食指指尖）
                    img = cv2.circle(img,(cx,cy), radius, (223,155,60), -1)
        
        scaler = 2 # 字体大小
        img = cv2.putText(img, handness_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        img = cv2.putText(img, index_finger_tip_str, (25 * scaler, 150 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        
        # 记录该帧处理完毕的时间
        end_time = time.time()
        # 计算每秒处理图像帧数FPS
        FPS = 1/(end_time - start_time)

        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    return img


# 调用摄像头逐帧实时处理模板
# 不需修改任何代码，只需修改process_frame函数即可
# 同济子豪兄 2021-7-8

# 导入opencv-python
import cv2
import time

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)

# 打开cap
cap.open(0)

# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        print('Error')
        break
    
    ## !!!处理帧函数
    frame = process_frame_gesture(frame)
    
    # 展示处理后的三通道图像
    cv2.imshow('my_window',frame)

    if cv2.waitKey(1) in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
        break
    
# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()