import dlib
import cv2 
import uuid
import numpy as np
from PIL import Image
from traning_module_CNN import classifier
from traning import training_set
from runpy import run_path
import scipy.misc

# 開啟影片檔案
cap = cv2.VideoCapture(0)

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

# 以迴圈從影片檔案讀取影格，並顯示出來
while(cap.isOpened()):
  ret, frame = cap.read()

  # 偵測人臉
  face_rects, scores, idx = detector.run(frame, 0)

  # 取出所有偵測的結果
  for i, d in enumerate(face_rects):
    #取得結果的長寬高座標點
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    #text = "%2.2f(%d)" % (scores[i], idx[i])

    # 以方框標示偵測的人臉
    img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 0, cv2.LINE_AA)

  #搜集資訊
  if img.any():   
    try:

      #對於圖片截圖
      roi = img[y1:y2, x1:x2]

      #存取圖片（測試用）
      ##filename = 'data/' + str(uuid.uuid4()) + '.png'

      #圖片寫入（測試用）
      ##cv2.imwrite(filename,roi);

      #透過spicy 將npArray 轉成圖片格式
      im = scipy.misc.toimage(roi,255)

      #轉為64X64 像素
      crim = im.resize((64,64))

      #???
      test_image = np.expand_dims(crim, axis = 0)

      #送入模型判斷
      result = classifier.predict_classes(test_image)
 
      #取得結果指標
      index = result.item()

      #將dict_array （key list）轉為list
      dis_arr = list(training_set.class_indices.keys())

      #將對應的結果分類讀出
      cv2.putText(frame, str(dis_arr[index]), (x1, y1) , cv2.FONT_HERSHEY_DUPLEX,
                  0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
      #並顯示在畫面上
      cv2.imshow("Face Detection", frame)

    except :
      print("An exception occurred")
   
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  
cap.release()
cv2.destroyAllWindows()