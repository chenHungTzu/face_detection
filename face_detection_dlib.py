import dlib
import cv2 
import imutils 
import uuid
import numpy as np
from PIL import Image
from traning_module_CNN import classifier
from traning import training_set
from runpy import run_path
import os
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
      roi = img[y1:y2, x1:x2]
      ##filename = 'data/' + str(uuid.uuid4()) + '.png'
      ##cv2.imwrite(filename,roi);
      im = scipy.misc.toimage(roi,255)
      crim = im.resize((64,64))
      test_image = np.expand_dims(crim, axis = 0)
      result = classifier.predict_classes(test_image)
 
      index = result.item()
      dis_arr = list(training_set.class_indices.keys())

      cv2.putText(frame, str(dis_arr[index]), (x1, y1) , cv2.FONT_HERSHEY_DUPLEX,
                  0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
      cv2.imshow("Face Detection", frame)
    except :
      print("An exception occurred")
   
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  
cap.release()
cv2.destroyAllWindows()