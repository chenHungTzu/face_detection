import dlib
import cv2 
import imutils 
import uuid
from PIL import Image

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
    text = "%2.2f(%d)" % (scores[i], idx[i])

    # 以方框標示偵測的人臉
    img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)


    # 標示分數
    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
            0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # 顯示結果
    cv2.imshow("Face Detection", frame)

  #搜集資訊
  if img.any():   
    crop = img;
    roi = crop[y1:y2, x1:x2]
    filename = 'data/' + str(uuid.uuid4()) + '.png'
    cv2.imwrite(filename, roi);
    # crpim = img.crop((x1,y1,x1 + x2,y1 + y2)).resize((64,64))
    # crpim.save('data/' + str(uuid.uuid4()) + '.png')

  #data tranung

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  
cap.release()
cv2.destroyAllWindows()