import cv2 as cv
import time
import os
from ultralytics import YOLO

try:
    os.mkdir( "mySaves")
except:
    # Directory already exists
    pass

def getNewFileName():
    return "_".join( map( str, time.localtime()[:6]))

model = YOLO( "yolo11s.pt")

cap = cv.VideoCapture( "test\\demo480.mp4")

try:
    while True:
        ret,frame = cap.read()
        dogC,catC = 0,0
        if not ret:
            break
        try:
            results = model.predict( frame, classes = [15,16], conf = 0.4, verbose = False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = ( map( int, box.xyxy[0]))
                    cv.rectangle(frame, (x1, y1), (x2, y2), [(0,0,255) if box.cls.item() == 15 else (0,255,0)][0], 1)
                print( f"{sum( result.speed.values()):.2f} ms")
            
            catC = results[0].boxes.cls.tolist().count( 15)
            dogC = results[0].boxes.cls.tolist().count( 16)
        except:
            pass
        finally:
            cv.putText( frame, f"Dogs: { dogC}", ( 10, 20), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            cv.putText( frame, f"Cats: { catC}", ( 10, 50), cv.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)

        cv.imshow( "Display", frame)
        
        key = cv.waitKey(1)
        if key == ord( "q"):
            break
        elif key == ord( "s"):
            cv.imwrite( f"mySaves\\{getNewFileName()}.jpg", frame)
            print( "Image Saved!")
except Exception as e:
    print( e)
finally:
    cap.release()
    cv.destroyAllWindows()