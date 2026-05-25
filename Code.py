MAIN CODE
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
model=YOLO('yolov8s.pt')#pre trained model
def RGB(event, x, y, flags, param):
if event == cv2.EVENT_MOUSEMOVE :
point = [x, y]
print(point)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('vidp.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
DEPARTMENT OF CSE (ARTIFICIAL INTELLIGENCE & MACHINE LEARNING) 43
persondown={}
tracker=Tracker()
counter1=[]
personup={}
counter2=[]
cy1=194
cy2=220
offset=6
while True:
ret,frame = cap.read()
if not ret:
break
# frame = stream.read()
count += 1
if count % 3 != 0:
continue
frame=cv2.resize(frame,(1020,500))
results=model.predict(frame)
# print(results)
a=results[0].boxes.data
px=pd.DataFrame(a).astype("float")
# print(px)
list=[]
for index,row in px.iterrows():
# print(row)
DEPARTMENT OF CSE (ARTIFICIAL INTELLIGENCE & MACHINE LEARNING) 44
x1=int(row[0])
y1=int(row[1])
x2=int(row[2])
y2=int(row[3])
d=int(row[5])
c=class_list[d]
if 'person' in c:
list.append([x1,y1,x2,y2])
bbox_id=tracker.update(list)
for bbox in bbox_id:
x3,y3,x4,y4,id=bbox
cx=int(x3+x4)//2
cy=int(y3+y4)//2
cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
if cy1<(cy+offset) and cy1>(cy-offset):
cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
persondown[id]=(cx,cy)
if id in persondown:
if cy2<(cy+offset) and cy2>(cy-offset):
cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
if counter1.count(id)==0:
counter1.append(id)
#for upside
if cy2<(cy+offset) and cy2>(cy-offset):
DEPARTMENT OF CSE (ARTIFICIAL INTELLIGENCE & MACHINE LEARNING) 45
cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
personup[id]=(cx,cy)
if id in personup:
if cy1<(cy+offset) and cy1>(cy-offset):
cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
if counter2.count(id)==0:
counter2.append(id)
cv2.line(frame,(3,cy1),(1018,cy1),(0,255,0),2)
cv2.line(frame,(5,cy2),(1019,cy2),(0,255,255),2)
down=(len(counter1))
up=(len(counter2))
cvzone.putTextRect(frame,f'Down{down}',(50,60),2,2)
cvzone.putTextRect(frame,f'Up{up}',(50,160),2,2)
cv2.imshow("RGB", frame)
if cv2.waitKey(1)&0xFF==27:
break
cap.release()
cv2.destroyAllWindows()

#TRACKER CODE
import math
class Tracker:
def init (self):
# Store the center positions of the objects
DEPARTMENT OF CSE (ARTIFICIAL INTELLIGENCE & MACHINE LEARNING) 46
self.center_points = {}
# Keep the count of the IDs
# each time a new object id detected, the count will increase by one
self.id_count = 0
def update(self, objects_rect):
# Objects boxes and ids
objects_bbs_ids = []
# Get center point of new object
for rect in objects_rect:
x, y, w, h = rect
cx = (x + x + w) // 2
cy = (y + y + h) // 2
# Find out if that object was detected already
same_object_detected = False
for id, pt in self.center_points.items():
dist = math.hypot(cx - pt[0], cy - pt[1])
if dist < 35:
self.center_points[id] = (cx, cy)
# print(self.center_points)
objects_bbs_ids.append([x, y, w, h, id])
same_object_detected = True
break
# New object is detected we assign the ID to that object
if same_object_detected is False:
self.center_points[self.id_count] = (cx, cy)
DEPARTMENT OF CSE (ARTIFICIAL INTELLIGENCE & MACHINE LEARNING) 47
objects_bbs_ids.append([x, y, w, h, self.id_count])
self.id_count += 1
# Clean the dictionary by center points to remove IDS not used anymore
new_center_points = {}
for obj_bb_id in objects_bbs_ids:
_, _, _, _, object_id = obj_bb_id
center = self.center_points[object_id]
new_center_points[object_id] = center
# Update dictionary with IDs not used removed
self.center_points = new_center_points.copy()
return objects_bbs_ids
