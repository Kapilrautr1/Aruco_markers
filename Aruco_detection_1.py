import cv2
import cv2.aruco as aruco

VideoCap=True
cap=cv2.VideoCapture(0)

def DetectorParameters_create():
    return aruco.DetectorParameters()

def findAruco(img,marker_size=6,total_markers=250,draw=True):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    key=getattr(aruco,f'DICT_{marker_size}X{marker_size}_{total_markers}')

    arucoDict=aruco.getPredefinedDictionary(key)
    arucoParam=DetectorParameters_create()
    bbox,ids,_=aruco.detectMarkers(gray,arucoDict,parameters=arucoParam)
    print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bbox)
        if ids is not None:
            for i in range(len(ids)):
               
                top_left_corner = tuple(bbox[i][0][0].astype(int))  # Get the top-left corner of the bounding box to place the text
               
                cv2.putText(img, str(ids[i][0]), top_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)   # Draw the ID of the Aruco marker on the image

    return bbox,ids

while True:
    if VideoCap:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
    else: 
        img=cv2.imread("/home/kapil/codes/2.png")
        img= cv2.resize(img,(0,0),fx=3,fy=3)

    findAruco(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("img",img)    