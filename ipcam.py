import cv2

url = 'http://10.27.2.75:8080/video'
cap = cv2.VideoCapture(url)

while(cap.isOpened()):
    # Capture frame-by-frame  
    ret, frame = cap.read()

    # Display the resulting frame  
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture  
cap.release()
cv2.destroyAllWindows()
