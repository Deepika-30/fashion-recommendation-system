

import cv2
from cvzone.PoseModule import PoseDetector

cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
cascade = cv2.CascadeClassifier(cascade_path)

detector = PoseDetector()
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()



    img = detector.findPose(img)
    lmlist, boxinfo = detector.findPosition(img, bboxWithHands=True)



# Load the pre-trained Haar cascade for person detection


# Open a video capture object (0 for default camera)




    if not success:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform person detection using the Haar cascade
    persons = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected persons and calculate height and width
    for (x, y, w, h) in persons:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Calculate person height and width
        person_height = h
        person_width = w
        ratio = person_height/person_width
        text = ''
        if ratio>1.5:
            text = 'ectomorph'
        elif ratio<1.4:
            text = 'mesomorph'
        elif ratio<1 or ratio<1.4:
            text = 'ectomorph'
        # Display the height and width on the frame
        cv2.putText(img, f'Height: {h} Width: {w} body-type: {text}' , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with bounding boxes and measurements
    cv2.imshow("Person Detection", img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the height and width of the last detected person
print("Last Detected Person Height:", person_height)
print("Last Detected Person Width:", person_width)
print("Person bodytype:", text)

