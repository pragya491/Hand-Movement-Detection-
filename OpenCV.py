import cv2
import numpy as np

# Function to detect hand movements
def detect_hand_movement():
    cap = cv2.VideoCapture(0)  # Initialize the webcam
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()  # Background subtraction method

    # Load the pre-trained human detection model
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        ret, frame = cap.read()  # Read frame from the webcam
        if not ret:
            break

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Remove noise
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        # Threshold the image
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process the contours
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            if area > 1000:  # Adjust the area threshold as per your requirement
                # Convex hull
                hull = cv2.convexHull(contour)

                # Calculate the aspect ratio of the bounding rectangle
                x, y, w, h = cv2.boundingRect(hull)
                aspect_ratio = float(w) / h

                # Filter contours based on aspect ratio and area
                if 0.5 < aspect_ratio < 1.5 and area > 3000:
                    # Draw bounding rectangle around the contour
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect humans in the frame
        boxes, _ = hog.detectMultiScale(frame)

        # Draw bounding boxes around the detected humans
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Hand Movement and Human Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()

# Run the combined detection function
detect_hand_movement()