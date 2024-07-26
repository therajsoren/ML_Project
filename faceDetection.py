"""
Created on Fri Jul 25 15:17:33 2024

@author: 7rajk
"""

import cv2


image_path = r"C:\Projects\sagar\aman.jpg"

# Load the image
img = cv2.imread(image_path)

# Check if image load or not
if img is None:
    print(f"Error loading image: {image_path}")
else:
    print("Image loaded successfully.")
    
    # Define the size of the output window
    desired_width = 800  
    desired_height = 600  

    # Calculate the aspect ratio
    (h, w) = img.shape[:2]
    aspect_ratio = w / h

    
    if w > h:
        new_width = desired_width
        new_height = int(desired_width / aspect_ratio)
    else:
        new_height = desired_height
        new_width = int(desired_height * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Load the cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    # Convert to grayscale for detection
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    
    for (x, y, w, h) in faces:
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Draw rectangle around face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = resized_img[y:y + h, x:x + w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)  # Draw rectangle around eyes


    output_path = r"C:\Projects\sagar\output_image.jpg"
    cv2.imwrite(output_path, resized_img)


    cv2.imshow('Processed Image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
