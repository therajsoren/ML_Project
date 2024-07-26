# ML_Project

#Image Processing with OpenCV

#Requirements

    Python  3.12.1
    OpenCV (cv2)

# install OpenCv

    pip install opencv-python

# Write a code

    import cv2

    file_path = "imagePath"

# Way to load image in python

    if image and faceDetection.py are in same directory you have to just write image name with extension like jpg,png
    if image is in different location copy the path of image by selecting the image and pressing ctrl + shift + c in keyword (for windows 11)

# Write a code to check if the image is loaded or Not

    for these we have to use if else statement
    We will be going to use None keyword by writing following line of code

    if img is None:
    print(f"Error loading image: {image_path}")
    else:
        print("Image loaded successfully.")

# Define the size of the output window
    desired_width = 800  
    desired_height = 600  

    (h, w) = img.shape[:2]


    aspect_ratio = w / h

# Resizing the image

    we declare a variable resized_img and assign it using cv2.resize which is openCv function to resize the images
    cv2 function take three parameters which will be img(source images), (new_width , new_height), and interpolation method
    which is method estimate the values between two known values . Here, cv2.INTER_LINEAR is used, which is a bilinear
    interpolation method. It's a good balance between speed and quality for most resizing tasks

# Load the cascade classifiers

    We will load load the pre-trained Haar cascade classifiers for face and eye detection.
    These classifiers will be used later in the script to detect faces and eyes in the given image.

    for tutorial on how to install refers to these link ->
    https://scribehow.com/shared/Find_and_Download_Haar_Cascade_XML_Files_from_OpenCVGitHub__fm5Szmu_SI2LhTzwjcM9mw

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    cv2.CascadeClassifier(): Function to load pre-trained Haar cascade classifiers.
    cv2.data.haarcascades: Path to the directory containing Haar cascade XML files.
    haarcascade_frontalface_default.xml: XML file for the frontal face detection classifier.
    haarcascade_eye_tree_eyeglasses.xml: XML file for the eye detection classifier (including eyeglasses).
    face_cascade and eye_cascade: Initialized objects for face and eye detection, respectively.

    These classifiers are now ready to be used to detect faces and eyes in images.

#Convert to grayscale for detection

    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Detect faces

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Iterate over image

    for (x, y, w, h) in faces:
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Draw rectangle around face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = resized_img[y:y + h, x:x + w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)  # Draw rectangle around eyes


# Defining output path and saving the output image

    output_path = r"C:\Projects\sagar\output_image.jpg"
    cv2.imwrite(output_path, resized_img)

# Display the Image in a Window
    cv2.imshow('Processed Image', resized_img)
# Wait for a Key Press
    cv2.waitKey(0)
# Close All Open Windows:
    cv2.destroyAllWindows()
