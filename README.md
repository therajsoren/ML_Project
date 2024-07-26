# ML_Project

#Image Processing with OpenCV

#Requirements

    Python  3.12.1
    OpenCV (cv2)

# install OpenCv

    open the terminal and move to the directory in which the faceDetection.py is stored
    then install OpenCV by writing these command in terminal

    pip install opencv-python

# Write a code

    In faceDetection.py write these line of code to import OpenCv

    import cv2

    upon writing these code cv2 library is loaded which we going to use for face and eye detection

    write these line of code after that

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

    we chose two variable one will be storing the desired width of the window
    other will be storing desiring height of the window

    After that we write these code to get the original dimension of the image

    (h, w) = img.shape[:2]

    after we will calculate the aspect ratio of the image
    aspect_ratio = w / h

    and after that we check if check the condition if the width of the original image
    is greater then height then the width of image will be set to desired width
    and new height will be given by these formula desired_width / aspect_ratio
    and reverse of it we will be doing if the height of image is greater than width

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

        cv2.cvtColor(): Function to convert the color space of an image.
        resized_img: The resized image to be converted.
        cv2.COLOR_BGR2GRAY: Conversion code to change the image from BGR to grayscale.
        gray: The resulting grayscale image.
        By converting the resized image to grayscale, you're preparing it for efficient and effective face and eye detection.

# Detect faces

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_cascade.detectMultiScale(): Method to detect objects in the image.
    gray: Grayscale image to be analyzed.
    1.1: Scale factor for the image pyramid.
    4: Minimum number of neighbors to retain a detection.
    faces: List of detected faces, each represented by a rectangle (x, y, w, h).
    By calling this method, you are identifying regions in the image where faces are
    likely present, which can then be used for further processing like drawing rectangles around the faces.

# Iterate over image

    Iterate Over Faces:

    Loop through each detected face.
    Draw a blue rectangle around each face.
    Define regions of interest (ROIs) for face regions in both grayscale and color images.
    Detect Eyes:

    Use the grayscale ROI to detect eyes within the face.
    Loop through each detected eye and draw a green rectangle around each one in the color ROI.
    By following these steps, you visually mark the detected faces and eyes on the resized image, making it easier to see the results of the detection process.\

# Defining output path and saving the output image

    output_path = r"C:\Projects\sagar\output_image.jpg"
    cv2.imwrite(output_path, resized_img)

# Display the Image in a Window
    cv2.imshow('Processed Image', resized_img)
# Wait for a Key Press
    cv2.waitKey(0)
# Close All Open Windows:
    cv2.destroyAllWindows()
