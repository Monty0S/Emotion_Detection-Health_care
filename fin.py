import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
import webbrowser
import easygui
import time
import subprocess

# Load pre-trained emotion detection models
model1 = load_model("C:/Users/deboj/Downloads/RP/best_model.h5")
model2 = load_model("C:/Users/deboj/Downloads/RP/best1_model.h5")
model3 = load_model("C:/Users/deboj/Downloads/RP/best_model_vgg.h5")

# Define the weights for each model
weights = [0.44, 0.32, 0.24]

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

face_detection_paused = False  # Flag to control face detection

while True:
    ret, test_img = cap.read()
    if not ret:
        continue

    if not face_detection_paused:
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            # Make predictions using all three models
            predictions = [model1.predict(img_pixels)[0], model2.predict(img_pixels)[0], model3.predict(img_pixels)[0]]

            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

            # Calculate weighted average of predictions
            weighted_prediction = np.zeros_like(predictions[0])
            for i, prediction in enumerate(predictions):
                weighted_prediction += weights[i] * prediction

            max_index = np.argmax(weighted_prediction)
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check if the detected emotion is "fear"
            if predicted_emotion == "fear":
                # Pause face detection
                face_detection_paused = True

                # Open a pop-up window to ask the user if everything is alright
                user_response = easygui.enterbox("Hey, I detected fear in your face. Is everything alright? Please type your response:", "Emotion Detection")

                # Resume face detection
                face_detection_paused = False

                # Check if the user responded
                if user_response:
                    if user_response.lower() in ["ok", "fine", "everything is alright"]:
                        easygui.msgbox("Glad to know about your wellbeing!", "Response")
                    elif user_response.lower() in ["no", "help", "save me"]:
                        # If user responds with "no", "help", or "save me", send a message for help
                        webbrowser.open("https://web.whatsapp.com/")  # Open WhatsApp web
                        # Replace the phone numbers with the numbers you want to send the help message to
                        phone_numbers = ["+918658952843", "+91 9903595796"]
                        message = "Emergency! I need help!"
                        for number in phone_numbers:
                            webbrowser.open(f"https://wa.me/{number}?text={message}")
                        easygui.msgbox("Help message sent!", "Response")
                        #
                        subprocess.Popen(["python", "press_enter.py"])  # Run the script to press Enter
                    else:
                        easygui.msgbox("Unknown response!", "Response")
                else:
                    # If there is no response from the user, open WhatsApp and send message for help
                    time.sleep(7)  # Wait for 7 seconds
                    webbrowser.open("https://web.whatsapp.com/")  # Open WhatsApp web
                    # Replace the phone numbers with the numbers you want to send the help message to
                    phone_numbers = ["+918658952843", "+919903595796"]
                    message = "Emergency! I need help!"
                    for number in phone_numbers:
                        webbrowser.open(f"https://wa.me/{number}?text={message}")
                    easygui.msgbox("Help message sent!", "Response")
                    ##
                    subprocess.Popen(["python", "press_enter.py"])  # Run the script to press Enter

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
