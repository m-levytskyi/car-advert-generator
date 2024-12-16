import cv2
import os

# needed to get rid of the warning in output
os.environ["QT_STYLE_OVERRIDE"] = "fusion"

class WebcamCapture:
    def __init__(self, save_dir="captured_images"):
        """
        Initialize the WebcamCapture class.

        :param save_dir: Directory where the captured images will be saved.
        """
        self.save_dir = save_dir
        self.image_count = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def capture_images(self):
        """
        Captures images from the webcam based on user input and saves them.
        Press:
        - SPACE: To capture an image.
        - ENTER: To start the generation process.
        - ESC: To exit the program without starting generation.
        """
        # Open the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to access the webcam.")
            return

        print(
            "Instructions:\n"
            "- Point the webcam at an object and press SPACE to record an image.\n"
            "- Press ENTER to start the generation process.\n"
            "- Press ESC to exit without starting the process.\n"
        )
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            # Show the video feed
            cv2.imshow("Webcam - Press SPACE to Capture, ENTER to Start the generation", frame)


            # Wait for user input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                print("Exiting without saving.")
                break
            elif key == 32:  # SPACE key to capture
                self.image_count += 1
                image_path = os.path.join(self.save_dir, f"image_{self.image_count}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Image {self.image_count} recorded and saved as {image_path}")
            elif key == 13:  # ENTER key to start generation process
                print(f"Starting the generation process with {self.image_count} images.")
                break
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     webcam = WebcamCapture("Code/webcam/webcam_image.jpg")
#     webcam.capture_image()
