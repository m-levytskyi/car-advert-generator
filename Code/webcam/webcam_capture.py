import cv2

class WebcamCapture:
    def __init__(self, save_path="captured_image.jpg"):
        """
        Initialize the WebcamCapture class.

        :param save_path: Path where the captured image will be saved. Default is 'captured_image.jpg'.
        """
        self.save_path = save_path

    def capture_image(self):
        """
        Captures an image from the webcam and saves it to the specified path.
        """
        # Open the webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Unable to access the webcam.")
            return

        print("Press 'SPACE' to capture an image or 'ESC' to exit.")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            # Show the video feed
            cv2.imshow("Webcam - Press SPACE to Capture", frame)

            # Wait for user input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                print("Exiting without saving.")
                break
            elif key == 32:  # SPACE key to capture
                # Save the image
                cv2.imwrite(self.save_path, frame)
                print(f"Image saved to {self.save_path}")
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam = WebcamCapture("Code/webcam/webcam_image.jpg")
    webcam.capture_image()
