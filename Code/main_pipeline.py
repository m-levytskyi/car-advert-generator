from webcam.webcam_capture import WebcamCapture

# from image_classifier ...
# TODO 
# a function to classify a car captured on webcam
# takes a webcam image as input, returns brand + body type


from article_agent.agent_pipeline import AgentPipeline
from article_assembler.assembler_pipeline import assemble_article

webcam_img_path = "Code/webcam/webcam_image.jpg"

def run_pipeline():
    print("Step 1: Capturing image...")
    webcam = WebcamCapture(webcam_img_path)
    webcam.capture_image()



    print("\n Step 2: Classifying image...")
    # TODO
    # calling a classification function
    # otput example: 
    car_dict = {
        'brand' : 'BMW',
        'body_type' : 'SUV'
    }
    print(f"Classification result: {car_dict}")



    print("Step 3: Creating JSON...")
    agent_pipeline = AgentPipeline(brand=car_dict['brand'], car_type=car_dict['body_type'])
    response = agent_pipeline()
    # save json object to a file
    with open(f"Code/article_agent/json/output.json", "w") as f:
        f.write(response)


    print("Step 4: Assembling article...")
    json_path = "Code/article_agent/json/output.json" #article agent output
    images_path = "Code/article_assembler/tmp/imgs" #generated images will be stored here
    output_pdf_path = "Code/article.pdf"
    # Initialize the pipeline
    assembler = AssemblerPipeline(json_path, images_path, output_pdf_path)
    # Set up the environment
    assembler.setup()
    # Run the pipeline
    assembler.run_pipeline()


if __name__ == "__main__":
    run_pipeline()
