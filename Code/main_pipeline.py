from webcam.webcam_capture import WebcamCapture
from image_classifier.classify import CarClassifier
from article_agent.article_agent import ArticleAgent
from article_agent.agent_pipeline import AgentPipeline
from article_assembler.assembler_pipeline import AssemblerPipeline

import pandas as pd

webcam_imgs_path = "Code/webcam/webcam_images"
csv = "Code/dataset/data/reduced_dataset_adjusted.csv" #needed to get possible brands and body types
model_path = "Code/image_classifier/Resnet50/trained_model_on_test.pth"

json_path = "Code/article_agent/json/output.json" #article agent output
images_path = "Code/article_assembler/tmp/imgs" #generated images will be stored here
output_pdf_path = "Code/article.pdf"

def run_pipeline():
    print("Step 1: Capturing image...")
    webcam = WebcamCapture(webcam_imgs_path)
    webcam.capture_images()

    print("\n Step 2: Classifying image...")
    df = pd.read_csv(csv)
    brand_classes = sorted(df['brand'].unique())
    body_type_classes = sorted(df['body_style'].unique())

    classifier = CarClassifier(
        model_path=model_path,
        brand_classes=brand_classes,
        body_type_classes=body_type_classes
    )

    result = classifier.classify(webcam_imgs_path)
    print(f"Brand: {result['brand']}")
    print(f"Body type: {result['body_type']}")


    print("Step 3: Creating JSON...")
    agent_pipeline = AgentPipeline(brand=result['brand'], car_type=result['body_type'])
    response = agent_pipeline()
    # save json object to a file
    with open(json_path, "w") as f:
        f.write(response)


    print("Step 4: Assembling article...")
    # Initialize the pipeline
    assembler = AssemblerPipeline(json_path, images_path, output_pdf_path)
    # Set up the environment
    assembler.setup()
    # Run the pipeline
    assembler.run_pipeline()


if __name__ == "__main__":
    run_pipeline()
