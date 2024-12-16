print("**** Loading imports. Please wait ****")

from webcam.webcam_capture import WebcamCapture
from image_classifier.classify import CarClassifier
from article_agent.agent_pipeline import AgentPipeline
from article_assembler.assembler_pipeline import AssemblerPipeline

import pandas as pd

webcam_imgs_path = "Code/webcam/webcam_images"

# csv has 5 body types and 22 brands
csv= "Code/dataset/data/reduced_dataset_adjusted.csv" #TODO -ferrari

json_path = "Code/article_agent/json/output.json" #article agent output
images_path = "Code/article_assembler/tmp/imgs" #generated images will be stored here
output_pdf_path = "Code/article.pdf"
 
model_name = "alexnet" #"vit", "alexnet" or "resnet"
# weights with 27(5+22) classes needed
model_path = "Code/image_classifier/alexnet/alexnet_27classes.pth" #.pth file


def run_pipeline():
    print("\n Step 1: Capturing image...")
    webcam = WebcamCapture(webcam_imgs_path)
    webcam.capture_images()

    print("\n Step 2: Classifying image...")
    df = pd.read_csv(csv)
    brand_classes = sorted(df['brand'].unique())
    body_type_classes = sorted(df['body_style'].unique())

    # print(f"\n body_type_classes ({len(body_type_classes)}): {body_type_classes}")
    # print(f"\n brand_classes ({len(brand_classes)}): {brand_classes} \n")

    classifier = CarClassifier(
        model_name=model_name,
        model_path=model_path,
        body_type_classes=body_type_classes,
        brand_classes=brand_classes
    )

    result = classifier.classify_folder(webcam_imgs_path)
    car_type = result['most_common_body_type']
    brand = result['most_common_brand']
   
    # Get probabilities from first prediction (as they're similar)
    first_pred = result['all_predictions'][0]
    brand_probs = first_pred['brand_probabilities']
    body_probs = first_pred['body_type_probabilities']
    
    # Find highest probabilities
    brand_confidence = float(max(brand_probs)) * 100
    body_confidence = float(max(body_probs)) * 100
    
    print("\n=== Classification Results ===")
    print(f"Brand:     {brand:<10} (confidence: {brand_confidence:.1f}%)")
    print(f"Body Type: {car_type:<10} (confidence: {body_confidence:.1f}%)")
    print("===========================\n")


    print("\n Step 3: Creating JSON... \n")
    agent_pipeline = AgentPipeline(brand, car_type)
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
