from webcam.webcam_capture import WebcamCapture
from image_classifier.classify import CarClassifier
from article_agent.article_agent import ArticleAgent
from article_agent.agent_pipeline import AgentPipeline
from article_assembler.assembler_pipeline import AssemblerPipeline

import pandas as pd

webcam_imgs_path = "Code/webcam/webcam_images"
#csv = "Code/dataset/sort/reduced_dataset.csv" #needed to get possible brands and body types

csv= "Code/dataset/data/reduced_dataset_adjusted.csv"
model_path = "Code/image_classifier/Resnet50/trained_model_on_test.pth"

json_path = "Code/article_agent/json/output.json" #article agent output
images_path = "Code/article_assembler/tmp/imgs" #generated images will be stored here
output_pdf_path = "Code/article.pdf"
 
model_name = "alexnet" #"visiontransformer", "alexnet" or "resnet"
model_path = "Code/image_classifier/alexnet/alexnet_body-style_epoch80_loss0.04466895014047623_weights.pth" #.pth file

def run_pipeline():
    print("Step 1: Capturing image...")
    webcam = WebcamCapture(webcam_imgs_path)
    webcam.capture_images()

    print("\n Step 2: Classifying image...")
    df = pd.read_csv(csv)
    brand_classes = sorted(df['brand'].unique())
    body_type_classes = sorted(df['body_style'].unique())

    print(f"\n\n\n body_type_classes: {body_type_classes} \n\n\n")

    classifier = CarClassifier(
        model_name=model_name,
        model_path=model_path,
        body_type_classes=body_type_classes
    )

    result = classifier.classify_folder(webcam_imgs_path)
    print(result)

    brand_dummy = "BMW"

    print("Step 3: Creating JSON...")
    agent_pipeline = AgentPipeline(brand=brand_dummy, car_type=result['most_common_body_type'])
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
