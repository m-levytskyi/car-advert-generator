import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_weights():
    brands_path = "Code/image_classifier/alexnet/alexnet_epoch15_bestValLoss.pth"
    body_path = "Code/image_classifier/alexnet/alexnet_body-style_epoch80_loss0.04466895014047623_weights.pth"
    output_path = "Code/image_classifier/alexnet/alexnet_27classes.pth"
    
    try:
        brands_dict = torch.load(brands_path, map_location=torch.device('cpu'))
        body_dict = torch.load(body_path, map_location=torch.device('cpu'))
        
        # Use correct layer names
        classifier_key = 'step.22.weight'
        classifier_bias_key = 'step.22.bias'
        
        logger.info(f"Using classifier key: {classifier_key}")
        
        # Get weights
        brands_weights = brands_dict[classifier_key]
        brands_bias = brands_dict[classifier_bias_key]
        body_weights = body_dict[classifier_key]
        body_bias = body_dict[classifier_bias_key]
        
        logger.info(f"Original shapes - Brands: {brands_weights.shape}, Body: {body_weights.shape}")
        
        # Remove Ferrari class
        brands_weights = brands_weights[:-1]
        brands_bias = brands_bias[:-1]
        
        logger.info(f"After Ferrari removal - Brands: {brands_weights.shape}, Body: {body_weights.shape}")
        
        # Merge weights
        new_weights = torch.cat([brands_weights, body_weights], dim=0)
        new_bias = torch.cat([brands_bias, body_bias], dim=0)
        
        logger.info(f"Final shapes - Weights: {new_weights.shape}, Bias: {new_bias.shape}")
        
        # Update dict and save
        brands_dict[classifier_key] = new_weights
        brands_dict[classifier_bias_key] = new_bias
        torch.save(brands_dict, output_path)
        
        logger.info(f"Successfully saved merged weights to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during weight merging: {str(e)}")
        raise

if __name__ == "__main__":
    merge_weights()



# RESNET
# 23 classes - Resnet50/resnet_epoch49_bestTrainLoss_bestValLoss_bestAccuracy.pth
# error - Resnet50/trained_model_on_test.pth

# ALEXNET
# 23 classes - Code/image_classifier/alexnet/alexnet_epoch89_bestTrainLoss_bestValAccuracy.pth
# 5 classes - Code/image_classifier/alexnet/alexnet_body-style_epoch80_loss0.04466895014047623_weights.pth
# 23 - Code/image_classifier/alexnet/alexnet_epoch15_bestValLoss.pth

# VIT
# 23 - Code/image_classifier/VisualTransformer/visiontransformer_epoch49_best_ValLoss.pth