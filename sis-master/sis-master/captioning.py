from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import os
from tqdm import tqdm  # Import tqdm for the progress bar

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Define a function to process a folder of images and save the results in a text file
def process_images_in_folder(input_folder, output_file):
    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))]

    results = []
    
    # Use tqdm to create a progress bar
    for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        pixel_values = feature_extractor(images=i_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        # Extract the filename from the image path
        filename = os.path.basename(image_path)

        results.append(f"{filename},{preds[0]}\n")

    with open(output_file, 'w') as f:
        f.writelines(results)

# Specify the input folder containing images and the output text file
input_folder = 'static/img'
output_file = 'captions.txt'

# Process images and save results in the text file
process_images_in_folder(input_folder, output_file)