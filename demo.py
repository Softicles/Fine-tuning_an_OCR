from paddleocr import PaddleOCR
from PIL import Image
import os

ocr_pretrained = PaddleOCR(rec_model_dir="/home/thinh/Desktop/Fine-tuning_an_OCR/pretrained_model/inference_model",use_angle_cls=True,lang='en')
ocr_trained = PaddleOCR(rec_model_dir="/home/thinh/Desktop/Fine-tuning_an_OCR/checkpoints/inference_model",use_angle_cls=True,lang='en') # need to run only once to download and load model into memory

# an image to test 
img_path = '/home/thinh/Desktop/Fine-tuning_an_OCR/output_images/crop_img/assembly_hw_227_crop_15.jpg'
image = Image.open(img_path).convert('RGB')
image.show()

print("\n=== OCR Result from Pre-trained Model ===")
result_pretrained = ocr_pretrained.ocr(img_path, cls=True)

for res in result_pretrained[0]:
    print(res)

txts_pretrained = [line[1][0] for line in result_pretrained[0]]
print("\nExtracted Texts:", txts_pretrained)

print("\n=== OCR Result from Trained Model ===")

result_trained = ocr_trained.ocr(img_path, cls=True)
for res in result_trained[0]:
    print(res)

txts_trained = [line[1][0] for line in result_trained[0]]
print("\nExtracted Texts:", txts_trained)