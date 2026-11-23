from paddleocr import PaddleOCR
from PIL import Image
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ocr_pretrained = PaddleOCR(rec_model_dir=os.path.join(HERE, "pretrained_model", "inference_model"),use_angle_cls=True,lang='en')
ocr_trained = PaddleOCR(rec_model_dir=os.path.join(HERE, "checkpoints", "inference_model"),use_angle_cls=True,lang='en') # need to run only once to download and load model into memory

# an image to test (have never seen by the OCR) 
img_path = os.path.join(HERE, "test_img.jpg")
image = Image.open(img_path).convert('RGB')
image.show()

print("\n=== OCR Result from Pre-trained Model ===")
result_pretrained = ocr_pretrained.ocr(img_path, cls=True)

for res in result_pretrained[0]:
    print(res)

txts_pretrained = [line[1][0] for line in result_pretrained[0]]
print("\nExtracted Texts:", "".join(txts_pretrained))

print("\n=== OCR Result from Trained Model ===")

result_trained = ocr_trained.ocr(img_path, cls=True)
for res in result_trained[0]:
    print(res)

txts_trained = [line[1][0] for line in result_trained[0]]
print("\nExtracted Texts:", "".join(txts_trained))