# Fine-tune PaddleOCR to Suit My Handwriting

This project was developed as part of **CS3600 - Introduction to Artificial Intelligence** taught by **Professor Alexander Karpekov** during **Summer 2025** at **Georgia Tech**.

## Project Overview

The goal of this project is to **fine-tune PaddleOCR’s recognition model** to better recognize and transcribe my personal handwriting. While PaddleOCR provides powerful general-purpose text detection and recognition tools, its accuracy drops significantly when used with my handwriting, even though I think my hand writing is not that bad. Therefore, I want to train it using my handwriting so that it can recognize my handwriting with high accuracy. 

Inspired by:  
[Fine-Tuning PaddleOCR’s Recognition Model (Anush Som's Tutorial)](https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6)

## Tools & Technologies

- **PaddleOCR**: Open-source OCR system developed by PaddlePaddle
- **PPOCRLabel**: GUI tool for labeling OCR datasets ([GitHub Repo](https://github.com/PFCCLab/PPOCRLabel))
- **Python**
- **PaddlePaddle deep learning framework**

## Data Preparation

The labeled dataset was created using **PPOCRLabel**, which supports text detection and recognition annotations tailored for PaddleOCR training.

Steps:
1. Converted my handwriting pdfs to JPEG (_train_test_split.py_)
2. Annotated bounding boxes and corresponding transcriptions in PPOCRLabel. (done by PPOCRLabel)
3. Exported in the format required for PaddleOCR training (`rec_gt.txt` in `output_images` and image files in `output_images`/`crop_img`).
4. Corrected translations in `rec_gt.txt` using ChatGPT API, output `data.txt` in `dataset` (_preprocessing.py_)
5. Splitted `data.txt` into `train.txt` and `test.txt`

## Fine-Tuning Process
The configurations of the fine-tuning process is in `pretrain_model/en_PP-OCRv3_mobile_rec.yml`

To train the model to recognize my hand-writings, in the terminal, I will navigate to the `PaddleOCR` folder directory of training virtual environment, then execute this command:

```
python3 tools/train.py   -c  path/to/project/folder/pretrained_model/en_PP-OCRv3_mobile_rec.yml   -o Global.pretrained_model=path/to/project/folder/pretrained_model/en_PP-OCRv3_rec_train/best_accuracy
```

To output the trained model, I will execute this command (also in the `PaddleOCR` directory):

```
python3 tools/export_model.py   -c /path/to/project/folder/pretrain_models/en_PP-OCRv3_mobile_rec.yml   -o Global.pretrained_model=/path/to/project/folder/checkpoints/v3_en_mobile/latest    Global.save_inference_dir=/path/to/project/folder/Fine-tuning_an_OCR/checkpoints/inference_model
```

## Outcomes

## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PPOCRLabel Tool](https://github.com/PFCCLab/PPOCRLabel)
- [Fine-Tuning Tutorial](https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6)

---

