# Fine-tune PaddleOCR to Suit My Handwriting

This project was developed as part of **CS3600 - Introduction to Artificial Intelligence** taught by **Professor Alexander Karpekov** during **Summer 2025** at **Georgia Tech**.

## 📌 Project Overview

The goal of this project is to **fine-tune PaddleOCR’s recognition model** to better recognize and transcribe my personal handwriting. While PaddleOCR provides powerful general-purpose text detection and recognition tools, its accuracy drops significantly when used with my handwriting, even though I think my hand writing is not that bad. Therefore, I want to train it using my handwriting so that it can recognize my handwriting with high accuracy. 

Inspired by:  
[Fine-Tuning PaddleOCR’s Recognition Model (Anush Som's Tutorial)](https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6)

## 🧰 Tools & Technologies

- **PaddleOCR**: Open-source OCR system developed by PaddlePaddle
- **PPOCRLabel**: GUI tool for labeling OCR datasets ([GitHub Repo](https://github.com/PFCCLab/PPOCRLabel))
- **Python**
- **PaddlePaddle deep learning framework**

## 📂 Data Preparation

The labeled dataset was created using **PPOCRLabel**, which supports text detection and recognition annotations tailored for PaddleOCR training.

Steps:
1. Converted my handwriting pdfs to JPEG
2. Annotated bounding boxes and corresponding transcriptions in PPOCRLabel.
3. Exported in the format required for PaddleOCR training (`.txt` and image files in `output_images`/`crop_img` structure).

## 🧠 Fine-Tuning Process

## 📈 Outcomes

## 📚 References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PPOCRLabel Tool](https://github.com/PFCCLab/PPOCRLabel)
- [Fine-Tuning Tutorial](https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6)

---

