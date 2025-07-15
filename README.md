# Fine-tune PaddleOCR to Suit My Handwriting

This project was developed as part of **CS3600 - Introduction to Artificial Intelligence** taught by **Professor Alexander Karpekov** during **Summer 2025** at **Georgia Tech**.

## 📌 Project Overview

The goal of this project is to **fine-tune PaddleOCR’s recognition model** to better recognize and transcribe my personal handwriting. While PaddleOCR provides powerful general-purpose text detection and recognition tools, its accuracy may drop significantly when used with highly stylized or non-standard handwriting. This project adapts the model to overcome that limitation by training it on custom-labeled samples of my own handwriting.

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
1. Collected scanned or photographed handwritten text samples.
2. Annotated bounding boxes and corresponding transcriptions in PPOCRLabel.
3. Exported in the format required for PaddleOCR training (`.txt` and image files in `train`/`val` structure).

## 🧠 Fine-Tuning Process

- Used the CRNN-based recognition model (`rec_mv3_none_bilstm_ctc`) as a base.
- Followed PaddleOCR’s training pipeline and configuration.
- Adjusted training parameters for character dictionary, learning rate, and number of epochs to optimize for the limited dataset.

## 📈 Outcomes

- Achieved improved accuracy on handwritten samples compared to the default model.
- Demonstrated the effectiveness of transfer learning and data annotation in OCR adaptation tasks.

## 📚 References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PPOCRLabel Tool](https://github.com/PFCCLab/PPOCRLabel)
- [Fine-Tuning Tutorial](https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6)

---

