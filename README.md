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
python3 tools/export_model.py   -c /path/to/project/folder/pretrained_models/en_PP-OCRv3_mobile_rec.yml   -o Global.pretrained_model=/path/to/project/folder/checkpoints/v3_en_mobile/latest    Global.save_inference_dir=/path/to/project/folder/Fine-tuning_an_OCR/checkpoints/inference_model
```

## Outcomes

I run `demo.py` to output the extracted texts from the `test.jpg` image by both the initial model and the fine-tuned model (both have never seen this image). Here is the terminal output:

```
=== OCR Result from Pre-trained Model ===
Opening in existing browser session.
[2025/07/24 21:13:07] ppocr DEBUG: dt_boxes num : 13, elapsed : 0.2664971351623535
[2025/07/24 21:13:07] ppocr DEBUG: cls num  : 13, elapsed : 0.04652094841003418
[2025/07/24 21:13:07] ppocr DEBUG: rec_res num  : 13, elapsed : 0.38838863372802734
[[[21.0, 85.0], [333.0, 85.0], [333.0, 125.0], [21.0, 125.0]], (')Iqoreali%m :', 0.7874861359596252)]
[[[76.0, 148.0], [662.0, 151.0], [662.0, 200.0], [76.0, 198.0]], ('Wonidbut they contain', 0.9432706236839294)]
[[[638.0, 149.0], [1321.0, 144.0], [1322.0, 200.0], [638.0, 205.0]], ('meaninqY9regardingour wo9rld', 0.8459545373916626)]
[[[5.0, 431.0], [1362.0, 431.0], [1362.0, 495.0], [5.0, 495.0]], (') "Storouy Night:teflct Van frogh\'xemotionalxtate duing', 0.8527947664260864)]
[[[1368.0, 442.0], [1536.0, 451.0], [1535.0, 484.0], [1366.0, 475.0]], ('hiYtime', 0.6358948349952698)]
[[[79.0, 509.0], [586.0, 511.0], [586.0, 559.0], [78.0, 556.0]], ('at St.Paul -de- Mauol', 0.9045497179031372)]
[[[605.0, 518.0], [748.0, 518.0], [748.0, 561.0], [605.0, 561.0]], ('a&4lum', 0.6818017959594727)]
[[[83.0, 662.0], [276.0, 662.0], [276.0, 702.0], [83.0, 702.0]], ('Quextion%', 0.9229845404624939)]
[[[100.0, 724.0], [859.0, 716.0], [860.0, 777.0], [100.0, 785.0]], ('Whatdo you think about thix & cene ?', 0.9077663421630859)]
[[[364.0, 792.0], [1171.0, 789.0], [1171.0, 846.0], [364.0, 848.0]], ('thinkthe an(mato9Waqt totell ux 2', 0.9051201343536377)]
[[[112.0, 806.0], [207.0, 806.0], [207.0, 841.0], [112.0, 841.0]], ('What', 0.8156619071960449)]
[[[260.0, 818.0], [286.0, 818.0], [286.0, 839.0], [260.0, 839.0]], ('do', 0.8976622819900513)]

Extracted Texts: )Iqoreali%m :Wonidbut they containmeaninqY9regardingour wo9rld) "Storouy Night:teflct Van frogh'xemotionalxtate duinghiYtimeat St.Paul -de- Mauola&4lumQuextion%Whatdo you think about thix & cene ?thinkthe an(mato9Waqt totell ux 2Whatdo

=== OCR Result from Trained Model ===
[2025/07/24 21:13:07] ppocr DEBUG: dt_boxes num : 13, elapsed : 0.14784693717956543
[2025/07/24 21:13:07] ppocr DEBUG: cls num  : 13, elapsed : 0.020457983016967773
[2025/07/24 21:13:08] ppocr DEBUG: rec_res num  : 13, elapsed : 0.29923224449157715
[[[21.0, 85.0], [333.0, 85.0], [333.0, 125.0], [21.0, 125.0]], ('Iralism:', 0.9266902208328247)]
[[[76.0, 148.0], [662.0, 151.0], [662.0, 200.0], [76.0, 198.0]], ('world but they contain', 0.9846042990684509)]
[[[638.0, 149.0], [1321.0, 144.0], [1322.0, 200.0], [638.0, 205.0]], ('meanings regarding our world', 0.9595417976379395)]
[[[5.0, 431.0], [1362.0, 431.0], [1362.0, 495.0], [5.0, 495.0]], (') "Starry Night: reflect Van Gogh\'s emotional state during', 0.9540849328041077)]
[[[1368.0, 442.0], [1536.0, 451.0], [1535.0, 484.0], [1366.0, 475.0]], ('his time', 0.9089252948760986)]
[[[79.0, 509.0], [586.0, 511.0], [586.0, 559.0], [78.0, 556.0]], ('at St.faul-de-Mausole', 0.9411256909370422)]
[[[605.0, 518.0], [748.0, 518.0], [748.0, 561.0], [605.0, 561.0]], ('asylum', 0.9986608028411865)]
[[[19.0, 655.0], [53.0, 667.0], [41.0, 700.0], [7.0, 688.0]], (')', 0.5024682879447937)]
[[[83.0, 662.0], [276.0, 662.0], [276.0, 702.0], [83.0, 702.0]], ('Question8', 0.9227574467658997)]
[[[100.0, 724.0], [859.0, 716.0], [860.0, 777.0], [100.0, 785.0]], ('What do you think about this scene?', 0.9948091506958008)]
[[[364.0, 792.0], [1171.0, 789.0], [1171.0, 846.0], [364.0, 848.0]], ('think the animator want to tell us ?', 0.9607620239257812)]
[[[112.0, 806.0], [207.0, 806.0], [207.0, 841.0], [112.0, 841.0]], ('What', 0.9922924637794495)]
[[[260.0, 818.0], [286.0, 818.0], [286.0, 839.0], [260.0, 839.0]], ('do', 0.9385541677474976)]

Extracted Texts: Iralism:world but they containmeanings regarding our world) "Starry Night: reflect Van Gogh's emotional state duringhis timeat St.faul-de-Mausoleasylum)Question8What do you think about this scene?think the animator want to tell us ?Whatdo
```


## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PPOCRLabel Tool](https://github.com/PFCCLab/PPOCRLabel)
- [Fine-Tuning Tutorial](https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6)

---

