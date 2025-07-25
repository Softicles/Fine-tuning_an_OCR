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

To train the model to recognize my hand-writings, in the terminal, I navigated to the `PaddleOCR` folder directory of training virtual environment, then execute this command:

```
python3 tools/train.py   -c  path/to/project/folder/pretrained_model/en_PP-OCRv3_mobile_rec.yml   -o Global.pretrained_model=path/to/project/folder/pretrained_model/en_PP-OCRv3_rec_train/best_accuracy
```

To output the trained model, I executed this command (also in the `PaddleOCR` directory):

```
python3 tools/export_model.py   -c /path/to/project/folder/pretrained_models/en_PP-OCRv3_mobile_rec.yml   -o Global.pretrained_model=/path/to/project/folder/checkpoints/v3_en_mobile/latest    Global.save_inference_dir=/path/to/project/folder/Fine-tuning_an_OCR/checkpoints/inference_model
```

I trained the model for 10 epochs, here is the details:

```
[2025/07/22 00:14:40] ppocr INFO: During the training process, after the 0th iteration, an evaluation is run every 2000 iterations
[2025/07/22 00:22:10] ppocr INFO: epoch: [1/10], lr: 0.000024, acc: 0.109375, CTCLoss: 36.289278, SARLoss: 2.587582, loss: 38.897942, eta: 4:30:02
[2025/07/22 00:29:33] ppocr INFO: epoch: [1/10], lr: 0.000051, acc: 0.117187, CTCLoss: 33.585308, SARLoss: 2.507226, loss: 36.276060, eta: 4:20:36
[2025/07/22 00:36:56] ppocr INFO: epoch: [1/10], lr: 0.000105, acc: 0.152344, CTCLoss: 27.627853, SARLoss: 2.179113, loss: 29.816208, eta: 4:12:28
[2025/07/22 00:42:07] ppocr INFO: epoch: [1/10], lr: 0.000143, acc: 0.164062, CTCLoss: 21.134651, SARLoss: 1.886903, loss: 22.924883, eta: 4:07:03
[2025/07/22 00:42:07] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 00:44:26] ppocr INFO: epoch: [2/10], lr: 0.000159, acc: 0.167969, CTCLoss: 19.733342, SARLoss: 1.798387, loss: 21.525607, eta: 4:05:32
[2025/07/22 00:51:55] ppocr INFO: epoch: [2/10], lr: 0.000214, acc: 0.214844, CTCLoss: 14.636453, SARLoss: 1.576685, loss: 16.263607, eta: 3:58:26
[2025/07/22 00:59:28] ppocr INFO: epoch: [2/10], lr: 0.000268, acc: 0.253906, CTCLoss: 10.846145, SARLoss: 1.306704, loss: 12.149431, eta: 3:51:26
[2025/07/22 01:06:59] ppocr INFO: epoch: [2/10], lr: 0.000322, acc: 0.269531, CTCLoss: 9.325074, SARLoss: 1.159356, loss: 10.492249, eta: 3:44:13
[2025/07/22 01:09:59] ppocr INFO: epoch: [2/10], lr: 0.000343, acc: 0.320312, CTCLoss: 9.102239, SARLoss: 1.131532, loss: 10.202122, eta: 3:41:16
[2025/07/22 01:09:59] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 01:14:36] ppocr INFO: epoch: [3/10], lr: 0.000376, acc: 0.335937, CTCLoss: 8.193659, SARLoss: 1.061532, loss: 9.236146, eta: 3:37:16
[2025/07/22 01:22:14] ppocr INFO: epoch: [3/10], lr: 0.000430, acc: 0.378906, CTCLoss: 7.187253, SARLoss: 1.000582, loss: 8.110877, eta: 3:30:12
[2025/07/22 01:29:53] ppocr INFO: epoch: [3/10], lr: 0.000484, acc: 0.371094, CTCLoss: 6.819806, SARLoss: 0.941449, loss: 7.830772, eta: 3:23:06
[2025/07/22 01:37:39] ppocr INFO: epoch: [3/10], lr: 0.000538, acc: 0.386719, CTCLoss: 6.204155, SARLoss: 0.864264, loss: 7.068559, eta: 3:16:09
[2025/07/22 01:38:27] ppocr INFO: epoch: [3/10], lr: 0.000543, acc: 0.386719, CTCLoss: 6.159812, SARLoss: 0.858591, loss: 7.047923, eta: 3:15:29
[2025/07/22 01:38:27] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 01:38:27] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/iter_epoch_3
[2025/07/22 01:45:34] ppocr INFO: epoch: [4/10], lr: 0.000592, acc: 0.410156, CTCLoss: 5.769098, SARLoss: 0.837742, loss: 6.584170, eta: 3:09:23
[2025/07/22 01:53:14] ppocr INFO: epoch: [4/10], lr: 0.000646, acc: 0.421875, CTCLoss: 5.638283, SARLoss: 0.787253, loss: 6.426825, eta: 3:01:58
[2025/07/22 02:00:59] ppocr INFO: epoch: [4/10], lr: 0.000700, acc: 0.437500, CTCLoss: 5.374988, SARLoss: 0.763219, loss: 6.136869, eta: 2:54:39
[2025/07/22 02:07:14] ppocr INFO: epoch: [4/10], lr: 0.000743, acc: 0.441406, CTCLoss: 5.205883, SARLoss: 0.739347, loss: 5.987385, eta: 2:48:50
[2025/07/22 02:07:14] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 02:08:52] ppocr INFO: epoch: [5/10], lr: 0.000754, acc: 0.449219, CTCLoss: 5.205883, SARLoss: 0.731270, loss: 5.987385, eta: 2:47:29
[2025/07/22 02:16:40] ppocr INFO: epoch: [5/10], lr: 0.000808, acc: 0.460937, CTCLoss: 4.865515, SARLoss: 0.688503, loss: 5.541889, eta: 2:40:07
[2025/07/22 02:24:27] ppocr INFO: epoch: [5/10], lr: 0.000862, acc: 0.464844, CTCLoss: 4.956313, SARLoss: 0.681412, loss: 5.594249, eta: 2:32:41
[2025/07/22 02:32:15] ppocr INFO: epoch: [5/10], lr: 0.000916, acc: 0.468750, CTCLoss: 4.741332, SARLoss: 0.655479, loss: 5.443601, eta: 2:25:13
[2025/07/22 02:36:12] ppocr INFO: epoch: [5/10], lr: 0.000943, acc: 0.476562, CTCLoss: 4.741332, SARLoss: 0.654965, loss: 5.443601, eta: 2:21:31
[2025/07/22 02:36:12] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 02:40:13] ppocr INFO: epoch: [6/10], lr: 0.000970, acc: 0.480469, CTCLoss: 4.279113, SARLoss: 0.643472, loss: 4.922718, eta: 2:17:53
[2025/07/22 02:48:11] ppocr INFO: epoch: [6/10], lr: 0.000998, acc: 0.492187, CTCLoss: 4.369420, SARLoss: 0.646944, loss: 5.023855, eta: 2:10:29
[2025/07/22 02:56:09] ppocr INFO: epoch: [6/10], lr: 0.000996, acc: 0.488281, CTCLoss: 4.493403, SARLoss: 0.636215, loss: 5.161893, eta: 2:03:02
[2025/07/22 03:04:11] ppocr INFO: epoch: [6/10], lr: 0.000989, acc: 0.511719, CTCLoss: 4.341220, SARLoss: 0.628070, loss: 5.023137, eta: 1:55:34
[2025/07/22 03:05:49] ppocr INFO: epoch: [6/10], lr: 0.000987, acc: 0.515625, CTCLoss: 4.288525, SARLoss: 0.628070, loss: 4.954180, eta: 1:54:06
[2025/07/22 03:05:50] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 03:05:50] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/iter_epoch_6
[2025/07/22 03:12:33] ppocr INFO: epoch: [7/10], lr: 0.000979, acc: 0.531250, CTCLoss: 3.924403, SARLoss: 0.606863, loss: 4.547284, eta: 1:48:16
[2025/07/22 03:21:12] ppocr INFO: epoch: [7/10], lr: 0.000965, acc: 0.519531, CTCLoss: 3.924403, SARLoss: 0.594110, loss: 4.532217, eta: 1:41:02
[2025/07/22 03:29:43] ppocr INFO: epoch: [7/10], lr: 0.000947, acc: 0.511719, CTCLoss: 4.102991, SARLoss: 0.579895, loss: 4.682191, eta: 1:33:37
[2025/07/22 03:37:14] ppocr INFO: epoch: [7/10], lr: 0.000929, acc: 0.515625, CTCLoss: 3.961594, SARLoss: 0.564712, loss: 4.565799, eta: 1:26:48
[2025/07/22 03:37:14] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 03:38:08] ppocr INFO: epoch: [8/10], lr: 0.000927, acc: 0.519531, CTCLoss: 3.874051, SARLoss: 0.555200, loss: 4.439003, eta: 1:26:05
[2025/07/22 03:46:41] ppocr INFO: epoch: [8/10], lr: 0.000903, acc: 0.519531, CTCLoss: 3.849123, SARLoss: 0.553709, loss: 4.363479, eta: 1:18:31
[2025/07/22 03:55:15] ppocr INFO: epoch: [8/10], lr: 0.000877, acc: 0.527344, CTCLoss: 3.617082, SARLoss: 0.535218, loss: 4.143805, eta: 1:10:54
[2025/07/22 04:03:57] ppocr INFO: epoch: [8/10], lr: 0.000847, acc: 0.531250, CTCLoss: 3.738000, SARLoss: 0.530616, loss: 4.278790, eta: 1:03:15
[2025/07/22 04:09:10] ppocr INFO: epoch: [8/10], lr: 0.000829, acc: 0.531250, CTCLoss: 3.664553, SARLoss: 0.505472, loss: 4.204647, eta: 0:58:37
[2025/07/22 04:09:11] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 04:12:46] ppocr INFO: epoch: [9/10], lr: 0.000816, acc: 0.535156, CTCLoss: 3.676602, SARLoss: 0.505472, loss: 4.204647, eta: 0:55:33
[2025/07/22 04:21:41] ppocr INFO: epoch: [9/10], lr: 0.000782, acc: 0.570312, CTCLoss: 3.365801, SARLoss: 0.497441, loss: 3.834283, eta: 0:47:48
[2025/07/22 04:30:31] ppocr INFO: epoch: [9/10], lr: 0.000746, acc: 0.550781, CTCLoss: 3.385992, SARLoss: 0.486614, loss: 3.862138, eta: 0:39:58
[2025/07/22 04:39:00] ppocr INFO: epoch: [9/10], lr: 0.000708, acc: 0.550781, CTCLoss: 3.326879, SARLoss: 0.488899, loss: 3.800971, eta: 0:32:02
[2025/07/22 04:41:31] ppocr INFO: epoch: [9/10], lr: 0.000696, acc: 0.542969, CTCLoss: 3.368734, SARLoss: 0.490650, loss: 3.864728, eta: 0:29:39
[2025/07/22 04:41:32] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 04:41:32] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/iter_epoch_9
[2025/07/22 04:47:24] ppocr INFO: epoch: [10/10], lr: 0.000669, acc: 0.546875, CTCLoss: 3.298858, SARLoss: 0.479036, loss: 3.768306, eta: 0:24:03
[2025/07/22 04:55:40] ppocr INFO: epoch: [10/10], lr: 0.000628, acc: 0.542969, CTCLoss: 3.445397, SARLoss: 0.467953, loss: 3.905267, eta: 0:16:03
[2025/07/22 05:04:06] ppocr INFO: epoch: [10/10], lr: 0.000587, acc: 0.566406, CTCLoss: 3.432148, SARLoss: 0.458043, loss: 3.905267, eta: 0:08:02
[2025/07/22 05:12:39] ppocr INFO: epoch: [10/10], lr: 0.000545, acc: 0.574219, CTCLoss: 3.422845, SARLoss: 0.444814, loss: 3.874272, eta: 0:00:00
[2025/07/22 05:12:40] ppocr INFO: save model in .../Fine-tuning_an_OCR/checkpoints/v3_en_mobile/latest
[2025/07/22 05:12:40] ppocr INFO: best metric, acc: 0, is_float16: False
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
![test_img](https://github.com/user-attachments/assets/bf1e03d7-faf7-440b-8b52-abf24b5dcde2)

## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PPOCRLabel Tool](https://github.com/PFCCLab/PPOCRLabel)
- [Fine-Tuning Tutorial](https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6)

---

