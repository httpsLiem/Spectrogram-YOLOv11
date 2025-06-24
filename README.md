# Spectrogram-YOLOv11
This project focuses on enhancing the YOLOv11 object detection architecture to accurately recognize and classify wireless signal spectrums, specifically 5G NR and LTE. Unlike traditional visual object detection, this task involves interpreting spectrogram-like representations of radio frequency signals, requiring specialized model adaptation.

### 🧠 Key Contributions

- ✅ Fine-tuned **YOLOv11** to detect and differentiate between **5G** and **LTE** signal patterns.  
- 📊 Applied **domain-specific preprocessing** to convert raw IQ or frequency data into **visual spectrum representations**.  
- 🧩 Modified **backbone** and **detection heads** for improved feature extraction in signal-based imagery.  
- 📈 Evaluated performance using **mAP@0.5** and **mAP@0.5:0.95** metrics on a **custom-built wireless spectrum dataset**.

### 🚀 Goal

To develop a **lightweight** and **robust AI model** for **real-time signal classification** in **cognitive radio** and **wireless monitoring systems**.

 ## 📊 Comparison:

| Model                | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) |
|----------------------|-------------|------------------|------------|
| YOLOv11              | 94.6        | 88.8             | 9.4        | 
| Spectrogram-YOLOv11  | 96.1        | 90.7             | 6.8        |

> 📝 *Note: The above results were evaluated on a custom 5G/LTE signal spectrum dataset.*
