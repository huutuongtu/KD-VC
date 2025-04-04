# [ICASSP 2025] Voice Conversion for Low-Resource Languages via Knowledge Transfer and Domain-Adversarial Training

[Link to paper and pretrained model](https://drive.google.com/drive/folders/1QIsClIFMgZizeG6Is9WxlWVueUvbUuu7?usp=sharing)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UiCEtZtu0jIe1zy2t86prM9AHwyrvUWc?usp=sharing)


## Pre-requisites
1. Download WavLM-Large and put it under directory 'wavlm/', download checkpoint and put it under directory checkpoint/
2. Install requirements: pip install -r requirements.txt

## Inference :
```
python3 convert.py --hpfile ./checkpoint/vi_200_10_sr/config.json --ptfile ./checkpoint/vi_200_10_sr/G_266000.pth
```

## References:
* https://github.com/OlaWod/FreeVC
* https://github.com/KrishnaDN/Attentive-Statistics-Pooling-for-Deep-Speaker-Embedding

## Citation:
```
@INPROCEEDINGS{10889083,
  author={Tu, Huu Tuong and Thanh Long, Luong and Huan, Vu and Phuong Thao, Nguyen Thi and Van Thang, Nguyen and Cuong, Nguyen Tien and Thi Thu Trang, Nguyen},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Voice Conversion for Low-Resource Languages via Knowledge Transfer and Domain-Adversarial Training}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889083}}
```
