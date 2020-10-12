# Enhanced-JPEG-compression

This is the official implementation of the following [paper](https://ieeexplore.ieee.org/abstract/document/9093951?casa_token=YOD8BLtooH0AAAAA:az8dFriaKCkn_iDI-de5MKg2MKsxwRRVWXg3jAvnlRzIV8cCeAv0BDxoislhHWPK-13rJqMesr4):

> *Deep Residual Learning based Enhanced JPEG Compression in the Internet of Things*
>
> Han Qiu, Qinkai Zheng, Gerard Memmi, Jialiang Lu, Meikang Qiu, Bhavani Thuraisingham
>
> *Abstract*: With the development of big data and network technology, there are more use cases like edge computing that require more secure and efficient multimedia big data transmission. Data compression methods can help to achieve many tasks like providing data integrity, protection, and efficient transmission. Classical multimedia big data compression relies on methods like the spatial-frequency transformation for compressing with loss. Recent approaches use Deep Learning (DL) to further explore the limit of the data compression methods in communication constrained scenarios like the Internet of Things (IoT). In this paper, we propose a novel method to significantly enhance the transformation-based compression standards like JPEG by transmitting much fewer data of one image at the sender's end. At the receiver's end, we propose a two-step method by combining the state-of-the-art signal processing based recovery method with a deep residual learning model to recover the transmitted data. Therefore, in the scenarios like IoT, the sender like edge device can transmit only 60% data of the original JPEG image without any additional calculation steps but the image quality can still be recovered at the receiver's end like cloud servers with PSNR more than 31 dB.

If you have any question, please raise an issue or contact ```qinkai.zheng1028@gmail.com```. 

## DEMO



## Requirements

* cv2==3.4.2
* numpy==1.15.4

## Citation

```
@article{qiu2020deep,
  title={Deep Residual Learning based Enhanced JPEG Compression in the Internet of Things},
  author={Qiu, Han and Zheng, Qinkai and Memmi, Gerard and Lu, Jialiang and Qiu, Meikang and Thuraisingham, Bhavani},
  journal={IEEE Transactions on Industrial Informatics},
  year={2020},
  publisher={IEEE}
}
```



