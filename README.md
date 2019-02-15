# Audio-Classification-With-Deep-Learning

Bu proje [micah5](https://github.com/micah5/pyAudioClassification) adlı kullanıcının bizlere sunduğu kütüphanenin parçalanıp editlenip bir proje haline dönüştürülmesi ile oluşmuştur. Hızlı bir kullanım için üstte verdiğim linki kullanabilirsiniz. Modeller üzerinde değişiklik yapmak için, data üzerinde manipülasyonlar yapmak içinse attığım .ipynb uzantılı dosyayı kullanabilirsiniz.

### Requirements
* __Python 3__
* Keras
* Tensorflow
* librosa
* NumPy
* Soundfile
* tqdm
* matplotlib

### Step 1: Data

Ses veriniz herhangi bir uzantıya (.mp3,.wav,.ogg) sahip olabilir. Ben .ogg kullanılmasını öneriyorum. Karışık uzatılarda kullanabilirsiniz. Data klasörünüzün aşağıdaki hiyerarşide olması yeterlidir. Data için herhangi bir csv dosyasına ihtiyacınız yoktur.

```
data/
├── cat/
│   ├── cat1.ogg
│   ├── cat2.ogg
│   ├── cat3.wav
│   └── cat4.wav
└── dog/
    ├── dog1.ogg
    ├── dog2.ogg
    ├── dog3.wav
    └── dog4.wav
```
### Step 2: Training

Aşağıdaki parametreleri train fonksyonuna gönderebilirsiniz. features, labels parametrelerini kesinlikle göndermeniz gerekmektedir. Diğer parametrelere deafult değerler atanmıştır.

* `num_classes`: Sınıf sayınız. Eğer parametre olarak göndermezseniz otomatik olarak hesaplanmaktadır.


* `epochs`: Epoch sayınız. Default `50`.

* `lr`: Learning rate.  Default  `0.01`.

* `optimiser`: Herhangi birini seçebilirsiniz. [these](https://keras.io/optimizers/). Default `'SGD'`.

* `print_summary`: Modeliniz hakkında bilgiler verir. Default  `False`.

* `loss_type`: Classification type. Default is `categorical` for >2 classes, and `binary` otherwise.

You can add any of these as optional arguments, for example `train(features, labels, lr=0.05)`

---
Eğer Modelenizi kaydetmek ve sonradan tekrar kullanmak isterseniz aşağıdaki kod parçacığını eklemeniz gerekmektedir.

```python
from keras.models import load_model

model.save('my_model.h5')
model = load_model('my_model.h5')
```

### Step 3: Prediction

leaderboard bütün sınıflara olan üyelik derecesini belirtir. Aşağıdaki gibi.

```
1. Cow 100.0% (index 2)
2. Rooster 0.0% (index 0)
3. Frog 0.0% (index 3)
4. Pig 0.0% (index 1)
```

```python
pred = predict(model, '/home/munir/Desktop/pyAudioClassification-master/example/cow_test.wav')
print_leaderboard(pred, '/home/munir/Desktop/pyAudioClassification-master/example/data')
```


```python
pred = predict(model, <data_path>)
```

Your `<data_path>` should point to a new, untested audio file.

## References
* Large parts of the code (particularly the feature extraction) are based on [mtobeiyf/audio-classification](https://github.com/mtobeiyf/audio-classification)
* [panotti](https://github.com/drscotthawley/panotti)
