# SpecAugment with numpy & scipy

This is SpecAugment code with numpy and scipy.

In my test environment, this code is about 100 time faster than using tensorflow.

## Usage
```
pip install -r requirements.txt
python3 specaugment.py
```

if you want to test your test audio, run this command
```
python3 specaugment.py -i <input file>
```

## Example output

<p align="center">
  <img src="https://github.com/KimJeongSun/SpecAugment_numpy_scipy/blob/master/image/spectrum.png" alt="Example result of spectrogram"/ width=600>
  
  <img src="https://github.com/KimJeongSun/SpecAugment_numpy_scipy/blob/master/image/spectrum_warped.png" alt="Example result of warped spectrogram"/ width=600>
  
  <img src="https://github.com/KimJeongSun/SpecAugment_numpy_scipy/blob/master/image/spectrum_masked.png" alt="Example result of masked spectrogram"/ width=600>
</p>

```
start to SpecAugment 100 times
whole processing time : 0.9117 second
average processing time : 9.12 ms
```


## Reference

1. https://arxiv.org/pdf/1904.08779.pdf
