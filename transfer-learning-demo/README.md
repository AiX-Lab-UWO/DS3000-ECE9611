# Transfer learning and fine-tuning with CNNs

Teach a CNN to recognise 5 kinds of flowers (daisy, dandelion, rose, sunflower, tulip) from a few hundred photos, by reusing networks that were pretrained on ImageNet (1.28 million photos, 1000 classes).

`transfer_learning_demo.ipynb` has three parts:

1. **Pretrained CNNs as frozen feature extractors.** ResNet50, MobileNetV2 and EfficientNetB0 each run every photo once, and only a new output layer is trained on what they produce. The cell compares their accuracy and speed.
2. **Freezing.** MobileNetV2 is frozen with `base.trainable = False`. The cell shows the number of trainable weights before and after, and checks after training that the pretrained weights did not change while the new output layer did.
3. **Fine-tuning.** The last 30 layers of MobileNetV2 are unfrozen and trained further with a learning rate 100 times smaller, so the pretrained weights are adjusted rather than overwritten.

## Run

```
pip install -r requirements.txt
```

Open `transfer_learning_demo.ipynb` and run the cells in order. The first run downloads the flowers dataset (about 230 MB) and the pretrained weights (about 140 MB) into `~/.keras`.

It runs on a CPU in a few minutes and much faster on a GPU. To use a free GPU in Google Colab, upload the notebook and choose Runtime > Change runtime type > T4 GPU.

## What you will see

| Model | Pretrained weights (frozen) | Weights trained |
|---|---|---|
| ResNet50 | 23,587,712 | 10,245 |
| MobileNetV2 | 2,257,984 | 6,405 |
| EfficientNetB0 | 4,049,571 | 6,405 |

* **Freezing:** 2,223,872 trainable weights before `base.trainable = False`, 0 after. After training, the check prints `True` for "pretrained weights unchanged" and `True` for "new output layer changed".
* **Fine-tuning:** 1,532,485 trainable weights (the last 30 layers plus the output layer), and the validation accuracy before and after.

MobileNetV2 has 2,257,984 weights in total, but only 2,223,872 are trainable even before freezing. The other 34,112 are BatchNorm running statistics, which are never trained by gradient descent.

## Things to try

1. Use all the photos: replace `train.take(15).cache(), val.take(8).cache()` with `train.cache(), val.cache()` (2,936 training photos instead of 480). How much does the accuracy improve when the network is pretrained?
2. In the fine-tuning cell, unfreeze more layers (`base.layers[:-60]`) or use a learning rate of `1e-4`. Compare the training and validation curves: is the model starting to overfit?
3. Compare the time per photo and the accuracy of the three models. Which one would you deploy on a Raspberry Pi or a Jetson?

## Using an NVIDIA GPU on Windows

TensorFlow 2.11 and newer cannot use an NVIDIA GPU on native Windows; TensorFlow 2.10 was the last version that can. Set up a separate environment for it once, in the **Anaconda Prompt**:

```
conda create -n tf-gpu python=3.10 -y
conda activate tf-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install "tensorflow<2.11" "numpy<2" matplotlib ipykernel
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -m ipykernel install --user --name tf-gpu --display-name "Python (TF GPU)"
```

The second-last line should print a GPU. Then restart Jupyter and choose **Kernel > Change kernel > Python (TF GPU)**. The first line the notebook prints should now be `Running on: GPU`.

On a Mac with an Apple chip, TensorFlow uses the GPU through Apple's `tensorflow-metal` plugin.

## Troubleshooting

* **`TypeError: Unable to convert function return value to a Python type! The signature was () -> handle`** when TensorFlow is imported: your TensorFlow is older than your NumPy (TensorFlow 2.15 and older need NumPy below 2). Run `pip install "numpy<2"`, then restart the kernel.
* **`Running on: CPU only`**: TensorFlow does not see a GPU. See the Windows section above, or use Google Colab.

## Credits

* Dataset: the TensorFlow flowers dataset (3,670 photos in 5 folders). The download includes a `LICENSE.txt` with the photo credits.
* Models: ResNet50 (He et al., 2016), MobileNetV2 (Sandler et al., 2018) and EfficientNetB0 (Tan and Le, 2019), with ImageNet weights from Keras Applications.
