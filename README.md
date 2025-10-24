🟢⚪
# The Development of Detection Model of Human and AI Generated Voice 

#### Abstract

&nbsp;&nbsp;&nbsp;&nbsp; AI-generated voices, also known as text-to-speech (TTS), produce speech in diverse styles and accents using artificial intelligence. While this technology has been widely adopted across industries, it poses significant security risks, including voice cloning scams that have resulted in financial losses. This study addresses these concerns by developing a deepfake identification model to detect AI-generated voices and a speaker identification model to determine the speaker. The models analyzed features such as phonation, frequency, rate, and volume using six datasets, comprising 10,173 synthetic voice files and 10,142 human voice files. For speaker identification, a test participant recorded 10 English sentences containing all 44 English phonemes. The convolutional neural network (CNN) models, implemented in Python, achieved an accuracy of 96% across all performance metrics. This research highlights the potential of these models to improve voice authentication systems and mitigate the risks associated with AI voice cloning. However, further testing on speaker identification data is recommended to enhance performance. The study underscores the importance of integrating advanced detection technologies to safeguard privacy and security.

#### Scope
This research study is centered on designing and developing a software detection model to analyze and compare human and AI-generated voices. The outcome includes two models: a "Deepfake Identification Model" and a "Speaker Identification Model." The Deepfake Identification Model focuses on distinguishing whether an audio sample is of human origin or AI-generated. The Speaker Identification Model, on the other hand, determines how accurately modern AI replicates human speech.
# Methodology

#### Voice Features

| Simple term             | Hex         | Description (summary) 
| ----------------- | ------------------------------------------------------------------ |------------------------|
| Phonation | Mel-Frequency Cepstral Coefficients (MFCC)| MFCCs are a representation of the short-term power spectrum of sound, commonly used  voice recognition.|
| Frequency | Spectral Values & Spectral Centroid | Spectral values represent the energy distribution across frequencies, while the spectral centroid indicates the 'center of mass' of the spectrum and is perceived as the brightness of the sound.|
| Rate | Zero Crossing Rate | This measures how often the signal changes from positive to negative or vice versa in a given time frame. It’s indicative of the signal's noisiness. |
| Volume | Root-Mean-Square | RMS provides a mathematical average of the signal's amplitude, representing the perceived loudness.|



#### Conceptual Framework

![img alt](https://github.com/Heiseinosay/Heiseinosay-img-draft/blob/main/conceptual%20framework.png?raw=true)

#### File format
**The model only accepts 3 audio file types:** ```.mp3```, ```.wav```, and ```.flac```


#### Data Pre-processing

Human and AI-generated voices were located in a separate folder. Each audio in each folder was annotated with binary format according to their class, ```“0”``` represented the human voices, and ```“1”``` represented the AI-generated voices. This is necessary for identification of each audio during data extraction and model prediction. Instead of processing the filename of every audio, the model just refered to the annotated binary of the audio.

There are numerous audio channels such as mono, stereo, 5.1,
49
and 7.1. The model using the mono audio channel because all the elements of the sound, such as vocals, instruments, and effects, are combined and played through a single channel at the same volume as compared to stereo which used two separate channels, and 5.1 and 7.1 having different directions such as but not limited to front left, side left, rear right, and center. If the file is accepted, the model was then convert to the mono audio channel with a sampling rate of ```22050 Hz``` allowing for consistency between all the audio files.

![img alt](https://github.com/Heiseinosay/Heiseinosay-img-draft/blob/main/tomono.png?raw=true)

#### Segmentation
To effectively extract information from these signals, splitting the entire audio into sufficiently short ```2 seconds``` segments was essential. In this research, the audio dataset, synthetic and human voice, was processed by partitioning each signal up to two seconds. The segmentation aimed to enable a more detailed analysis of the audio features present in short, time-bound segments of the overall audio.

![image alt](https://github.com/Heiseinosay/Heiseinosay-img-draft/blob/main/segment.png?raw=true)


#### Feature Extraction
Thirteen Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each audio sample. These coefficients result in a 13-column matrix and multiple rows based on the audio's duration. In this study, the duration of each processed audio sample is fixed at 2 seconds. The MFCCs are then reduced by calculating their mean values, yielding a set of 13 average coefficients. Additionally, Delta 1 and Delta 2 are extracted from the original coefficients, resulting in a total of 39 MFCC features per audio sample 
```(13 coefficients × 3)```.

The remaining features—Zero Crossing Rate (ZCR), Root-Mean-Square (RMS), Spectral Bandwidth (SB), and Spectral Centroid (SC)—are all calculated based on the 2-second audio duration, providing 87 values per audio sample.

![img alt](https://github.com/Heiseinosay/Heiseinosay-img-draft/blob/main/partition.png?raw=true)

#### Convolutional Neural Networks (CNN) Detection Model
The CNN model utilized Tensorflow which is a deep learning framework library and Long Short Term Memory model. The CNN, along with Tensorflow framework, analyzed the voices (labeled 1 and 0 for human and AI respectively) and looked for any specific features or patterns in the sound waves that might indicate if it is human or AI-generated. The preprocessed audio data was then converted into spectrograms for visual representation. After that, feature extraction was performed, where multiple layers across the spectrogram were analyzed for specific patterns. As the CNN processed more and more spectrograms, the extracted features could be associated with specific sounds or words. The Long Short-Term Memory (LSTM) model could remember the patterns or features and compare them to previous findings. Once the CNN analyzed the extracted features, it output a prediction based on the identified patterns or features.
## Tech Stack

**Client:** &nbsp; ```ReactJs```, ```Animejs```

**Server:** &nbsp; ```Python Flask```

**Model Building:** &nbsp; ```Tensorflow```, ```Scikit-Learn```, ```Librosa```, ```Pandas```, ```Numpy```, ```Matplotlib```




# Dataset

#### Synthetic
| Source             | Number of data                                                                |
| ----------------- | ------------------------------------------------------------------ |
| birgermoell/synthetic_compassion_wav | 1722 files = rows |
| saahith/synthetic_with_val | 405 files = rows |
| Fake or Real | 6978  |
| Speechify(self-built dataset) | 1068 files = 1769 rows |
| Total synthetic files | 10,173 | 
| Total synthetic | 10,874 |

#### Human
| Source             | Number of data                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Common-voice-filo | 326 files = 627 rows |
| Common-voice-otheraccent | 700 files = 1142 rows |
| Fake or Real | 6978 files & rows  |
| Speech accent archive | 2138(tagalog(18) and other accent(2120) seperated) files = 2120 rows & 250 rows (segmented Filipino accent) |
| Total human files | 10,142 | 
| Total human | 11,117 |

Total Rows (Human + Synthetic) = ```21,991```

Total hours = ```12.217 hours```


**References:**
 - [Fake or Real](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
 - [Speech accent archive](https://www.kaggle.com/datasets/rtatman/speech-accent-archive)
 - [Hugging face birgermoell](https://huggingface.co/datasets/saahith/synthetic_with_val)
 - [Hugging face saahith](https://huggingface.co/datasets/birgermoell/synthetic_compassion_wav)
 - [Common voice](https://www.kaggle.com/datasets/mozillaorg/common-voice)
# Output Summary

![img alt](https://github.com/Heiseinosay/Heiseinosay-img-draft/blob/main/waveform.png?raw=true)

&nbsp; &nbsp; &nbsp; A 22-second audio sample
ranges up to 0.6 in amplitude. Amplitude varies significantly, reflecting natural
loudness changes as the speaker emphasizes different words. Lower 94
amplitudes, closer to zero, indicate softer parts of the audio, such as pauses or
quieter syllables. In comparison, the synthetic audio waveform (Figure #
Synthetic Audio Waveform) shows a more uniform amplitude pattern, where each
word begins at a similar volume level, and pauses around punctuation are
consistently spaced. This regularity reflects how AI-generated voices maintain
steady intensity and pause duration.

![img alt](https://github.com/Heiseinosay/Heiseinosay-img-draft/blob/main/accuracy.png?raw=true)

&nbsp; &nbsp; &nbsp; The figure illustrates the accuracy progression of the deep fake detection
model throughout its training. The X-axis represents the epochs (limited to 30),
while the Y-axis represents the accuracy metric, ranging from 0 to 1. The training
accuracy shows a strong initial performance, starting at 0.75 and steadily
103 increasing over time, eventually reaching values close to 1. The validation
accuracy also demonstrates promising results, beginning at 0.85 and consistently
rising, despite minor fluctuations. It stabilizes between 0.93 and 0.96, culminating
at 0.95.

![img alt](https://github.com/Heiseinosay/Heiseinosay-img-draft/blob/main/confusionmatrix.png?raw=true)

The deepfake detection model was trained on an 80-20 data split, with
80% of the dataset used for training and 20% reserved for testing. Out of a total
of 21,991 rows (each representing a 2-second audio segment), 17,592 rows
were used for model training, while the remaining 4,399 rows constituted the test
set. When evaluated on this test set, the model correctly classified 4,228 out of
4,399 samples, achieving an accuracy rate of 96%.

#### Conclusion

In terms of evaluation by the human ear, AI-cloned voices do an impressive
job of imitating the speaker’s voice, often sounding nearly indistinguishable from
real voices. However, they still fall short in terms of the cleanliness of the audio,
which can serve as an indicator of synthetic origin. Moreover, the results highlight
that AI-cloned voices tend to be more controlled and bound to a specific range of
signals. This highlights that the uniformity and consistency of AI-generated
speech remain significant weaknesses in fully mimicking human speech.

#### Recommendation
the proponents recommend increasing the voice variability to
create a wider sample for the model to learn from. Expanding the number of data
sets that shall be used is also endorsed to increase the accuracy and avoid
overfitting in the results. In line with this, and to create a better version of the
model, the utilization of a more diverse speech accent is also being put forward
to ensure the diverse use of the system. In connection with the aforementioned
proposition, the proponents encourage including other medium aside from
English.


# Setup
1. Go to project's root directory/client
2. Install dependencies
```
npm install
```
3. Run the server
```
npm start
```
4. Go to server dir

5. create venv 
```
python -m venv venv
```

6. then install packages
```
pip install -r requirements.txt
```

7. run the python server
```
python main.py
```
