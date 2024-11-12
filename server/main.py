from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib

import librosa
import numpy as np
import pandas as pd
import json
import joblib
import math
import io
import base64
import threading
matplotlib.use('Agg')

app = Flask(__name__)
cors = CORS(app, origins='*')

# uploaded audio
uploaded_audio = None
segment_data = None
segment_data_df = None

# load the model
df_model = load_model('models/DeepFake_model_ver5_full.keras')

# load scaler
df_scaler = joblib.load('scalers/df_scaler.pkl')
si_scaler = joblib.load('scalers/vr_scaler.pkl')

others_profile = pd.read_csv('results/voice_recognition/training_full/vr_other_segment.csv')

# preprocessing the audio input
def pre_process(audio_file):
    target_sr = 22050

    signal, sr = librosa.load(audio_file, sr=target_sr, mono=False, res_type='kaiser_fast')
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)
    
    return signal, target_sr

# extracting mfcc, delta, delta 2
def extract_MFCC(signal, sr):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    comprehensive_mfcc = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    return comprehensive_mfcc

# extracting RMS, ZCR, SC, SB
def extract_features(audio, sr, target_sr = 22050):
    FRAME_LENGTH = 1024
    HOP_LENGTH = 512

    rms_cropped = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    zcr_cropped = librosa.feature.zero_crossing_rate(y=audio, frame_length=FRAME_LENGTH)[0]
    sc_cropped = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    sb_cropped = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    return rms_cropped, zcr_cropped, sc_cropped, sb_cropped


# processing audio
def process_audio_segments(files, segment_duration = 2):
    mfcc_features = []
    rms_values = []
    zcr_values = []
    sc_values = []
    sb_values = []
    file_names = []

    for audio_file in files:
        segment_number = 0
        signal, sr = pre_process(audio_file)
        segment_length = int(segment_duration*sr)
        total_length = len(signal)

        for start in range(0, total_length, segment_length):
            segment_number += 1
            end = start + segment_length
            if end > total_length:
                break

            segment = signal[start:end]

            mfcc = extract_MFCC(segment, sr)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            mfcc_features.append(mfcc_mean)

            rms, zcr, sc, sb = extract_features(segment, sr)
            rms_values.append(rms)
            zcr_values.append(zcr)
            sc_values.append(sc)
            sb_values.append(sb)

            file_names.append(f"Segment {segment_number} {audio_file.filename}")

    return file_names, mfcc_features, rms_values, zcr_values, sc_values, sb_values

# segment data for deepfake detection
def df_segment_data(file_names, mfcc_features, rms_values, zcr_values, sc_values, sb_values):
    mfcc_df = pd.DataFrame(mfcc_features, columns=[f'MFCC{i+1}' for i in range(mfcc_features[0].shape[0])])
    rms_df = pd.DataFrame(rms_values, columns=[f'RMS{i+1}' for i in range(len(rms_values[0]))])
    zcr_df = pd.DataFrame(zcr_values, columns=[f'ZCR{i+1}' for i in range(len(zcr_values[0]))])
    sc_df = pd.DataFrame(sc_values, columns=[f'SpectralCentroid{i+1}' for i in range(len(sc_values[0]))])
    sb_df = pd.DataFrame(sb_values, columns=[f'SpectralBandwidth{i+1}' for i in range(len(sb_values[0]))])

    combined_df = pd.concat([pd.DataFrame(file_names, columns=['File Name']), mfcc_df, rms_df, zcr_df, sc_df, sb_df], axis=1)
    return combined_df

def reshape_segment(segment_features, scaler):
    reshaped = scaler.transform([segment_features])
    reshaped = reshaped.reshape(1, reshaped.shape[1], 1, 1)

    return reshaped

def evaluate_segments(segment_data, model, scaler):
    confidence_scores = []
    for i in range(0, segment_data.shape[0]):
        segment_row = segment_data[i]
        reshaped_segment = reshape_segment(segment_row, scaler)

        predictions = model.predict(reshaped_segment)
        confidence = predictions[0][0]
        confidence_scores.append(float(confidence))
    
    return confidence_scores

def average(confidence_scores):
    avg = sum(confidence_scores)/len(confidence_scores)
    rounded = math.ceil(avg * 100)
    return rounded

def plot(uploaded_sequence, speaker_sequence, ylabel, title):
    
    plt.figure(figsize=(12, 6))
    plt.plot(uploaded_sequence, linestyle='-', color='red', label='Uploaded')
    plt.plot(speaker_sequence, linestyle='-', color='green', label='Speaker')

    plt.xlabel('Feature Column')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # convert to base64
    bytesIO = io.BytesIO()
    plt.savefig(bytesIO, format='jpg')
    bytesIO.seek(0)
    base64_data = base64.b64encode(bytesIO.read()).decode()
    return(base64_data)

def isolate_df(upload_df, speaker_df, header):
    u_data_df = upload_df[[col for col in upload_df.columns if col.startswith(header)]]
    s_data_df = speaker_df[[col for col in speaker_df.columns if col.startswith(header)]]
    u_data_df.columns = range(1, len(u_data_df.columns) + 1)
    s_data_df.columns = range(1, len(s_data_df.columns) + 1)
    u_mean =  u_data_df.mean(axis=0)
    s_mean = s_data_df.mean(axis=0)

    return u_mean, s_mean

@app.route("/api/upload", methods=['POST'])
def audio_uploaded():
    if 'audio_file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    if file:
        global segment_data
        global segment_data_df
        uploaded_audio = file
        processed_upload = process_audio_segments([uploaded_audio])
        segment_data_df = df_segment_data(*processed_upload)

        segment_data = segment_data_df.drop(columns=['File Name']).values

        evaluate = evaluate_segments(segment_data, df_model, df_scaler)
        overall = average(evaluate)

        segment_data_json_df = segment_data_df.to_dict(orient="records")
        response_data = {
            "overall": overall,
            "uploaded_data": segment_data_json_df
        }

        response_json = json.dumps(response_data)
        return Response(response_json, mimetype='application/json')


@app.route("/api/record", methods=['POST'])
def audio_record():

    files = request.files.getlist('audio_files')
    if len(files) != 10:
        return jsonify({"error": f"Exactly 10 files are required. {len(files)}"}), 400

    for idx, file in enumerate(files):
        if file.filename == '':
            return jsonify({"error": f"File {idx+1} is missing a name"}), 400
        
    si_segment_data = process_audio_segments(files)
    speaker_profile = df_segment_data(*si_segment_data)
    speaker_data = speaker_profile

    speaker_profile['label'] = 1
    others_profile['label'] = 0
    others_profile.columns = speaker_profile.columns

    data_df = pd.concat([speaker_profile, others_profile], ignore_index=True)
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    X = data_df.drop(columns=['File Name','label']).values
    y = data_df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))
    input_shape = (X_train.shape[1], 1, 1)
    print(f'input shape {input_shape}')

    speaker_identification_model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2,1)),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2,1)),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    speaker_identification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    speaker_identification_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.01)
    speaker_identification_model.evaluate(X_test, y_test)
    si_model = speaker_identification_model
    
    
    if segment_data is None:
        return jsonify({'message': "Data is empty"}), 400
    
    evaluate = evaluate_segments(segment_data, si_model, si_scaler)
    overall = average(evaluate)
    
    speaker_data_json_df = speaker_data.to_dict(orient="records")

    mfcc = isolate_df(segment_data_df, speaker_profile, 'MFCC')
    mfcc_plot = plot(*mfcc, 'mean MFCC value', 'Mean MFCC')
    
    rms = isolate_df(segment_data_df, speaker_profile, 'RMS')
    rms_plot = plot(*rms, 'mean RMS value', 'Mean RMS')
    
    zcr = isolate_df(segment_data_df, speaker_profile, 'ZCR')
    zcr_plot = plot(*zcr, 'mean ZCR value', 'Mean ZCR')

    sc = isolate_df(segment_data_df, speaker_profile, 'SpectralCentroid')
    sc_plot = plot(*sc, 'mean Spectral Centroid value', 'Mean Spectral Centroid')
    
    sb = isolate_df(segment_data_df, speaker_profile, 'SpectralBandwidth')
    sb_plot = plot(*sb,'mean Spectral Bandwidth value', 'Mean Spectral Bandwidth')

    response_data = {
        "overall": overall,
        "mfcc_plot": mfcc_plot,
        "rms_plot": rms_plot,
        "zcr_plot": zcr_plot,
        "sc_plot": sc_plot,
        "sb_plot": sb_plot,
        "uploaded_data": speaker_data_json_df
    }

    response_json = json.dumps(response_data)
    return Response(response_json, mimetype='application/json')

if __name__ == "__main__":
    app.run(debug=True, port=8080)