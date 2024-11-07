import React, { useContext } from 'react'
import { useState, useRef, } from 'react'
import { Link, useNavigate } from 'react-router-dom';

import '../style/dropzone.css'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCloudArrowUp } from '@fortawesome/free-solid-svg-icons'
import { DataContext } from './DataContext';

import axios from "axios";

function DragDropFiles() {
    const {setUploadData} = useContext(DataContext)

    const [files, setFiles] = useState(null);
    const inputRef = useRef();
    const navigate = useNavigate();

    const handleDragOver = (event) => {
        event.preventDefault();
    }

    const handleDrop = (event) => {
        event.preventDefault();
        const droppedFiles = event.dataTransfer.files;
        if (droppedFiles.length > 1) {
            console.log(true)
            alert("Please upload 1 audio file only");
            return
        }
        // Check file type
        const file = droppedFiles[0];
        const allowedTypes = ["audio/mpeg", "audio/wav", "audio/flac"];
        const allowedExtensions = [".mp3", ".wav", ".flac"];
        const fileType = file.type;
        const fileName = file.name;
        const fileExtension = fileName.substring(fileName.lastIndexOf("."));
        // console.log(fileType)
        if (!allowedTypes.includes(fileType) || !allowedExtensions.includes(fileExtension)) {
            alert("please upload audio file only.")
            return
        }
        // Check file size
        const maxFileSizeMB = 50;
        const maxFileSizeBytes = maxFileSizeMB * 1024 * 1024;
        if (file.size > maxFileSizeBytes) {
            alert("Please upload a file smaller than 50MB.");
            return;
        }
        setFiles(event.dataTransfer.files)
    }

    const validateFile = (droppedFiles) => {
        if (droppedFiles.length > 1) {
            alert("Please upload 1 audio file only");
            return false;
        }

        const file = droppedFiles[0];
        const allowedTypes = ["audio/mpeg", "audio/wav", "audio/flac"];
        const allowedExtensions = [".mp3", ".wav", ".flac"];
        const fileType = file.type;
        const fileName = file.name;
        const fileExtension = fileName.substring(fileName.lastIndexOf("."));

        if (!allowedTypes.includes(fileType) || !allowedExtensions.includes(fileExtension)) {
            alert("Please upload an audio file with .mp3, .wav, or .flac extension.");
            return false;
        }

        const maxFileSizeMB = 50;
        const maxFileSizeBytes = maxFileSizeMB * 1024 * 1024;
        if (file.size > maxFileSizeBytes) {
            alert("Please upload a file smaller than 50MB.");
            return false;
        }

        return true;
    };


    const handleFileSelect = (event) => {
        const selectedFile = event.target.files[0];
        if (!validateFile([selectedFile])) {
            return;
        }
        setFiles([selectedFile]);
    };

    const handleUpload = async() => {
        // navigate("/record")
        // navigate("/result")
        if (!files || files.length === 0) return;

        const formData = new FormData();
        formData.append("audio_file", files[0]);

        try{
            navigate("/record")
            const response = await axios.post('http://127.0.0.1:8080/api/upload', formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });
            
            console.log('Upload response:', response.data);
            setUploadData(response.data);
            
        } catch (err){
            console.error('Upload error:', err)
        }
    };

    if (files)
        return (
            <div className='uploads'>
                <ul>
                    {Array.from(files).map((file, index) => (
                        <li key={index}>{file.name}</li>
                    ))}
                </ul>
                <div className="actions">
                    <button className='btn btn-upload' onClick={handleUpload}>Upload</button>
                    <button className='btn btn-cancel' onClick={() => setFiles(null)}>Cancel</button>
                </div>
            </div>
        )

    return (
        <>
            {!files && (
                <div className='dropzone' onDragOver={handleDragOver} onDrop={handleDrop}>
                    <FontAwesomeIcon className='icon-cloud' icon={faCloudArrowUp} />
                    <input type="file" onChange={handleFileSelect} ref={inputRef} hidden />
                    <p><button onClick={() => inputRef.current.click()}>Choose a file</button> or drag it here</p>
                </div>
            )}
        </>
    )
}

export default DragDropFiles
