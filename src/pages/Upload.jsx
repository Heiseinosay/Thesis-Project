import React, {useContext, useState, useEffect} from 'react'
import '../style/root.css'
import '../style/font.css'
import '../style/upload.css'

import Navigation from '../components/NavigationProgress'
import DropZone from '../components/DragDropFiles'
import Loading from './Loading'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCircleInfo } from '@fortawesome/free-solid-svg-icons'

import axios from "axios";
import { DataContext } from '../components/DataContext'
import { useNavigate } from 'react-router-dom'

function Upload() {
    const {setUploadData} = useContext(DataContext);
    const [files, setFiles] = useState(null);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    

    const handleFiles = (file) => {
        setFiles(file);
    }
    
    // call handleSubmit upload once files is updated
    useEffect(() => {
        if (files) {
            handleSubmitUpload();
        }
    }, [files]);

    const handleSubmitUpload = async() =>{       
        if (!files || files.length === 0) {
            alert('No files selected. Please upload a file before submitting.');
            return;
        }

        const formData = new FormData();
        formData.append("audio_file", files[0]);
        try{
            setLoading(true)
            const response = await axios.post('http://127.0.0.1:8080/api/upload', formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });
            console.log('Upload response:', response.data);
            setUploadData(response.data);
            navigate('/result-upload');
        } catch (err){
            console.error('Upload error:', err)
            alert(err)
        } finally {
            setLoading(false)
        }
    };
    return (
        <div className='upload-body'>
            {loading ? (
                <Loading />
            ) : (
            <>
            <div className="column column-first">
                <div className="restrictions">
                    <h1 className='inter-bold'>Restrictions</h1>
                    <p className='inter-light'><FontAwesomeIcon className='icon' icon={faCircleInfo} />File size 50 mb</p>
                    <p className='inter-light'><FontAwesomeIcon className='icon' icon={faCircleInfo} />File format .wav, .mp3,. flac, etc.</p>
                </div>
            </div>
            <div className="column column-midle">
                <h1 className='inter-bold'>Verify Audio</h1>
                <DropZone onUpload={handleFiles}/>
            </div>
            <div className="column"></div>
            <Navigation active={1} />
            </>
            )}
        </div>
    )
}

export default Upload
