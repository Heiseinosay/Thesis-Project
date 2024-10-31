import React, {useState, useRef, useEffect} from "react";
import '../style/recordingButton.css'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMicrophone } from '@fortawesome/free-solid-svg-icons'
import { faStop } from '@fortawesome/free-solid-svg-icons'
import { faCheck } from '@fortawesome/free-solid-svg-icons'
import { faRotateRight } from "@fortawesome/free-solid-svg-icons";
import Record from '../pages/Record';

const RecordingButton = ({onAudioSubmit}) => {
    const [isRecording, setIsRecording] = useState(false);
    const [showRecordingButton, setShowRecordingButton] = useState(true); 
    const [showConfirmation, setConfirmation] = useState(false)
    const [audioBlob, setAudioBlob] = useState(null);
    const [audioLink, setAudioLink] = useState(null);
    // const [audioURLs, setAudioURLs] = useState(Array(10).fill(null));

    const audioRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    const handleImageClick = () =>{
        if (!isRecording){
            startRecording();
        } else{
            stopRecording();
            setShowRecordingButton(false);
            setConfirmation(true);
        }
    };

    const startRecording = () => {
        navigator.mediaDevices.getUserMedia({ audio:true})
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            chunksRef.current = [];

            mediaRecorder.ondataavailable = e =>{
                chunksRef.current.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, {type: 'audio/mpeg'});
                setAudioBlob(blob)
                setAudioLink(URL.createObjectURL(blob));
                // const updatedAudioURLs = [...audioURLs];
                // updatedAudioURLs[currentIndex] = audioUrl;
            };

            mediaRecorder.start();
            setIsRecording(true);
        })
        .catch(error =>{
            console.error('Error accessing microphone', error);
        });
    };

    const stopRecording = () => {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
    };

    const handleConfirm = () => {
        if (audioBlob) {
            onAudioSubmit(audioBlob);
            setAudioLink(null)
        }
        setShowRecordingButton(true);
        setConfirmation(false);
        

        // setCurrentIndex(prevIndex => prevIndex + 1 );
        // console.log(audioURLs)
    };

    const handleRetry = () => {
        setShowRecordingButton(true);
        setConfirmation(false);
        setAudioBlob(null);
        setAudioLink(null);
        // const updatedAudioURLs = [...audioURLs];
        // updatedAudioURLs[currentIndex] = null;
        // setAudioURLs(updatedAudioURLs);
    };

    // let hideMic = document.querySelector('.record')
    // hideMic.style.display = 'none';
    // useEffect(() => {
    //     const hideMic = document.querySelector('.record');
    //     if (hideMic) {
    //         hideMic.style.display = isRecording ? 'none' : 'block';
    //     }
    // }, [isRecording]);

    // const handleSubmit = async () => {
    //     const formData = new FormData();
    //     audioURLs.forEach((audioBlob, index)=> {
    //         if (audioBlob) {
    //             formData.append('audio_${index}', audioBlob, 'audio_${index}.mp3');
    //         }
    //     });

    //     try {
    //         onAudioSubmit(formData);
    //     } catch(error){
    //         console.error('Error uploading audio:', error)
    //     }
    // };

    return(
        <div className="mic">
            {audioLink && <audio controls className='playback' ref={audioRef} src={audioLink}/>}
            {/* <FontAwesomeIcon className="record" icon={icon} onClick={()=>{handleImageClick(); setAudioURL(null)}} /> */}
            {showRecordingButton && (
                <FontAwesomeIcon className="record" icon={isRecording ? faStop : faMicrophone} onClick={handleImageClick} />
            )}

            {showConfirmation && (
                    <div>
                        <FontAwesomeIcon className="check" icon={faCheck} onClick={handleConfirm} />
                        <FontAwesomeIcon className="again" icon={faRotateRight} onClick={handleRetry}/>
                    </div> 
            )}

            {/* {currentIndex >=10 && (
                <button onClick={handleSubmit} className="submit-button">Submit All Recordings</button>
            )} */}
        </div>
    );
};

export default RecordingButton;