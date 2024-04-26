import React, {useState, useRef, useEffect} from "react";
import '../style/recordingButton.css'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMicrophone } from '@fortawesome/free-solid-svg-icons'
import { faStop } from '@fortawesome/free-solid-svg-icons'
import { faCheck } from '@fortawesome/free-solid-svg-icons'
import { faRotateRight } from "@fortawesome/free-solid-svg-icons";

const RecordingButton = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [showRecordingButton, setShowRecordingButton] = useState(true); 
    const [showConfirmation, setConfirmation] = useState(false)
    const [audioURL, setAudioURL] = useState(null);
    const audioRef = useRef(null);
    const mediaRecorderRef = useRef(null);


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
            const chunks = [];

            mediaRecorder.ondataavailable = e =>{
                chunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(chunks, {type: 'audio/mp3'})
                const audioUrl = URL.createObjectURL(audioBlob);
                setAudioURL(audioUrl);
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

    // let hideMic = document.querySelector('.record')
    // hideMic.style.display = 'none';
    // useEffect(() => {
    //     const hideMic = document.querySelector('.record');
    //     if (hideMic) {
    //         hideMic.style.display = isRecording ? 'none' : 'block';
    //     }
    // }, [isRecording]);

    return(
        <div className="mic">
            {audioURL && <audio controls className='playback' ref={audioRef} src={audioURL}/>}
            {/* <FontAwesomeIcon className="record" icon={icon} onClick={()=>{handleImageClick(); setAudioURL(null)}} /> */}
            {showRecordingButton && (
                <FontAwesomeIcon className="record" icon={isRecording ? faStop : faMicrophone} onClick={handleImageClick} />
            )}

            {showConfirmation && (
                    <div>
                        <FontAwesomeIcon className="check" icon={faCheck} onClick={()=>{setShowRecordingButton(true); setConfirmation(false); setAudioURL(null)}} />
                        <FontAwesomeIcon className="again" icon={faRotateRight} onClick={()=>{setShowRecordingButton(true); setConfirmation(false); setAudioURL(null)}}/>
                    </div> 
            )}
        </div>
    );
};

export default RecordingButton;