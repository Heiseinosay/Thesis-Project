import React, {useState, useRef, useEffect} from "react";
import '../style/recordingButton.css'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMicrophone } from '@fortawesome/free-solid-svg-icons'
import { faStop } from '@fortawesome/free-solid-svg-icons'
import { faCheck } from '@fortawesome/free-solid-svg-icons'
import { faRotateRight } from "@fortawesome/free-solid-svg-icons";
import MicRecorder from 'mic-recorder-to-mp3';
import Record from '../pages/Record';
import { ariaHidden } from "@mui/material/Modal/ModalManager";
import { set } from "animejs";

const RecordingButton = ({onAudioSubmit, onStart, onEnd}) => {
    const [isRecording, setIsRecording] = useState(false);
    const [showRecordingButton, setShowRecordingButton] = useState(true); 
    const [showConfirmation, setConfirmation] = useState(false)
    const [audioBlob, setAudioBlob] = useState(null);
    const [audioLink, setAudioLink] = useState(null);

    const audioRef = useRef(null);
    const Mp3Recorder = useRef(new MicRecorder({ bitRate: 128 }));

    const handleImageClick = () =>{
        if (!isRecording){
            startRecording();
            onStart()
        } else{
            stopRecording();
        }
    };
    
    const startRecording = () => {
        navigator.getUserMedia(
          { audio: true },
          () => {
            console.log('Permission Granted');
            Mp3Recorder.current
              .start()
              .then(() => {
                setIsRecording(true);
              })
              .catch((e) => console.error(e));
          },
          () => {
            console.log('Permission Denied');
            alert('Microphone permission is denied');
          }
        );
      };

    const stopRecording = () => {
        Mp3Recorder.current
            .stop()
            .getMp3()
            .then(([buffer, blob]) => {
                console.log("Blob size:", blob.size);
                setIsRecording(false);

                const blobURL = URL.createObjectURL(blob);
                console.log("Generated blob URL:", blobURL);

                const audio = new Audio(blobURL);
                 audio.onloadedmetadata = () => {
                    const duration = audio.duration;
                    if (duration > 1) {
                        setAudioBlob(blob);
                        setAudioLink(blobURL);
                        setShowRecordingButton(false);
                        setConfirmation(true);
                    } else {
                        console.log('Invalid Audio Duration', duration)
                        alert("Audio must at least have a 1 second duration. Please try recording again.");
                        onEnd()
                    }
                };
                audio.onerror = () => {
                    console.error("Error loading audio metadata.");
                    alert("Audio not captured. Please try recording again.");
                    onEnd()
                };
        })
        .catch((e) => console.log("Error stopping recording", e));
    };


    const handleConfirm = () => {
        if (audioBlob) {
            onAudioSubmit(audioBlob);
            setAudioLink(null)
        }
        setShowRecordingButton(true);
        setConfirmation(false);
        onEnd()
    };

    const handleRetry = () => {
        setAudioBlob(null);
        setAudioLink(null);
        setShowRecordingButton(true);
        setConfirmation(false);
        onEnd()
    };

    return(
        <div className="mic">
            {audioLink && <audio controls className='playback' ref={audioRef} type={'audio/mpeg'} src={audioLink}/>}
            {showRecordingButton && (
                <FontAwesomeIcon className="record" icon={isRecording ? faStop : faMicrophone} onClick={handleImageClick}/>
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