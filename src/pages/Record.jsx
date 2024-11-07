import '../style/root.css'
import '../style/font.css'
import '../style/record.css'
import React, { Component, useState, useEffect, useContext } from 'react'
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCircleInfo } from '@fortawesome/free-solid-svg-icons'
import { faAngleRight } from '@fortawesome/free-solid-svg-icons'
import { faAngleLeft } from '@fortawesome/free-solid-svg-icons'
import { faMicrophone } from '@fortawesome/free-solid-svg-icons'

import Navigation from '../components/NavigationProgress'
import RecordButton from '../components/RecordingButton'
import Loading from './Loading'
import { DataContext } from '../components/DataContext';

//sentences
const sentenceArray =
    ["That quick beige fox jumped in the air over each thin dog. Look out, I shout, for he's foiled you again, creating chaos.",
        "Are those shy Eurasian footwear, cowboy chaps, or jolly earthmoving headgear?",
        "The hungry purple dinosaur ate the kind, zingy fox, the jabbering crab, and the mad whale and started vending and quacking.",
        "With tenure, Suzie’d have all the more leisure for yachting, but her publications are no good.",
        "Shaw, those twelve beige hooks are joined if I patch a young, gooey mouth.",
        "The beige hue on the waters of the loch impressed all, including the French queen, before she heard that symphony again, just as young Arthur wanted.",
        "Arthur stood and watched them hurry away. \"I think I'll go tomorrow,\" he calmly said to himself, but then again \"I don't know; it's so nice and snug here.\"",
        "The fuzzy caterpillar slowly crawled up the tall oak tree, seeking shelter from the impending rain.",
        "Ivan fixed the broken lock on the rusty gate with a sturdy hammer and a handful of nails.",
        "The mischievous child giggled as he splashed in the muddy puddles, making a mess of his new shoes."]



function Record() {
    const { setRecordData } = useContext(DataContext);
    //changing sentences  
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [sentenceCount, setCount] = useState(0);
    const [recordings, setRecordings] = useState(Array(10).fill(null));
    
    const incrementCount = (polarity) => {
        if (polarity === 0 && sentenceCount > 0) {
            setCount(sentenceCount - 1)
            console.log(sentenceCount)
        } else if (polarity === 1 && sentenceCount < 9) {
            setCount(sentenceCount + 1)
            console.log(sentenceCount)
        }
    }

    const handleDetect = async () => {
        const formData = new FormData();
        recordings.forEach((recording, index)=> {
            formData.append('audio_files', recording, `Recording_${index}.mp3`)
        });

        try{
            setLoading(true);
            const response = await axios.post('http://127.0.0.1:8080/api/record', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            console.log('Audio submitted successfully:', response.data);
            setRecordData(response.data);
            navigate("/result")
        } catch(error) {
            console.error('Error submitting audio:', error);
        } finally {
            setLoading(false);
        }

    };

    const handleAudioSubmit = (audioBlob) => {
        setRecordings((prevRecordings) => {
            const updatedRecordings = [...prevRecordings];
            updatedRecordings[sentenceCount] = audioBlob;
            return updatedRecordings;
            
        });

        incrementCount(1)
        // console.log(recordings);

        // try{
        //     const response = await axios.post('http://127.0.0.1:8080/api/record', formData, {
        //         headers: {
        //             'Content-Type': 'multipart/formdata',
        //         },
        //     });
        //     console.log('Audio sumbitted successfully:', response.data);

        // } catch (error){
        //     console.error('Error submitting audio:', error);
        // }
    };

    useEffect(() => {
        console.log("Updated recordings:", recordings);
    }, [recordings]);

    // const handleTest = async () => {
    //     try {
    //         const response = await axios.post('http://127.0.0.1:8080/api/test');

    //         // Check if the response is ok (status in the range 200-299)
    //         console.log('Response data:', response.data); // Log the response data
    //     } catch (error) {
    //         console.error('Error submitting test request:', error); // Log any errors
    //     }
    // };

    return (
        <div className='body-record'>
            {loading ? (
                <Loading />
            ) : (
                <>
                <Navigation active={2} />
                <h1 className='title'>Record your voice</h1>
                <div className='reminder'>
                    <FontAwesomeIcon className='info' icon={faCircleInfo} />
                    <h6 className='inter-regular'>For best result speak clearly in microphone</h6>
                </div>

                <h1 className='inter-regular sentence'>{sentenceArray[sentenceCount]}</h1>

                <div className='progress'>
                    <FontAwesomeIcon className='next-prev' icon={faAngleLeft} onClick={() => incrementCount(0)} />
                    <h4 className='inter-bold'>{sentenceCount + 1}</h4>
                    <FontAwesomeIcon className='next-prev' icon={faAngleRight} onClick={() => incrementCount(1)} />
                </div>

                <div className="mic">
                    <RecordButton onAudioSubmit={handleAudioSubmit}/>
                    {/* <FontAwesomeIcon icon={faMicrophone} /> */}
                </div>

                {recordings.every(recording => recording != null) ? (
                    <button className='btn-detect' onClick={handleDetect}>Detect</button>
                ): null}
                <p className='inter-light note'><b className='inter-bold'>Note:</b> We need to analyze a short audio sample focusing on aspects like how you sound, the energy in your voice, and your speaking pace. Your privacy is important to us. Your voice recordings will be anonymized and used solely for voice recognition technology.</p>
                {/* <button className="test" onClick={handleTest}>TEST</button> */}
                </>
            )}
        </div>

    )
}

export default Record