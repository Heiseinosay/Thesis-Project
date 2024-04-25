import 'bootstrap/dist/css/bootstrap.min.css'
import '../style/root.css'
import '../style/font.css'
import '../style/record.css'
import React, {Component, useState} from 'react'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCircleInfo } from '@fortawesome/free-solid-svg-icons'
import { faAngleRight } from '@fortawesome/free-solid-svg-icons'
import { faAngleLeft } from '@fortawesome/free-solid-svg-icons'
import { faMicrophone } from '@fortawesome/free-solid-svg-icons'

import Navigation from '../components/NavigationProgress'


//images
import infoImage from '../images/info.png'
import nextImage from '../images/next.png'
import prevImage from '../images/prev.png'


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
    //changing sentences  
    const [sentenceCount, setCount] = useState(0);
    const incrementCount = (polarity) => {
        if (polarity == 0 && sentenceCount > 0){
            setCount(sentenceCount - 1)
            console.log(sentenceCount)
        } else if (polarity == 1 && sentenceCount < 9){
            setCount(sentenceCount + 1)
            console.log(sentenceCount)
        }
    }

    return (
        <div className='body-record'>
            <Navigation active={2} />
            <h1 className='title'>Record your voice</h1>
            <div className='reminder'>
                <FontAwesomeIcon className='info' icon={faCircleInfo} />
                <h6>For best result speak clearly in microphone</h6>
            </div>

            <h1 className='sentence'>{sentenceArray[sentenceCount]}</h1>

            <div className='progress'>
                <FontAwesomeIcon className='next-prev' icon={faAngleLeft} onClick={() => incrementCount(0)} />
                <h4>{sentenceCount + 1}</h4>
                <FontAwesomeIcon className='next-prev' icon={faAngleRight} onClick={() => incrementCount(1)} />
            </div>

            <div className="mic">
                <FontAwesomeIcon icon={faMicrophone} />
            </div>

            <p className='note'>Note: we need to analyze a short audio sample focusing on aspects like how you sound, the energy in your voice, and your speaking pace. Your privacy is important to us. Your voice recordings will be anonymized and used solely for voice recognition technology.</p>
        </div>
        
    )
}

export default Record