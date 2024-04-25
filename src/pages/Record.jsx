import 'bootstrap/dist/css/bootstrap.min.css'
import '../style/root.css'
import '../style/font.css'
import '../style/record.css'
import React, {useState} from 'react'

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
        <div className='container text-center'>
            <div className='row align-items-center'>
                <div className='col'>
                    three
                </div>
                <div className='col-6'>
                    <div className='row'>
                        <div className='row'>
                            <h1 className='inter-bold h3 '>Record your voice</h1>
                        </div>
                        <div className='row d-flex align-items-center justify-content-center '>
                            <div className='col-2 d-flex justify-content-end '>
                                <img src={infoImage} className='img-fluid'/>
                            </div>
                            <div className='col-10 d-flex align-items-center m-0'>
                                <p className='display-6 inter-regular m-0'>For best results, speak clearly into the microphone</p>
                            </div>
                        </div>
                    </div>

                    <div className='row mt-5 '>
                        <div className='row border'>
                            <p className='display-5 p-5 border'>{sentenceArray[sentenceCount]}</p>
                        </div>
                        <div className='row d-flex justify-content-center align-items-center'>
                            <div className='col-6 d-flex justify-content-center align-items-center border'>
                                <div className='arrow-image'>
                                    <img src={prevImage} alt='Previous' onClick={() => incrementCount(0)}/>
                                </div>
                                <p className='inter-bold'>{sentenceCount + 1}</p>
                                <div className='arrow-image'>
                                    <img src={nextImage} alt='Next' onClick={() => incrementCount(1)}/>
                                </div>
                            </div>
                        </div>
                    </div>
                        

                    <div className='row'>
                        
                    </div>
                </div>
                <div className='col'>
                    three
                </div>
            </div>
        </div>
        
    )
}

export default Record

// <div className='body-container'>
        //     <div className='div-side'>
        //     </div>

        //     <div className='div-center'>

        //         <div className='title-container'>
        //             <h1 className='inter-bold record'>Record your voice</h1>
        //             <div className='container'>
        //                 <img src={infoImage} className='info-image' alt='info'/>
        //                 <p className='inter-regular info-paragraph'>For best results, speak clearly into the microphone</p>
        //             </div>
        //         </div>

        //         <div className='sentence-container'>
        //             <p className='inter-record sentence'>{sentenceArray[sentenceCount]}</p>
        //         </div>

        //         <div className='button-container'>

        //             <div className='arrow-image'>
        //                 <img src={prevImage} alt='Previous' onClick={() => incrementCount(0)}/>
        //             </div>

        //             <div>
        //                 <p className='inter-bold'>{sentenceCount + 1}</p>
        //             </div>

        //             <div className='arrow-image'>
        //                 <img src={nextImage} alt='Next' onClick={() => incrementCount(1)}/>
        //             </div>

        //         </div>

        //     </div>
            
        //     <div className='div-side'>
        //     </div>
        // </div>