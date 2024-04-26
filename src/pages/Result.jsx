import React from 'react'
import Navigation from '../components/NavigationProgress'
import '../style/result.css'

function Result() {
    return (
        <div className='body-result'>
            <div className='result-title'>
                {/* <h4>The Audio is</h4> */}
                <div className="title-1 titles inter-regular">
                    <h1>AI Generate</h1>
                    <p>confidence score: 95%</p>
                </div>
                <div className="title-2 titles inter-regular">
                    <h1>Similar with speaker</h1>
                    <p>confidence score: 95%</p>
                </div>
            </div>

            <p className='result-tag inter-light'>Result:</p>

            <div className="result-container">
                <div className="result-label inter-regular">
                    <h2>MFCC</h2>
                    <h2>Frequency</h2>
                    <h2>Rate</h2>
                    <h2>Volume</h2>
                </div>

                <div className="chart inter-reular">
                    <div className="group1 bar-group">
                        <div className="human1 human-bar"></div>
                        <div className="ai1 ai-bar"></div>
                    </div>
                    <div className="group2 bar-group">
                        <div className="human2 human-bar"></div>
                        <div className="ai2 ai-bar"></div>
                    </div>
                    <div className="group3 bar-group">
                        <div className="human3 human-bar"></div>
                        <div className="ai3 ai-bar"></div>
                    </div>
                    <div className="group4 bar-group">
                        <div className="human4 human-bar"></div>
                        <div className="ai4 ai-bar"></div>
                    </div>
                </div>
            </div>
            <div className='chart-label'>
                <div className="label">
                    <div className="label-box-1"></div>
                    <p>Your Voice</p>
                </div>

                <div className="label">
                    <div className="label-box-2"></div>
                    <p>Suspicous</p>
                </div>
            </div>

            <h1 className='chart-title inter-bold'>Speaker Identification Comparison</h1>
            <a href='#' className='inter-light'>See detailed comparison</a>

            <Navigation active="3" />
        </div>
    )
}

export default Result
