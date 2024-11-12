import React, { useState,useContext, useEffect, useCallback } from 'react'
import Navigation from '../components/NavigationProgress'
import '../style/result.css'
import TableScale from '../components/TableScale'
import { DataContext } from '../components/DataContext'
import Circular from '../components/Circular'
import {Buffer} from 'buffer'
import ImageViewer from 'react-simple-image-viewer'
import {Helmet} from 'react-helmet';

function Result() {
    const [imageCurrent, setImage] = useState('')
    const { uploadData, recordData } = useContext(DataContext);
    
    const [isViewerOpen, setIsViewerOpen] = useState(false);
    const openImageViewer = useCallback((image) => {
        console.log("current image", image)
        setImage(image)
        setIsViewerOpen(true);
    }, []);

    const closeImageViewer = () => {
        setImage('');
        setIsViewerOpen(false);
    };
    
    const res = (jsonData) => {
        const oa = jsonData.overall;
        if (oa >= 50) {
            return {
                result: 'Human Voice',
                colorResult: '#1DB954',
                resultValue: 'Human',
                overall: oa
            };
        } else {
            return {
                result: 'AI Generated',
                colorResult: '#EF6056',
                resultValue: 'AI',
                overall: 100 - oa
            };
        }
    };
    
    const rec = (jsonData) => {
        const oa = jsonData.overall;
        if (oa >= 50) {
            return {
                resultRec: 'Similar to Speaker',
                colorResultRec: '#1DB954',
                resultValueRec: 'Real',
                overallRec: oa
            };
        } else {
            return {
                resultRec: 'Not the Speaker',
                colorResultRec: '#EF6056',
                resultValueRec: 'Fake',
                overallRec: oa
            };
        }
    };
    
    const {result, colorResult, resultValue, overall} = res(uploadData)
    const {resultRec, colorResultRec, resultValueRec, overallRec} = rec(recordData)
    
    return (
        <div className='body-result'>
            <Helmet>
                <title>Result</title>
            </Helmet>
            {isViewerOpen ? (
                    <ImageViewer
                    src={[imageCurrent]}
                    disableScroll={ false }
                    closeOnClickOutside={ true }
                    onClose={ closeImageViewer }
                    />
                ) : (
                    <>
                    <div className='result-title'>
                        {/* <h4>The Audio is</h4> */}
                        <div className="title-1 titles inter-regular">
                            <Circular colorValue={colorResult} resultValue={resultValue} progressValue={overall}/>
                            <div className="block">
                                <h1>{result}</h1>
                                <p>{overall}% Probability AI generated</p>
                            </div>
                        </div>
                        <div className="title-2 titles inter-regular">
                            <Circular colorValue={colorResultRec} resultValue={resultValueRec} progressValue={overallRec}/>
                            <div className='block'>
                                <h1>{resultRec}</h1>
                                <p>{overallRec}% Probability User's voice</p>
                            </div>
                        </div>
                    </div>
                    <div className="details-holder">
                        <details open>
                            <summary>View Statistics</summary>
                            <div className='details-img-holder'>
                                <div>
                                    <h4>MFCC</h4>
                                    <img 
                                        src={"data:image/jpeg;base64," + recordData.mfcc_plot} 
                                        alt="MFCC Plot" 
                                        onClick={()=> openImageViewer("data:image/jpeg;base64," + recordData.mfcc_plot)} 
                                    />
                                </div>
                                <div>
                                    <h4>Root-Mean-Square</h4>
                                    <img 
                                        src={"data:image/jpeg;base64," + recordData.rms_plot} 
                                        alt="RMS Plot"
                                        onClick={()=> openImageViewer("data:image/jpeg;base64," + recordData.rms_plot)}
                                    />
                                </div>
                                <div>
                                    <h4>Zero Crossing Rate</h4>
                                    <img 
                                        src={"data:image/jpeg;base64," + recordData.zcr_plot} 
                                        alt="ZCR Plot" 
                                        onClick={()=> openImageViewer("data:image/jpeg;base64," + recordData.zcr_plot)}
                                    />
                                </div>
                                <div>
                                    <h4>Spectral Centroid</h4>
                                    <img 
                                        src={"data:image/jpeg;base64," + recordData.sc_plot} 
                                        alt="Spectral Centroid Plot" 
                                        onClick={()=> openImageViewer("data:image/jpeg;base64," + recordData.sc_plot)}
                                    />
                                </div>
                                <div>
                                    <h4>Spectral Bandwidth</h4>
                                    <img 
                                        src={"data:image/jpeg;base64," + recordData.sb_plot} 
                                        alt="Spectral Bandwidth Plot"
                                        onClick={()=> openImageViewer("data:image/jpeg;base64," + recordData.sb_plot)}
                                    />
                                </div>
                            </div>
                            <div className='chart-label'>
                                <div className="label">
                                    <div className="label-box-1"></div>
                                    <p>Your Voice</p>
                                </div>

                                <div className="label">
                                    <div className="label-box-2"></div>
                                    <p>Uploaded Audio</p>
                                </div>
                            </div>
                        </details>
                    </div>
                        
                    <div className="table-holder">
                        <div className="table-container">
                            <h3>Speaker Profile</h3>
                            <TableScale data={{recordData}} />
                        </div>
                        <div className='table-container' style={{marginTop:'40px'}}>
                            <h3>Uploaded File</h3>
                            <TableScale data={{uploadData}} />
                        </div>
                    </div>
                    
                    
                    {/* <p className='result-tag inter-light'>Result:</p>

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

                    <h1 className='chart-title inter-bold'>Speaker Identification Comparison</h1>
                    <a href='#' className='inter-light'>See detailed comparison</a> */}

                    <Navigation active="3" className='nav' />
            </>
            )}
        </div>
    )
}

export default Result
