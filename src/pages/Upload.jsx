import React, {useState} from 'react'
import '../style/root.css'
import '../style/font.css'
import '../style/upload.css'
import {Helmet} from 'react-helmet';

import Navigation from '../components/NavigationProgress'
import DropZone from '../components/DragDropFiles'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCircleInfo } from '@fortawesome/free-solid-svg-icons'

import axios from "axios";

function Upload() {
    return (
        <div className='upload-body'>
            <Helmet>
                <title>Upload</title>
            </Helmet>
            <div className="column column-first">
                <div className="restrictions">
                    <h1 className='inter-bold'>Restrictions</h1>
                    <p className='inter-light'><FontAwesomeIcon className='icon' icon={faCircleInfo} />File size 50 mb</p>
                    <p className='inter-light'><FontAwesomeIcon className='icon' icon={faCircleInfo} />File format .wav, .mp3,. flac, etc.</p>
                </div>
            </div>
            <div className="column column-midle">
                <h1 className='inter-bold'>Verify Audio</h1>
                <DropZone/>
            </div>
            <div className="column"></div>
            <Navigation active={1} />
        </div>
    )
}

export default Upload
