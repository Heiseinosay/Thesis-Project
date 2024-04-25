import React from 'react'
import '../style/root.css'
import '../style/font.css'
import '../style/upload.css'

import Navigation from '../components/NavigationProgress'

function Upload() {
    return (
        <div>
            <h1 className='inter-bold hello'>Upload</h1>
            <Navigation active="upload" />
        </div>
    )
}

export default Upload
