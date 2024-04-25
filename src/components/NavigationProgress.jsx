import '../style/navigation.css'

import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

function NavigationProgress(props) {

    console.log(props.active)

    const redirect = (event, link) => {
        event.preventDefault();
        if (link === "0") {
            alert(true);
        }
    }

    return (
        <div className='navigation'>
            <Link className='href' onClick={(event) => redirect(event, "0")}>
                <div className="links">
                    <h4 className='inter-bold active-link'>Upload</h4>
                    <div className="circle-active"></div>
                </div>
            </Link>
            <Link className='href'>
                <div className="links">
                    <h4 className={props.active >= 2 ? "inter-regular active-link " : "inter-regular"}>Record</h4>
                    <div className={props.active >= 2 ? "circle-active" : "circle"}></div>
                </div>
            </Link>
            <Link className='href'>
                <div className="links">
                    <h4 className={props.active >= 3 ? "inter-regular active-link " : "inter-regular"}>Detect</h4>
                    <div className={props.active >= 3 ? "circle-active" : "circle"}></div>
                </div>
            </Link>
        </div >
    )
}

export default NavigationProgress
