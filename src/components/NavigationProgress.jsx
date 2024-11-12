import '../style/navigation.css'

import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

function NavigationProgress(props) {
    // console.log(props.active)
    const activeLink = props.active
    const navigate = useNavigate();

    /* eslint-disable no-restricted-globals */
    const redirect = (event, link) => {
        event.preventDefault();
        console.log(activeLink)
        if (link === "/upload") {
            if (activeLink === 1) {
                return;
            }
            if (1 < activeLink) {
                if (confirm('All progress will be lost. Are you sure you want to go back?')) {
                    navigate(link);
                } else {
                    return;
                }
            }
        }

        if (link === "/record") {
            if (activeLink === 2) {
                return;
            }
            if (activeLink > 1) {
                if (confirm('All progress will be lost. Are you sure you want to go back?')) {
                    navigate(link);
                } else {
                    return;
                }
            } else {
                alert("Please accomplish the requirements")
            }
        }

        if (link === "/detect") {
            if (activeLink === 3) {
                return;
            }
            if (activeLink > 2) {
                if (confirm('All progress will be lost. Are you sure you want to go back?')) {
                    navigate(link);
                } else {
                    return;
                }
            } else {
                alert("Please accomplish the requirements")
            }
        }


    };
    /* eslint-enable no-restricted-globals */



    return (
        <div className='navigation'>
            <Link className='href' onClick={(event) => redirect(event, "/upload")}>
                <div className="links">
                    <h4 className={props.active >= 1 ? "inter-bold active-link" : "inter-regular"}>Upload</h4>
                    <div className="circle-active"></div>
                </div>
            </Link>
            <Link className='href' onClick={(event) => redirect(event, "/record")}>
                <div className="links">
                    <h4 className={props.active >= 2 ? "inter-bold active-link " : "inter-regular"}>Record</h4>
                    <div className={props.active >= 2 ? "circle-active" : "circle"}></div>
                </div>
            </Link>
            <Link className='href' onClick={(event) => redirect(event, "/detect")}>
                <div className="links">
                    <h4 className={props.active >= 3 ? "inter-bold active-link " : "inter-regular"}>Detect</h4>
                    <div className={props.active >= 3 ? "circle-active" : "circle"}></div>
                </div>
            </Link>
        </div >
    )
}

export default NavigationProgress
