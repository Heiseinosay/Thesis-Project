import React, { useEffect } from 'react';
import anime from 'animejs/lib/anime.es.js';
import '../style/loading.css';
import '../style/root.css';

function Loading() {

    useEffect(() => {
        anime({
            targets: '.loading-segment',
            easing: 'linear',
            duration: 1200,
            loop: true,
            direction: 'alternate',
            translateY: [
                { value: -70, delay: anime.stagger(100) },
                { value: 0, delay: anime.stagger(100) }
            ]
        });
    }, []);

    return (
        <div className='body-loading'>
            <h1>Detecting...</h1>
            <div id="loading-bar">
                {Array.from({ length: 7 }).map((_, index) => (
                    <div key={index} className="loading-segment"></div>
                ))}
            </div>
        </div>
    );
}

export default Loading;
