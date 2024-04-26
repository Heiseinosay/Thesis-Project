import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate for React Router v6
import anime from 'animejs/lib/anime.es.js';
import '../style/loading.css';
import '../style/root.css';

function Loading() {
    const navigate = useNavigate(); // Use useNavigate for React Router v6

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

        const myTimeout = setTimeout(() => {
            navigate('/Result'); // Redirect to '/Result' after 3000 milliseconds
        }, 8000);

        return () => {
            clearTimeout(myTimeout); // Clear the timeout when the component unmounts
        };
    }, [navigate]); // Add navigate to dependency array to avoid warning

    return (
        <div>
            <h1>Detecting...</h1>
            <div id="loading-bar">
                <div className="loading-segment"></div>
                <div className="loading-segment"></div>
                <div className="loading-segment"></div>
                <div className="loading-segment"></div>
                <div className="loading-segment"></div>
                <div className="loading-segment"></div>
                <div className="loading-segment"></div>
            </div>
        </div>
    );
}

export default Loading;
