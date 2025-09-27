// DataContext.js
import React, { createContext, useState } from 'react';

export const DataContext = createContext();

export const DataProvider = ({ children }) => {
    const [uploadData, setUploadData] = useState(null);
    const [recordData, setRecordData] = useState(null);

    return (
        <DataContext.Provider value={{ uploadData, setUploadData, recordData, setRecordData }}>
            {children}
        </DataContext.Provider>
    );
};
