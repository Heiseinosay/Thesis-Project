import logo from './logo.svg';
import './App.css';
import './style/root.css'
import './style/font.css'

import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Upload from './pages/Upload'
import Record from './pages/Record'
import Result from './pages/Result'
import Loading from './pages/Loading'
import ResultUpload from './pages/ResultUpload'
import { DataProvider } from './components/DataContext';

function App() {
  return (
    <div className="App">
      <DataProvider>
        <BrowserRouter>
          <Routes>
            <Route path='/upload' element={<Upload />}></Route>
            <Route path ='/result-upload' element={<ResultUpload/>}></Route>
            <Route path='/record' element={<Record />}></Route>
            <Route path='/result' element={<Result />}></Route>
            <Route path='/loading' element={<Loading />}></Route>
          </Routes>
        </BrowserRouter>
      </DataProvider>
    </div>
  );
}

export default App;
