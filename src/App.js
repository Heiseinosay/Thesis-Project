import logo from './logo.svg';
import './App.css';

import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Upload from './pages/Upload'
import Record from './pages/Record'
import Result from './pages/Result'
import Loading from './pages/Loading'

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path='/upload' element={<Upload />}></Route>
          <Route path='/record' element={<Record />}></Route>
          <Route path='/result' element={<Result />}></Route>
          <Route path='/loading' element={<Loading />}></Route>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
