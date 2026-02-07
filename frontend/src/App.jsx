import { BrowserRouter, Routes, Route } from "react-router-dom";
import Research from "./pages/Research";
import Predictor from "./pages/Predictor";
import Navbar from "./components/Navbar";

export default function App() {
  return (
    <>
    <Navbar />
      <Routes>
        <Route path="/" element={<Research />} />
        <Route path="/predictor" element={<Predictor />} />
      </Routes>
    </>
  );
}
