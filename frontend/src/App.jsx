import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import Footer from "./components/Footer";
import Navbar from "./components/Navbar/Navbar";
import About from "./pages/About";
import Data from "./pages/Data";
import Explore from "./pages/Explore";
import WatchList from "./pages/WatchList";

// Old links (/predictor?district=671) keep working: the watch list moved, with its query
function MovedTo({ path }) {
  const { search } = useLocation();
  return <Navigate to={{ pathname: path, search }} replace />;
}

export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Explore />} />
        <Route path="/watch-list" element={<WatchList />} />
        <Route path="/data" element={<Data />} />
        <Route path="/about" element={<About />} />
        <Route path="/predictor" element={<MovedTo path="/watch-list" />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Footer />
    </>
  );
}
