import { Link } from "react-router-dom";
import ApiStatus from "./ApiStatus";
import { REPO } from "../lib/api";

export default function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 text-sm text-slate-500 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <p>
          Data: NCRB, <i>Crime in India</i>, 2001–2022, re-based onto today's districts.{" "}
          <Link to="/data" className="text-blue-700 hover:underline">Download it</Link>.
        </p>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <ApiStatus />
          <a href={REPO} target="_blank" rel="noopener noreferrer" className="hover:underline">Source code</a>
          <Link to="/about" className="hover:underline">Built by Sanchit Rastogi</Link>
        </div>
      </div>
    </footer>
  );
}
