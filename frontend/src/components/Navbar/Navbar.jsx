import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { REPO } from "../../lib/api";
import { NAV_LINKS } from "./constants";

function GitHubIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex justify-between items-center gap-4">
        <Link to="/" className="flex items-center gap-2 shrink-0" onClick={() => setOpen(false)}>
          <span className="bg-linear-to-r from-blue-600 to-indigo-600 text-white p-1.5 rounded-lg">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </span>
          <span className="text-lg font-display font-bold text-slate-800 tracking-tight">
            Women's Safety <span className="text-blue-600">ML</span>
          </span>
        </Link>

        <div className="hidden md:flex items-center gap-6">
          {NAV_LINKS.map((link) => (
            <Link key={link.path} to={link.path}
                  className={`text-sm font-medium ${pathname === link.path ? "text-blue-700" : "text-slate-600 hover:text-slate-900"}`}>
              {link.name}
            </Link>
          ))}
          <a href={REPO} target="_blank" rel="noopener noreferrer" className="text-slate-600 hover:text-slate-900" aria-label="Source code on GitHub">
            <GitHubIcon className="w-5 h-5" />
          </a>
        </div>

        <button className="md:hidden text-slate-700 p-1" onClick={() => setOpen(!open)} aria-label="Menu" aria-expanded={open}>
          <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={open ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"} />
          </svg>
        </button>
      </div>

      {open && (
        <div className="md:hidden bg-white border-t border-slate-100 shadow-lg p-3 flex flex-col gap-1">
          {NAV_LINKS.map((link) => (
            <Link key={link.path} to={link.path} onClick={() => setOpen(false)}
                  className={`p-3 rounded-lg font-medium ${pathname === link.path ? "bg-blue-50 text-blue-700" : "text-slate-700 hover:bg-slate-50"}`}>
              {link.name}
            </Link>
          ))}
          <a href={REPO} target="_blank" rel="noopener noreferrer" className="p-3 rounded-lg font-medium text-slate-700 hover:bg-slate-50 flex items-center gap-2">
            <GitHubIcon className="w-5 h-5" /> Source code on GitHub
          </a>
        </div>
      )}
    </nav>
  );
}
