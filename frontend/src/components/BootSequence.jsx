import { useState, useEffect } from "react";

const BootSequence = () => {
  const [lines, setLines] = useState([]);
  const [currentLineIndex, setCurrentLineIndex] = useState(0);

  const bootLogs = [
    "INITIALIZING_KERNEL...",
    "LOADING_NEURAL_WEIGHTS...",
    "VERIFYING_INTEGRITY...",
    "CONNECTING_TO_SECURE_SERVER...",
    "ESTABLISHING_HANDSHAKE...",
    "DECRYPTING_Protocol_v4...",
    "ACCESS_GRANTED.",
    "SYSTEM_READY.",
  ];

  useEffect(() => {
    if (currentLineIndex < bootLogs.length) {
      const timeout = setTimeout(
        () => {
          setLines((prev) => [...prev, bootLogs[currentLineIndex]]);
          setCurrentLineIndex((prev) => prev + 1);
        },
        Math.random() * 300 + 100,
      ); // Random delay between 100ms and 400ms

      return () => clearTimeout(timeout);
    }
  }, [currentLineIndex]);

  return (
    <div className="flex items-center justify-center min-h-[400px] bg-black text-green-500 font-mono p-6 rounded-xl shadow-2xl border border-green-900/50">
      <div className="w-full max-w-lg">
        <div className="mb-4 text-xs text-green-700 border-b border-green-900/50 pb-2 flex justify-between">
          <span>TERMINAL_Tv1.0</span>
          <span>SECURE_CONNECTION</span>
        </div>
        <div className="space-y-2 h-64 overflow-y-auto font-bold text-sm sm:text-base">
          {lines.map((line, index) => (
            <div key={index} className="flex">
              <span className="mr-2 opacity-50">{">"}</span>
              <span className="typewriter-text">{line}</span>
            </div>
          ))}
          <div className="animate-pulse">_</div>
        </div>
      </div>
    </div>
  );
};

export default BootSequence;
