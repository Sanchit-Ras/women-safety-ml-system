import Hero from "../components/Hero";
import MapSection from "../components/MapSection";
import Stats from "../components/Stats";
import Findings from "../components/Findings";
import Methodology from "../components/Methodology";
import Architecture from "../components/Architecture";
import Decisions from "../components/Decisions";
import Performance from "../components/Performance";
import ResearchImpact from "../components/ResearchImpact";
import Problem from "../components/Problem";
export default function Research() {
  return (
    <div className="min-h-screen bg-slate-50 font-sans selection:bg-blue-100 selection:text-blue-900">
      <Hero />
      <Problem />
      <Stats />
      <Findings />
      <Methodology />
      <Architecture />
      <Decisions />
      <Performance />
      <MapSection />
      <ResearchImpact />
    </div>
  );
}
