import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import Pipeline from "../components/Pipeline";
import MRICompare from "../components/MRICompare";
import Results from "../components/Results";
import Publications from "../components/Publications";

export default function Home() {
  return (
    <div>

      <Navbar />

      <Hero />

      <Pipeline />

      {/* 🔥 THIS IS THE SLIDER SECTION */}
      <MRICompare />

      <Results />

      <Publications />

    </div>
  );
}