import { useState } from "react";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex justify-between items-center px-10 py-5 bg-[#11182a] sticky top-0 z-50">

      <h1 className="font-bold text-lg">
        MRI GAN Research
      </h1>

      {/* Desktop menu */}
      <div className="hidden md:flex space-x-6 text-sm">
        <a href="#home">Home</a>
        <a href="#pipeline">Pipeline</a>
        <a href="#compare">Comparison</a>
        <a href="#results">Results</a>
        <a href="#refs">References</a>
      </div>

      {/* Mobile button */}
      <button
        className="md:hidden"
        onClick={() => setOpen(!open)}
      >
        ☰
      </button>

      {/* Mobile menu */}
      {open && (
        <div className="absolute top-16 right-5 bg-[#161f33] p-4 rounded-lg space-y-2">
          <a href="#pipeline">Pipeline</a>
          <a href="#compare">Comparison</a>
          <a href="#results">Results</a>
          <a href="#refs">References</a>
        </div>
      )}
    </div>
  );
}