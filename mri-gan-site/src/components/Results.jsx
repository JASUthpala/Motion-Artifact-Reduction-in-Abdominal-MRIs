import { useState } from "react";

export default function Results() {
  const [open, setOpen] = useState(null);

  const data = [
    {
      title: "SSIM",
      desc: "Structural similarity improved significantly after GAN correction"
    },
    {
      title: "PSNR",
      desc: "Peak signal-to-noise ratio increased showing better reconstruction"
    }
  ];

  return (
    <div id="results" className="p-10">

      <h2 className="text-2xl font-bold mb-4">Quantitative Results</h2>

      {data.map((item, i) => (
        <div
          key={i}
          className="bg-[#161f33] p-4 rounded-xl mb-3 cursor-pointer"
          onClick={() => setOpen(open === i ? null : i)}
        >
          <h3 className="font-bold">{item.title}</h3>

          {open === i && (
            <p className="text-gray-300 mt-2">
              {item.desc}
            </p>
          )}
        </div>
      ))}

    </div>
  );
}