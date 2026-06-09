import { ReactCompareSlider, ReactCompareSliderImage } from "react-compare-slider";
import { motion } from "framer-motion";

export default function MRICompare() {
  return (
    <div id="compare" className="p-10">

      <motion.h2
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        className="text-2xl font-bold mb-4"
      >
        MRI Motion Correction Comparison
      </motion.h2>

      <p className="text-gray-400 mb-5">
        Drag the slider to compare motion-corrupted vs GAN-corrected MRI
      </p>

      <div className="rounded-xl overflow-hidden border border-gray-700">
        <ReactCompareSlider
          itemOne={
            <ReactCompareSliderImage
              src="/before.png"
              alt="Motion Corrupted MRI"
            />
          }
          itemTwo={
            <ReactCompareSliderImage
              src="/after.png"
              alt="GAN Corrected MRI"
            />
          }
        />
      </div>

    </div>
  );
}