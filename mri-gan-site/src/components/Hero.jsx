import { motion } from "framer-motion";

export default function Hero() {
  return (
    <div id="home" className="text-center py-20 px-6">

      <motion.h1
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-5xl font-bold"
      >
        Motion Artifact Reduction in Abdominal MRI
      </motion.h1>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-gray-400 mt-5 max-w-2xl mx-auto"
      >
        GAN-based deep learning framework for respiratory motion correction with
        quantitative evaluation and clinical-grade reconstruction quality.
      </motion.p>

      <div className="mt-8 flex justify-center gap-4">
        <a href="#pipeline" className="bg-blue-600 px-4 py-2 rounded">
          View Pipeline
        </a>

        <a href="#compare" className="bg-gray-700 px-4 py-2 rounded">
          See Results
        </a>
      </div>

    </div>
  );
}