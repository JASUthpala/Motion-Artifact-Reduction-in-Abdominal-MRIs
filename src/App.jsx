import { motion } from "framer-motion";
import {
  FaGithub,
  FaBrain,
  FaChartLine,
  FaDatabase,
} from "react-icons/fa";

function App() {
  return (
    <div className="bg-[#050816] text-white min-h-screen">

      {/* HERO */}
      <section className="min-h-screen flex flex-col justify-center items-center text-center px-6">

        <motion.h1
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="text-5xl md:text-7xl font-bold"
        >
          GAN-Based MRI Motion Correction
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-6 text-gray-300 max-w-3xl text-lg"
        >
          Deep Learning for Removing Respiratory Motion
          Artifacts in Abdominal MRI using GANs,
          CycleGANs, and Dense U-Net Architectures.
        </motion.p>

        <motion.a
          whileHover={{ scale: 1.1 }}
          href="#results"
          className="mt-10 bg-blue-600 px-8 py-4 rounded-xl text-white no-underline"
        >
          View Results
        </motion.a>
      </section>

      {/* ABSTRACT */}
      <section className="py-24 px-6 max-w-6xl mx-auto">

        <h2 className="text-4xl font-bold mb-10">
          Abstract
        </h2>

        <p className="text-gray-300 leading-8 text-lg">
          Respiratory motion during MRI acquisition introduces
          severe ghosting and blurring artifacts that reduce
          diagnostic image quality. This research investigates
          GAN-based deep learning approaches for retrospective
          motion correction in abdominal MRI. We explore
          Dense U-Net generators, perceptual losses, and
          adversarial training strategies for restoring
          motion-corrupted MRI slices while preserving
          anatomical details.
        </p>
      </section>

      {/* PIPELINE */}
      <section className="py-24 px-6 bg-[#0b1023]">

        <div className="max-w-6xl mx-auto">

          <h2 className="text-4xl font-bold mb-16">
            Research Pipeline
          </h2>

          <div className="grid md:grid-cols-4 gap-8">

            <div className="bg-[#131b36] p-8 rounded-2xl">
              <FaDatabase size={40} />
              <h3 className="text-2xl mt-4 mb-4">
                Dataset
              </h3>

              <p className="text-gray-400">
                MRI abdominal datasets with respiratory motion artifacts.
              </p>
            </div>

            <div className="bg-[#131b36] p-8 rounded-2xl">
              <FaBrain size={40} />
              <h3 className="text-2xl mt-4 mb-4">
                GAN Training
              </h3>

              <p className="text-gray-400">
                Dense U-Net generator with adversarial and perceptual loss.
              </p>
            </div>

            <div className="bg-[#131b36] p-8 rounded-2xl">
              <FaChartLine size={40} />
              <h3 className="text-2xl mt-4 mb-4">
                Evaluation
              </h3>

              <p className="text-gray-400">
                SSIM, PSNR, FSIM, IQI metrics for image quality analysis.
              </p>
            </div>

            <div className="bg-[#131b36] p-8 rounded-2xl">
              <FaGithub size={40} />
              <h3 className="text-2xl mt-4 mb-4">
                Deployment
              </h3>

              <p className="text-gray-400">
                Research showcase deployed using GitHub Pages.
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* RESULTS */}
      <section
        id="results"
        className="py-24 px-6 max-w-6xl mx-auto"
      >

        <h2 className="text-4xl font-bold mb-16">
          Quantitative Results
        </h2>

        <div className="grid md:grid-cols-4 gap-8">

          <div className="bg-[#131b36] p-8 rounded-2xl text-center">
            <h3 className="text-5xl font-bold text-blue-400">
              0.92
            </h3>

            <p className="mt-4 text-gray-300">
              FSIM
            </p>
          </div>

          <div className="bg-[#131b36] p-8 rounded-2xl text-center">
            <h3 className="text-5xl font-bold text-green-400">
              0.91
            </h3>

            <p className="mt-4 text-gray-300">
              SSIM
            </p>
          </div>

          <div className="bg-[#131b36] p-8 rounded-2xl text-center">
            <h3 className="text-5xl font-bold text-purple-400">
              35+
            </h3>

            <p className="mt-4 text-gray-300">
              PSNR
            </p>
          </div>

          <div className="bg-[#131b36] p-8 rounded-2xl text-center">
            <h3 className="text-5xl font-bold text-pink-400">
              GAN
            </h3>

            <p className="mt-4 text-gray-300">
              Dense U-Net
            </p>
          </div>

        </div>
      </section>

      {/* REFERENCES */}
      <section className="py-24 px-6 bg-[#0b1023]">

        <div className="max-w-6xl mx-auto">

          <h2 className="text-4xl font-bold mb-12">
            References
          </h2>

          <div className="space-y-6 text-gray-300">

            <p>
              Zhu et al. — CycleGAN (ICCV 2017)
            </p>

            <p>
              Isola et al. — Pix2Pix (CVPR 2017)
            </p>

            <p>
              Wang et al. — SSIM Metric
            </p>

            <p>
              Deep Learning for Retrospective Motion Correction in MRI
            </p>

          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="py-10 text-center text-gray-500">
        MRI Motion Correction Research © 2026
      </footer>

    </div>
  );
}

export default App;