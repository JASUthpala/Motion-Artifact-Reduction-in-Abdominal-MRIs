import { motion } from "framer-motion";

const steps = [
  "MRI Input",
  "Preprocessing",
  "GAN Generator",
  "Discriminator",
  "Reconstructed MRI"
];

export default function Pipeline() {
  return (
    <div id="pipeline" className="p-10">

      <h2 className="text-2xl font-bold mb-6">Pipeline</h2>

      <div className="grid md:grid-cols-5 gap-4">

        {steps.map((step, i) => (
          <motion.div
            key={i}
            whileHover={{ scale: 1.1 }}
            className="bg-[#161f33] p-4 rounded-xl text-center"
          >
            {step}
          </motion.div>
        ))}

      </div>
    </div>
  );
}