"use client";

import dynamic from "next/dynamic";
import type { ComponentType } from "react";
import type { SimId } from "@/lib/types";
import { Inline } from "@/components/content/Inline";

const loading = () => <div className="card my-6 h-64 animate-pulse bg-surface-2" aria-label="Loading simulator" />;

/** Every simulator is code-split so a lesson only downloads what it embeds. */
const COMPONENTS: Record<SimId, ComponentType> = {
  network: dynamic(() => import("./NetworkSim"), { ssr: false, loading }),
  convolution: dynamic(() => import("./ConvolutionSim"), { ssr: false, loading }),
  "cnn-pipeline": dynamic(() => import("./CnnPipelineSim"), { ssr: false, loading }),
  attention: dynamic(() => import("./AttentionSim"), { ssr: false, loading }),
  neuron: dynamic(() => import("./NeuronSim"), { ssr: false, loading }),
  activations: dynamic(() => import("./ActivationsSim"), { ssr: false, loading }),
  optimizers: dynamic(() => import("./OptimizersSim"), { ssr: false, loading }),
  "gradient-descent": dynamic(() => import("./GradientDescentSim"), { ssr: false, loading }),
  "loss-functions": dynamic(() => import("./LossSim"), { ssr: false, loading }),
  fitting: dynamic(() => import("./FittingSim"), { ssr: false, loading }),
  matmul: dynamic(() => import("./MatmulSim"), { ssr: false, loading }),
  "chain-rule": dynamic(() => import("./ChainRuleSim"), { ssr: false, loading }),
  "positional-encoding": dynamic(() => import("./PositionalEncodingSim"), { ssr: false, loading }),
  lstm: dynamic(() => import("./LstmSim"), { ssr: false, loading }),
  tokenizer: dynamic(() => import("./TokenizerSim"), { ssr: false, loading }),
  "linear-transform": dynamic(() => import("./LinearTransformSim"), { ssr: false, loading }),
  distributions: dynamic(() => import("./DistributionsSim"), { ssr: false, loading }),
  entropy: dynamic(() => import("./EntropySim"), { ssr: false, loading }),
  "logistic-regression": dynamic(() => import("./LogisticRegressionSim"), { ssr: false, loading }),
  approximator: dynamic(() => import("./ApproximatorSim"), { ssr: false, loading }),
  "gradient-flow": dynamic(() => import("./GradientFlowSim"), { ssr: false, loading }),
  batchnorm: dynamic(() => import("./BatchNormSim"), { ssr: false, loading }),
  dropout: dynamic(() => import("./DropoutSim"), { ssr: false, loading }),
  "lr-schedule": dynamic(() => import("./LrScheduleSim"), { ssr: false, loading }),
  regularization: dynamic(() => import("./RegularizationSim"), { ssr: false, loading }),
  "metric-learning": dynamic(() => import("./MetricLearningSim"), { ssr: false, loading }),
  "receptive-field": dynamic(() => import("./ReceptiveFieldSim"), { ssr: false, loading }),
  "conv-variants": dynamic(() => import("./ConvVariantsSim"), { ssr: false, loading }),
  rnn: dynamic(() => import("./RnnSim"), { ssr: false, loading }),
  bptt: dynamic(() => import("./BpttSim"), { ssr: false, loading }),
  gru: dynamic(() => import("./GruSim"), { ssr: false, loading }),
  transformer: dynamic(() => import("./TransformerSim"), { ssr: false, loading }),
  "seq2seq-attention": dynamic(() => import("./Seq2SeqAttentionSim"), { ssr: false, loading }),
  detection: dynamic(() => import("./DetectionSim"), { ssr: false, loading }),
  "word-embeddings": dynamic(() => import("./WordEmbeddingsSim"), { ssr: false, loading }),
  autoencoder: dynamic(() => import("./AutoencoderSim"), { ssr: false, loading }),
  vae: dynamic(() => import("./VaeSim"), { ssr: false, loading }),
  gan: dynamic(() => import("./GanSim"), { ssr: false, loading }),
  diffusion: dynamic(() => import("./DiffusionSim"), { ssr: false, loading }),
  broadcasting: dynamic(() => import("./BroadcastingSim"), { ssr: false, loading }),
};

export function SimEmbed({ id, caption }: { id: SimId; caption?: string }) {
  const C = COMPONENTS[id];
  return (
    <div className="not-prose">
      <C />
      {caption && (
        <p className="-mt-3 mb-6 text-center text-sm text-muted">
          <Inline text={caption} />
        </p>
      )}
    </div>
  );
}
