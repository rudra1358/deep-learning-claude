import type { Lesson } from "@/lib/types";

import numpyArrays from "./m00-numpy-arrays";
import vectorization from "./m00-vectorization";
import plottingData from "./m00-plotting-data";
import aiMlDl from "./m01-ai-ml-dl";
import history from "./m01-history";
import dlWorkflow from "./m01-dl-workflow";
import vectorsMatrices from "./m02-vectors-matrices";
import normsSpaces from "./m02-norms-spaces";
import eigenSvd from "./m02-eigen-svd";
import derivativesChainRule from "./m02-derivatives-chain-rule";
import jacobiansHessians from "./m02-jacobians-hessians";
import probability from "./m02-probability";
import mleMap from "./m02-mle-map";
import informationTheory from "./m02-information-theory";
import optimizationMath from "./m02-optimization-math";
import linearLogisticRegression from "./m03-linear-logistic-regression";
import artificialNeuron from "./m03-artificial-neuron";
import activationFunctions from "./m03-activation-functions";
import multilayerNetworks from "./m03-multilayer-networks";
import trainingBasics from "./m03-training-basics";
import forwardPropagation from "./m04-forward-propagation";
import backpropagation from "./m04-backpropagation";
import gradientProblems from "./m04-gradient-problems";
import gradientChecking from "./m04-gradient-checking";
import regressionLosses from "./m05-regression-losses";
import classificationLosses from "./m05-classification-losses";
import advancedLosses from "./m05-advanced-losses";
import gradientDescent from "./m06-gradient-descent";
import momentumAdaptive from "./m06-momentum-adaptive";
import lrSchedules from "./m06-lr-schedules";
import initialization from "./m07-initialization";
import normalization from "./m07-normalization";
import regularization from "./m07-regularization";
import residualConnections from "./m07-residual-connections";
import imagesAsTensors from "./m08-images-as-tensors";
import convolution from "./m08-convolution";
import poolingReceptiveField from "./m08-pooling-receptive-field";
import convVariants from "./m08-conv-variants";
import cnnArchitectures from "./m08-cnn-architectures";
import rnnBasics from "./m09-rnn-basics";
import bptt from "./m09-bptt";
import lstm from "./m09-lstm";
import gruVariants from "./m09-gru-variants";
import attentionIntro from "./m10-attention-intro";
import scaledDotProductAttention from "./m10-scaled-dot-product-attention";
import multiHeadAttention from "./m10-multi-head-attention";
import positionalEncoding from "./m10-positional-encoding";
import transformerArchitecture from "./m11-transformer-architecture";
import tokensEmbeddings from "./m11-tokens-embeddings";
import trainingInference from "./m11-training-inference";
import transformerFamilies from "./m11-transformer-families";
import transferLearning from "./m12-transfer-learning";
import objectDetection from "./m12-object-detection";
import segmentation from "./m12-segmentation";
import visionTasks from "./m12-vision-tasks";
import textPreprocessing from "./m13-text-preprocessing";
import wordEmbeddings from "./m13-word-embeddings";
import nlpTasks from "./m13-nlp-tasks";
import languageModeling from "./m13-language-modeling";
import autoencoders from "./m14-autoencoders";
import vae from "./m14-vae";
import gans from "./m14-gans";
import diffusion from "./m14-diffusion";

/** All fully-written lessons. Topics without an entry here show their syllabus + "in development". */
export const LESSONS: Lesson[] = [
  numpyArrays,
  vectorization,
  plottingData,
  aiMlDl,
  history,
  dlWorkflow,
  vectorsMatrices,
  normsSpaces,
  eigenSvd,
  derivativesChainRule,
  jacobiansHessians,
  probability,
  mleMap,
  informationTheory,
  optimizationMath,
  linearLogisticRegression,
  artificialNeuron,
  activationFunctions,
  multilayerNetworks,
  trainingBasics,
  forwardPropagation,
  backpropagation,
  gradientProblems,
  gradientChecking,
  regressionLosses,
  classificationLosses,
  advancedLosses,
  gradientDescent,
  momentumAdaptive,
  lrSchedules,
  initialization,
  normalization,
  regularization,
  residualConnections,
  imagesAsTensors,
  convolution,
  poolingReceptiveField,
  convVariants,
  cnnArchitectures,
  rnnBasics,
  bptt,
  lstm,
  gruVariants,
  attentionIntro,
  scaledDotProductAttention,
  multiHeadAttention,
  positionalEncoding,
  transformerArchitecture,
  tokensEmbeddings,
  trainingInference,
  transformerFamilies,
  transferLearning,
  objectDetection,
  segmentation,
  visionTasks,
  textPreprocessing,
  wordEmbeddings,
  nlpTasks,
  languageModeling,
  autoencoders,
  vae,
  gans,
  diffusion,
];

const BY_ID = new Map(LESSONS.map((l) => [l.id, l]));

export function getLesson(id: string): Lesson | undefined {
  return BY_ID.get(id);
}

export function hasLesson(id: string) {
  return BY_ID.has(id);
}
