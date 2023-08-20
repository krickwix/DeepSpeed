# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

hpu_xfail_tests = {
    "unit/moe/test_moe.py::TestPRMoE::test[2-True]":
    "Xfail, due to SW-118980.",
    "unit/moe/test_moe.py::TestPRMoE::test[2-False]":
    "Xfail, due to SW-118980.",
    "unit/moe/test_moe.py::TestMoE::test[False-4]":
    "Xfail, due to SW-118980.",
    "unit/moe/test_moe.py::TestMoE::test[False-2]":
    "Xfail, due to SW-118980.",
    "unit/moe/test_moe.py::TestMoE::test[True-2]":
    "Xfail, due to SW-118980.",
    "unit/moe/test_moe.py::TestMoE::test[True-4]":
    "Xfail, due to SW-118980.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe[4]":
    "Xfail, due to SW-118980.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroSupportedClientOptimizer::test[Adam]":
    "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[bfp16-bfp16]":
    "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[fp16-fp32]":
    "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[fp16-bfp16]":
    "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[bfp16-fp32]":
    "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroEmptyGrad::test":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[2-True]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[1-False]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[2-False]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[1-False]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[2-True]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[2-False]":
    "Xfail, due to SW-142972.",
    "unit/runtime/zero/test_zero.py::TestZeroOffloadOptim::test[False]":
    "Xfail, due to SW-142972.",
    "unit/runtime/zero/test_zero.py::TestZeroOffloadOptim::test[True]":
    "Xfail, due to SW-142972.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[1-False-Adam]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_hybrid_optimizer_state[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[2-False-Adam]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[2-True-deepspeed_adam]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_module_only[1]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[1-False-Adam]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[2-False-Adam]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_hybrid_optimizer_state[1]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[2-True-deepspeed_adam]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_module_only[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[1]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[1]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[1]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-True]":
    "Xfail, due to SW-116160.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-True]":
    "Xfail, due to SW-116160.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-False]":
    "Xfail, due to SW-116160.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-False]":
    "Xfail, due to SW-116160.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[3]":
    "Xfail, due to SW-100862.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[3]":
    "Xfail, due to SW-100862.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[3]":
    "Xfail, due to SW-100862.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[3]":
    "Xfail, due to SW-100862.",
    "unit/runtime/zero/test_zero_context_ancestry.py::TestSerialParamInit::test_subclass_param_init":
    "Xfail, due to SW-143227.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_fixed_dp[True-False-True]":
    "Xfail, due to SW-138014.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_fixed_dp[True-True-True]":
    "Xfail, due to SW-138014.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test[topo_config0]":
    "Xfail, due to SW-148103.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test[topo_config1]":
    "Xfail, due to SW-148103.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test[topo_config2]":
    "Xfail, due to SW-148103.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[EleutherAI/gpt-neo-1.3B-bsz=2]":
    "Xfail, due to SW-149474.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[EleutherAI/gpt-neo-1.3B-bsz=1]":
    "Xfail, due to SW-149474.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[facebook/opt-1.3b-bsz=2]":
    "Xfail, due to SW-149474.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[facebook/opt-1.3b-bsz=1]":
    "Xfail, due to SW-149474.",
    "unit/hybrid_engine/test_he_llama.py::TestHybridEngineLlama::test_functionality[huggyllama/llama-7b-bsz=1]":
    "Xfail, due to SW-149474.",
    "unit/hybrid_engine/test_he_llama.py::TestHybridEngineLlama::test_functionality[huggyllama/llama-7b-bsz=2]":
    "Xfail, due to SW-149474.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[default-bfp16]":
    "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[default-fp32]":
    "Xfail, due to SW-142972.",
    "unit/comm/test_dist.py::TestDistInitNoEnv::test":
    "Xfail, same failure observed on Vanila 0.9.4 as well.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[2]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[1]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[4]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[2]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[4]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[4]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[2]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[1]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[1]":
    "Xfail, due to SW-149262.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[False-2]":
    "Xfail, due to SW-142972.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[True-2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_not_load_optimizer_state[1]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_module_only[1]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_optimizer_state[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_module_only[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_not_load_optimizer_state[2]":
    "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_optimizer_state[1]":
    "Xfail, due to SW-142972.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[True-3]":
    "Xfail, due to SW-148819.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[False-3]":
    "Xfail, due to SW-148819.",
    "unit/runtime/zero/test_ignore_unused_parameters.py::TestStage2IgnoreUnusedParameters::test[False]":
    "Xfail, due to SW-149596.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-1-2]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-1-4]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-1-2]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-1-2]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-1-4]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-1-2]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-2-2]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-1-4]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-2-2]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-2-2]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-2-2]":
    "Xfail, due to SW-149735.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-1-4]":
    "Xfail, due to SW-149735.",
}

g1_xfail_tests = {
    "unit/runtime/zero/test_zero.py::TestZeroAdamOptimizerStepCount::test[2]":
    "Xfail, due to SW-137978.",
    "unit/runtime/zero/test_zero.py::TestZeroAdamOptimizerStepCount::test[3]":
    "Xfail, due to SW-137978.",
    "unit/runtime/zero/test_zero.py::TestZeroAdamOptimizerStepCount::test[1]":
    "Xfail, due to SW-137978.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_optimizer_state[3]":
    "Xfail, due to SW-149619.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_not_load_optimizer_state[3]":
    "Xfail, due to SW-149619.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_module_only[3]":
    "Xfail, due to SW-149619.",
}

g2_xfail_tests = {
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-neo-125M-int8]":
    "Xfail, due to SW-123615.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[bigscience/bloom-560m-int8]":
    "Xfail, due to SW-123615.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-j-6B-int8]":
    "Xfail, due to SW-123615.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[facebook/opt-125m-int8]":
    "Xfail, due to SW-123615.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[bfp16-fp16]": "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[fp16-fp16]": "Xfail, due to SW-142972.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[3]":
    "Xfail, due to SW-144290.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[3]":
    "Xfail, due to SW-144290.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyGrad::test[2]": "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyGrad::test[1]": "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[Adam-2]":
    "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[Adam-1]":
    "Xfail, due to SW-142972.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-True-1]": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-True-2]": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-True-2]": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-True-3]": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-True-1]": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-True-3]": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-3]":
    "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-1]":
    "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-2]":
    "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[True-2]": "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[True-1]": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16OptimizerForMoE::test_unfused_gradnorm":
    "Xfail, due to SW-118980.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyGrad::test[3]": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestZero2ReduceScatterOff::test": "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[False-1]":
    "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[False-3]":
    "Xfail, due to SW-145264.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[False-2]":
    "Xfail, due to SW-145264.",
    "unit/inference/test_model_profiling.py::TestModelProfiling::test[False-False-facebook/opt-125m-text-generation]":
    "Xfail, due to SW-143277/SW-124433.",
    "unit/inference/test_model_profiling.py::TestModelProfiling::test[True-False-facebook/opt-125m-text-generation]":
    "Xfail, due to SW-143277/SW-124433.",
    "unit/inference/test_model_profiling.py::TestModelProfiling::test[False-False-bigscience/bloom-560m-text-generation]":
    "Xfail, due to SW-143277/SW-124433.",
    "unit/inference/test_model_profiling.py::TestModelProfiling::test[False-False-gpt2-text-generation]":
    "Xfail, due to SW-143277/SW-124433.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[default-fp16]": "Xfail, due to SW-142972.",
    "unit/inference/test_inference.py::TestMPSize::test[fp16-noCG-bloom]": "Xfail, due to SW-149565.",
}

gpu_xfail_tests = {
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[64-fp16]":
    "Failing on GPU vanilla 0.9.4.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[128-fp16]":
    "Failing on GPU vanilla 0.9.4.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[1024-fp16]":
    "Failing on GPU vanilla 0.9.4.",
    "unit/comm/test_dist.py::TestDistInitNoEnv::test":
    "Failing on GPU vanilla 0.9.4.",
    "unit/ops/quantizer/test_fake_quantization.py::test_fake_quant_dequant[1-tensor_shape0]":
    "Failing on GPU vanilla 0.9.4.",
    "unit/ops/quantizer/test_fake_quantization.py::test_fake_quant_dequant[1-tensor_shape1]":
    "Failing on GPU vanilla 0.9.4.",
    "unit/ops/quantizer/test_fake_quantization.py::test_fake_quant_dequant[16-tensor_shape0]":
    "Failing on GPU vanilla 0.9.4.",
    "unit/ops/quantizer/test_fake_quantization.py::test_fake_quant_dequant[16-tensor_shape1]":
    "Failing on GPU vanilla 0.9.4.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[1]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[2]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[2]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[1]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[4]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[4]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[4]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[1]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[2]":
    "Xfail, due to SW-149262.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-False]":
    "Xfail, due to SW-149686.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-False]":
    "Xfail, due to SW-149686.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-True]":
    "Xfail, due to SW-149686.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-True]":
    "Xfail, due to SW-149686.",
    "unit/runtime/zero/test_zero.py::TestZeroOffloadOptim::test[True]":
    "Xfail, due to SW-149688.",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-j-6B]":
    "Large Model, OOM",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-gpt2-xl]":
    "Large Model, OOM",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B]":
    "Large Model, OOM",
    "unit/hybrid_engine/test_he_llama.py::TestHybridEngineLlama::test_functionality[huggyllama/llama-7b-bsz=2]":
    "Random Failure due to UnicodeEncodeError.",
    "unit/hybrid_engine/test_he_llama.py::TestHybridEngineLlama::test_functionality[huggyllama/llama-7b-bsz=1]":
    "Random Failure due to UnicodeEncodeError.",
}
