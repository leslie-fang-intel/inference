# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020 INTEL CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import mlperf_loadgen as lg
import subprocess


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",
                        choices=["pytorch", "onnxruntime", "tf", "ov"],
                        default="pytorch",
                        help="Backend")
    parser.add_argument(
        "--scenario",
        choices=["SingleStream", "Offline", "Server", "MultiStream"],
        default="Offline",
        help="Scenario")
    parser.add_argument("--accuracy",
                        action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--mlperf_conf",
                        default="build/mlperf.conf",
                        help="mlperf rules config")
    parser.add_argument("--user_conf",
                        default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument(
        "--model_dir",
        default=
        "build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1",
        help="Path to the directory containing plans.pkl")
    parser.add_argument("--model", help="Path to the ONNX, OpenVINO, or TF model")
    parser.add_argument("--preprocessed_data_dir",
                        default="build/preprocessed_data",
                        help="path to preprocessed data")
    parser.add_argument("--performance_count",
                        type=int,
                        default=16,
                        help="performance count")
    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
    parser.add_argument('--configure-dir', default='configure.json', type=str, metavar='PATH',
                        help = 'path to int8 configures, default file name is configure.json')
    parser.add_argument('--calibration', action='store_true', default=False,
                        help='doing calibration step')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='enable ipex int8 path')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='enable benchmark test model. False by default with mlperf running mode')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='batchsize to run the benchmark')
    parser.add_argument('--steps', type=int, default=30,
                        help='total running steps')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='warm up steps')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='enable ipex jit fusionpath')
    parser.add_argument('--autocast', action='store_true', default=False,
                        help='enable autocast')
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


def main_mlperf(args):
    print("Start the main_mlperf")
    if args.backend == "pytorch":
        from pytorch_SUT import get_pytorch_sut
        sut = get_pytorch_sut(args.model_dir, args.preprocessed_data_dir,
                              args.performance_count, use_ipex=args.ipex,
                              use_int8=args.int8, calibration=args.calibration, 
                              configure_dir=args.configure_dir, use_jit=args.jit)
    elif args.backend == "onnxruntime":
        from onnxruntime_SUT import get_onnxruntime_sut
        sut = get_onnxruntime_sut(args.model, args.preprocessed_data_dir,
                                  args.performance_count)
    elif args.backend == "tf":
        from tf_SUT import get_tf_sut
        sut = get_tf_sut(args.model, args.preprocessed_data_dir,
                         args.performance_count)
    elif args.backend == "ov":
        from ov_SUT import get_ov_sut
        sut = get_ov_sut(args.model, args.preprocessed_data_dir,
                         args.performance_count)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario)
    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_path = "build/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    print("Running Loadgen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)

    if args.accuracy:
        print("Running accuracy script...")
        cmd = "python3 accuracy-brats.py"
        subprocess.check_call(cmd, shell=True)

    print("Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl.qsl)

def main_benchmark(args):
    print("Start the main_benchmark")
    from pytorch_SUT import get_pytorch_sut
    sut = get_pytorch_sut(args.model_dir, args.preprocessed_data_dir,
                          args.performance_count, use_ipex=args.ipex,
                          use_int8=args.int8, calibration=args.calibration, configure_dir=args.configure_dir, use_jit=args.jit)
    sut.benchmark(args.batchsize, args.steps, args.warmup_steps)

def main():
    args = get_args()
    print(args)
    if args.benchmark and not args.accuracy:
        main_benchmark(args)
    else:
        main_mlperf(args)

if __name__ == "__main__":
    main()
