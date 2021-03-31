# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

import array
import json
import os
import sys
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import torch
import torch.nn.functional as F
from brats_QSL import get_brats_QSL
import time

sys.path.insert(0, os.path.join(os.getcwd(), "nnUnet"))
from nnunet.training.model_restore import load_model_and_checkpoint_files

class _3DUNET_PyTorch_SUT():
    def __init__(self, model_dir, preprocessed_data_dir, performance_count, folds, checkpoint_name, use_ipex,
                     use_int8, calibration, configure_dir, use_autocast, use_jit):

        print("Loading PyTorch model...")
        model_path = os.path.join(model_dir, "plans.pkl")
        assert os.path.isfile(model_path), "Cannot find the model file {:}!".format(model_path)
        self.trainer, params = load_model_and_checkpoint_files(model_dir, folds, fp16=False, checkpoint_name=checkpoint_name)
        self.trainer.load_checkpoint_ram(params[0], False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")
        self.qsl = get_brats_QSL(preprocessed_data_dir, performance_count)

        self.use_ipex = use_ipex
        self.use_int8 = use_int8
        self.calibration = calibration
        self.configure_dir = configure_dir
        self.use_autocast = use_autocast
        self.use_jit = use_jit

    def issue_queries(self, query_samples):
        with torch.no_grad():
            #print(self.trainer.network)
            #print("-----------------------------------------print mkldnn model-----------------------------")
            #from torch.utils import mkldnn as mkldnn_utils
            #mkldnn_model = mkldnn_utils.to_mkldnn(self.trainer.network)
            #print(mkldnn_model)
            if self.use_ipex:
                import intel_pytorch_extension as ipex

            conf = None
            if self.use_ipex:
                if self.use_int8 and self.calibration:
                    # INT8 calibration
                    conf = ipex.AmpConf(torch.int8)
                elif self.use_int8:
                    # INT8 inference
                    conf = ipex.AmpConf(torch.int8, self.configure_dir)
                else:
                    # BF16 inference
                    conf = ipex.AmpConf(torch.bfloat16)

            model = self.trainer.network

            if self.use_autocast:
                # use autocast
                model.eval()
                if self.use_jit:
                    print("Enable jit")
                    image = torch.randn(1, 4, 224, 224, 160).float().to(self.device)
                    with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
                        model = torch.jit.trace(model, image)
                    print(model.graph_for(image))
                    for i in range(len(query_samples)):
                        data = self.qsl.get_features(query_samples[i].index)

                        print("Processing sample id {:d} with shape = {:}".format(query_samples[i].index, data.shape))
                        image = torch.from_numpy(data[np.newaxis,...]).float().to(self.device)

                        #print(image.size())#torch.Size([1, 4, 224, 224, 160])
                        #dummy data
                        #image = torch.randn(28, 4, 224, 224, 160).float().to(self.device)

                        start_time = time.time()

                        #with torch.autograd.profiler.profile() as prof:
                        #    output = model(image)
                        #with open("fp32.prof", "w") as prof_f:
                        #    prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
                        #    prof.export_chrome_trace("fp32.json")
                        output = model(image)
                        #print(type(model))
                        end_time = time.time()

                        print("Running time for one sample is: {}".format(end_time-start_time))

                        #print(output[0].cpu().numpy())

                        output = output[0].to(torch.float32).cpu().numpy().astype(np.float16)

                        transpose_forward = self.trainer.plans.get("transpose_forward")
                        transpose_backward = self.trainer.plans.get("transpose_backward")
                        assert transpose_forward == [0, 1, 2], "Unexpected transpose_forward {:}".format(transpose_forward)
                        assert transpose_backward == [0, 1, 2], "Unexpected transpose_backward {:}".format(transpose_backward)

                        response_array = array.array("B", output.tobytes())
                        bi = response_array.buffer_info()
                        response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                        lg.QuerySamplesComplete([response])
                    return

                with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
                    print("Enable autocast in accuracy")
                    for i in range(len(query_samples)):
                        data = self.qsl.get_features(query_samples[i].index)

                        print("Processing sample id {:d} with shape = {:}".format(query_samples[i].index, data.shape))
                        image = torch.from_numpy(data[np.newaxis,...]).float().to(self.device)

                        #print(image.size())#torch.Size([1, 4, 224, 224, 160])
                        #dummy data
                        #image = torch.randn(28, 4, 224, 224, 160).float().to(self.device)

                        start_time = time.time()

                        #with torch.autograd.profiler.profile() as prof:
                        #    output = model(image)
                        #with open("fp32.prof", "w") as prof_f:
                        #    prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
                        #    prof.export_chrome_trace("fp32.json")
                        output = model(image)
                        end_time = time.time()

                        print("Running time for one sample is: {}".format(end_time-start_time))

                        #print(output[0].cpu().numpy())

                        output = output[0].to(torch.float32).cpu().numpy().astype(np.float16)

                        transpose_forward = self.trainer.plans.get("transpose_forward")
                        transpose_backward = self.trainer.plans.get("transpose_backward")
                        assert transpose_forward == [0, 1, 2], "Unexpected transpose_forward {:}".format(transpose_forward)
                        assert transpose_backward == [0, 1, 2], "Unexpected transpose_backward {:}".format(transpose_backward)

                        response_array = array.array("B", output.tobytes())
                        bi = response_array.buffer_info()
                        response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                        lg.QuerySamplesComplete([response])
                return
            #print(type(model))
            #model.eval()
            #torch.save(model, "test.pth")
            if self.use_ipex:
                model = model.to(device = ipex.DEVICE)
                model.eval()
            elif self.use_jit:
                print("Enable jit")
                model.eval()
                image = torch.randn(1, 4, 224, 224, 160).float().to(self.device)
                model = torch.jit.trace(model, image)
                #print(model)
                print(model.graph_for(image))
                #return
            else:
                #print("pass-----------")
                #print(model)
                pass
                #from torch.utils import mkldnn as mkldnn_utils
                #model = mkldnn_utils.to_mkldnn(model)

            for i in range(len(query_samples)):
                data = self.qsl.get_features(query_samples[i].index)

                print("Processing sample id {:d} with shape = {:}".format(query_samples[i].index, data.shape))
                image = torch.from_numpy(data[np.newaxis,...]).float().to(self.device)

                #print(image.size())#torch.Size([1, 4, 224, 224, 160])
                #dummy data
                #image = torch.randn(28, 4, 224, 224, 160).float().to(self.device)

                start_time = time.time()
                if self.use_ipex and self.use_int8 and self.calibration:
                    # INT8 calibration
                    with ipex.AutoMixPrecision(conf, running_mode="calibration"):
                        image = image.to(device = ipex.DEVICE)
                        output = model(image)
                elif self.use_ipex and self.use_int8:
                    # INT8 inference
                    image = image.to(device = ipex.DEVICE)
                    with ipex.AutoMixPrecision(conf, running_mode="inference"):
                        output = model(image)
                elif self.use_ipex:
                    # BF16 inference
                    image = image.to(device = ipex.DEVICE)
                    with ipex.AutoMixPrecision(conf, running_mode="inference"):
                        output = model(image)
                else:
                    #with torch.autograd.profiler.profile() as prof:
                    #    output = model(image)
                    #with open("fp32.prof", "w") as prof_f:
                    #    prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
                    #    prof.export_chrome_trace("fp32.json")
                    output = model(image)
                end_time = time.time()

                print("Running time for one sample is: {}".format(end_time-start_time))

                #print(output[0].cpu().numpy())

                output = output[0].cpu().numpy().astype(np.float16)

                transpose_forward = self.trainer.plans.get("transpose_forward")
                transpose_backward = self.trainer.plans.get("transpose_backward")
                assert transpose_forward == [0, 1, 2], "Unexpected transpose_forward {:}".format(transpose_forward)
                assert transpose_backward == [0, 1, 2], "Unexpected transpose_backward {:}".format(transpose_backward)

                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])
            if self.use_ipex and self.use_int8 and self.calibration:
                conf.save(self.configure_dir)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def benchmark(self, batchsize, steps, warmup_steps):
        with torch.no_grad():
            if self.use_ipex:
                import intel_pytorch_extension as ipex

            conf = None
            if self.use_ipex:
                if self.use_int8 and self.calibration:
                    # INT8 calibration
                    conf = ipex.AmpConf(torch.int8)
                elif self.use_int8:
                    # INT8 inference
                    conf = ipex.AmpConf(torch.int8, self.configure_dir)
                else:
                    # BF16 inference
                    conf = ipex.AmpConf(torch.bfloat16)

            model = self.trainer.network

            if self.use_autocast:
                # use autocast
                model.eval()
                #print(model)
                #onnx_path = "onnx_model_name.onnx"
                #torch.onnx.export(model, torch.randn(batchsize, 4, 224, 224, 160), onnx_path)
                #return
                if self.use_jit:
                    print("Benchmark enable autocast jit")
                    image = torch.randn(batchsize, 4, 224, 224, 160).float().to(self.device)
                    with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
                        #import pdb
                        #pdb.set_trace()
                        #pass
                        #model = torch.jit.trace(model, image, check_trace=False)
                        model = torch.jit.trace(model, image)
                    print(model)
                    #torch.onnx.export(model, torch.randn(batchsize, 4, 224, 224, 160), "/root/script_unset.onnx")
                    print(model.graph_for(image))
                    #return
                    total_time = 0
                    total_images = 0
                    #print("Enable jit autocast in benchmark dummy data")
                    for i in range(steps):
                        #print(i)
                        #dummy data
                        image = torch.randn(batchsize, 4, 224, 224, 160).float().to(self.device)
                        print("Processing image with shape = {:}".format(image.size()))
                        
                        start_time = time.time()
                        if i == 2:
                            with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                output = model(image)
                            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                            prof.export_chrome_trace("torch_throughput.json")
                        else:
                            output = model(image)
                        end_time = time.time()
                        if i > warmup_steps:
                            total_images += batchsize
                            total_time += end_time-start_time

                        print("Running time for one batchsize sample is: {}".format(end_time-start_time))
                    print("Finish the benchmark test with dummy data")
                    print("throughput is: {} samples/second".format(total_images/total_time))
                    return

                total_time = 0
                total_images = 0
                with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
                    print("Enable autocast in benchmark dummy data")
                    for i in range(steps):
                        #print(i)
                        #dummy data
                        image = torch.randn(batchsize, 4, 224, 224, 160).float().to(self.device)
                        print("Processing image with shape = {:}".format(image.size()))
                        
                        start_time = time.time()
                        if i == 2:
                            with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                output = model(image)
                            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                            prof.export_chrome_trace("torch_throughput.json")
                        else:
                            output = model(image)
                        end_time = time.time()
                        if i > warmup_steps:
                            total_images += batchsize
                            total_time += end_time-start_time

                        print("Running time for one batchsize sample is: {}".format(end_time-start_time))
                print("Finish the benchmark test with dummy data")
                print("throughput is: {} samples/second".format(total_images/total_time))
                return

            if self.use_ipex:
                model = model.to(device = ipex.DEVICE)
                model.eval()
            elif self.use_jit:
                print("fp32 jit path")
                model.eval()
                image = torch.randn(batchsize, 4, 224, 224, 160).float().to(self.device)
                model = torch.jit.trace(model, image)
                print(model.graph_for(image))
                #return
            else:
                pass
                #from torch.utils import mkldnn as mkldnn_utils
                #model = mkldnn_utils.to_mkldnn(model)

            total_time = 0
            total_images = 0
            for i in range(steps):
                #dummy data
                image = torch.randn(batchsize, 4, 224, 224, 160).float().to(self.device)
                print("Processing image with shape = {:}".format(image.size()))
                
                start_time = time.time()
                if self.use_ipex and self.use_int8 and self.calibration:
                    # INT8 calibration
                    with ipex.AutoMixPrecision(conf, running_mode="calibration"):
                        image = image.to(device = ipex.DEVICE)
                        output = model(image)
                elif self.use_ipex and self.use_int8:
                    # INT8 inference
                    image = image.to(device = ipex.DEVICE)
                    with ipex.AutoMixPrecision(conf, running_mode="inference"):
                        output = model(image)
                elif self.use_ipex:
                    # BF16 inference
                    image = image.to(device = ipex.DEVICE)
                    with ipex.AutoMixPrecision(conf, running_mode="inference"):
                        output = model(image)
                else:
                    #with torch.autograd.profiler.profile() as prof:
                    #    output = model(image)
                    #with open("fp32.prof", "w") as prof_f:
                    #    prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
                    #    prof.export_chrome_trace("fp32.json")
                    output = model(image)
                end_time = time.time()
                if i > warmup_steps:
                    total_images += batchsize
                    total_time += end_time-start_time

                print("Running time for one batchsize sample is: {}".format(end_time-start_time))
            print("Finish the benchmark test with dummy data")
            print("throughput is: {} samples/second".format(total_images/total_time))

def get_pytorch_sut(model_dir, preprocessed_data_dir, performance_count, folds=1, checkpoint_name="model_final_checkpoint", use_ipex=False,
                           use_int8=False, calibration=False, configure_dir="configure.json", use_autocast=False, use_jit=False):
    return _3DUNET_PyTorch_SUT(model_dir, preprocessed_data_dir, performance_count, folds, checkpoint_name, use_ipex,
                               use_int8, calibration, configure_dir, use_autocast, use_jit)
